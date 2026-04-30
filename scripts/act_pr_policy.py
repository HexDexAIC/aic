"""ACT-PR: ACT subclass with an auxiliary port-pose head.

Architecture choice (per the Phase 2 plan):

  images + state
    → existing ACT visual/state encoders (unchanged)
    → fused encoder memory (sequence of D-dim tokens)
    → mean-pool over sequence dim → fused observation summary
    → port_pose_head(summary) → 9-D port pose
    AND
    → existing ACT decoder/action head → 9-D port-frame residual chunk

Training loss:

  loss = action_l1 + kl_weight * kl_div + lambda_port * port_pose_l1

The port-pose head is taken off the fused observation embedding *just before*
the ACT decoder/action head — same representation the action head sees, so
forcing port-pose information into that representation directly improves
the action head's input. (Not on raw ResNet features, not as a decoder
token; both alternatives are heavier wiring with no clear benefit.)

At deploy time:
  - ACTPRPolicy.select_action(batch) returns BOTH:
      action_residual_chunk[t]       (1, 9)
      predicted_port_pose            (1, 9)  =  [port_xyz, port_rot6]
  - The deployment wrapper composes:
      pose_target_abs = port_pose ∘ residual_action
    and publishes the resulting Cartesian pose target.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACT, ACTPolicy

PORT_POSE_KEY = "observation.port_pose_gt"


@dataclass
class ACTPRConfig(ACTConfig):
    """Same config space as ACTConfig, with the auxiliary loss weight added.

    `port_pose_loss_weight` is the lambda for the aux loss term. Default 1.0
    per the Phase 2 plan; if the aux loss dominates the action loss by >3×
    after the first 1-2k steps, drop to 0.1.
    """

    port_pose_loss_weight: float = 1.0


class ACTPR(ACT):
    """ACT model that ALSO returns the fused encoder summary.

    Single line of override: forward() captures encoder_out, mean-pools it
    over the sequence dimension, and returns it alongside the usual
    (actions, latent_pdf_params) tuple.
    """

    def forward(
        self, batch: dict[str, Tensor]
    ) -> tuple[Tensor, tuple[Tensor | None, Tensor | None], Tensor]:
        # Replicate the parent's forward up to encoder_out, which is needed
        # for the port head, then continue as usual. We can't simply call
        # super().forward() because it doesn't expose encoder_out.
        #
        # The body below is a verbatim copy of ACT.forward (lerobot 0.5.1)
        # with one extra line that captures the mean-pooled encoder_out.
        import einops

        if self.config.use_vae and self.training:
            assert ACTION in batch
        batch_size = (
            batch[OBS_IMAGES][0].shape[0] if OBS_IMAGES in batch else batch[OBS_STATE].shape[0]
        )

        if self.config.use_vae and ACTION in batch and self.training:
            cls_embed = einops.repeat(
                self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=batch_size
            )
            if self.config.robot_state_feature:
                robot_state_embed = self.vae_encoder_robot_state_input_proj(batch[OBS_STATE]).unsqueeze(1)
            action_embed = self.vae_encoder_action_input_proj(batch[ACTION])
            if self.config.robot_state_feature:
                vae_encoder_input = [cls_embed, robot_state_embed, action_embed]
            else:
                vae_encoder_input = [cls_embed, action_embed]
            vae_encoder_input = torch.cat(vae_encoder_input, axis=1)
            pos_embed = self.vae_encoder_pos_enc.clone().detach()
            cls_joint_is_pad = torch.full(
                (batch_size, 2 if self.config.robot_state_feature else 1),
                False, device=batch[OBS_STATE].device,
            )
            key_padding_mask = torch.cat([cls_joint_is_pad, batch["action_is_pad"]], axis=1)
            cls_token_out = self.vae_encoder(
                vae_encoder_input.permute(1, 0, 2),
                pos_embed=pos_embed.permute(1, 0, 2),
                key_padding_mask=key_padding_mask,
            )[0]
            latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
            mu = latent_pdf_params[:, : self.config.latent_dim]
            log_sigma_x2 = latent_pdf_params[:, self.config.latent_dim :]
            latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
        else:
            mu = log_sigma_x2 = None
            latent_sample = torch.zeros(
                [batch_size, self.config.latent_dim], dtype=torch.float32
            ).to(batch[OBS_STATE].device)

        encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
        encoder_in_pos_embed = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))
        if self.config.robot_state_feature:
            encoder_in_tokens.append(self.encoder_robot_state_input_proj(batch[OBS_STATE]))
        if self.config.env_state_feature:
            encoder_in_tokens.append(self.encoder_env_state_input_proj(batch[OBS_ENV_STATE]))

        if self.config.image_features:
            for img in batch[OBS_IMAGES]:
                cam_features = self.backbone(img)["feature_map"]
                cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                cam_features = self.encoder_img_feat_input_proj(cam_features)
                cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
                cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")
                encoder_in_tokens.extend(list(cam_features))
                encoder_in_pos_embed.extend(list(cam_pos_embed))

        encoder_in_tokens = torch.stack(encoder_in_tokens, axis=0)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)

        encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)
        # NEW: capture the fused observation summary for the port-pose head.
        # encoder_out is (S, B, D); mean-pool over S → (B, D).
        encoder_summary = encoder_out.mean(dim=0)

        decoder_in = torch.zeros(
            (self.config.chunk_size, batch_size, self.config.dim_model),
            dtype=encoder_in_pos_embed.dtype,
            device=encoder_in_pos_embed.device,
        )
        decoder_out = self.decoder(
            decoder_in, encoder_out,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )
        decoder_out = decoder_out.transpose(0, 1)
        actions = self.action_head(decoder_out)

        return actions, (mu, log_sigma_x2), encoder_summary


class ACTPRPolicy(ACTPolicy):
    """ACTPolicy with a port-pose auxiliary head off the encoder summary."""

    name = "act_pr"

    def __init__(self, config: ACTPRConfig, **kwargs):
        super().__init__(config, **kwargs)
        # Replace the inner model with our subclassed version.
        self.model = ACTPR(config)
        # Aux head: takes the (B, D) fused obs summary, predicts (B, 9) port pose
        # in NORMALIZED space (so it shares the L1 loss scale with the action head).
        self.port_pose_head = nn.Sequential(
            nn.LayerNorm(config.dim_model),
            nn.Linear(config.dim_model, config.dim_model),
            nn.ReLU(),
            nn.Linear(config.dim_model, 9),
        )
        # Cache of the most recently predicted port pose (normalized), in case
        # the deploy wrapper wants to read it without re-running inference.
        self._last_pred_port_pose_norm: Tensor | None = None

    # ── training forward ──────────────────────────────────────────────
    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        actions_hat, (mu_hat, log_sigma_x2_hat), encoder_summary = self.model(batch)

        # Action L1 (same as parent ACTPolicy.forward).
        l1_loss = (
            F.l1_loss(batch[ACTION], actions_hat, reduction="none")
            * ~batch["action_is_pad"].unsqueeze(-1)
        ).mean()
        loss_dict = {"l1_loss": l1_loss.item()}

        # Aux: port pose head L1.
        port_pred = self.port_pose_head(encoder_summary)  # (B, 9), normalized
        # Target is in batch[PORT_POSE_KEY] (B, 9); already normalized by the
        # preprocessor pipeline at training time.
        port_target = batch[PORT_POSE_KEY]
        # If the dataset puts a sequence dim (B, 1, 9) we squeeze.
        if port_target.dim() == 3 and port_target.shape[1] == 1:
            port_target = port_target.squeeze(1)
        port_loss = F.l1_loss(port_target, port_pred)
        loss_dict["port_pose_loss"] = port_loss.item()

        # Total loss = action + KL (if VAE) + lambda * port_pose
        if self.config.use_vae:
            mean_kld = (
                (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - (log_sigma_x2_hat).exp()))
                .sum(-1)
                .mean()
            )
            loss_dict["kld_loss"] = mean_kld.item()
            loss = l1_loss + mean_kld * self.config.kl_weight
        else:
            loss = l1_loss

        loss = loss + self.config.port_pose_loss_weight * port_loss
        loss_dict["loss"] = loss.item()
        return loss, loss_dict

    # ── inference (override to also return port pose) ─────────────────
    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Returns the next action of shape (B, 9) — port-frame residual.

        Side-effect: caches the most recent predicted port pose (normalized)
        in self._last_pred_port_pose_norm for the deploy wrapper to consume.
        """
        self.eval()

        # If chunk queue is empty, repopulate via predict_action_chunk (which
        # invokes self.model and captures encoder_summary internally).
        if self.config.temporal_ensemble_coeff is not None:
            actions, encoder_summary = self.predict_action_chunk_with_summary(batch)
            action = self.temporal_ensembler.update(actions)
            self._last_pred_port_pose_norm = self.port_pose_head(encoder_summary).detach()
            return action

        if len(self._action_queue) == 0:
            actions, encoder_summary = self.predict_action_chunk_with_summary(batch)
            actions = actions[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
            self._last_pred_port_pose_norm = self.port_pose_head(encoder_summary).detach()
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk_with_summary(self, batch: dict[str, Tensor]):
        """Like predict_action_chunk but returns (actions, encoder_summary)."""
        self.eval()
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]
        actions, _, encoder_summary = self.model(batch)
        return actions, encoder_summary

    @torch.no_grad()
    def get_last_predicted_port_pose_norm(self) -> Tensor | None:
        """Returns the most recent normalized port-pose prediction.

        Caller is responsible for un-normalizing using the postprocessor's
        port_pose_gt stats. Returned shape (B, 9) — [port_xyz, port_rot6].
        """
        return self._last_pred_port_pose_norm
