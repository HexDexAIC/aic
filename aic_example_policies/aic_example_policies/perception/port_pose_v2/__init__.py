"""Phase-2 perception: landmark → 6D pose → tracked pose.

Public API:
    PortPoseSource   — controller-facing aggregator (Phase 4)
    PnPPortPose       — single-view PnP estimator
    SE3Tracker        — SE(3) smoothing + outlier-reject + coasting
"""
