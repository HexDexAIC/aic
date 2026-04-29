"""Save the SDF-derived calibration values to ~/aic_visible_mouth_calib.json
(overwrites the buggy server-side solve result).
"""
import json
from pathlib import Path

result = {
    "source": "simulator SDF (NIC Card Mount/model.sdf, sfp_port_0_link_entrance)",
    "T_canonical_to_visible_mouth": {
        "dx_mm": 0.0,
        "dy_mm": 0.0,
        "dz_mm": -45.8,
        "convention": "translate along port +z; sim defines this as link 'sfp_port_0_link_entrance' relative to 'sfp_port_0_link'",
    },
    "T_target_to_distractor": {
        "dx_mm": -23.2,
        "dy_mm": 0.0,
        "dz_mm": 0.0,
        "convention": "translate along port +x; from sfp_port_0_link to sfp_port_1_link in nic_card_link frame",
    },
    "rectangle_at_mouth": {
        "width_mm": 13.7,
        "height_mm": 8.5,
        "note": "SFP MSA spec port-mouth opening",
    },
    "validation": {
        "n_user_clicked_frames": 16,
        "median_corner_err_vs_user_clicks_px": 8.11,
        "click_calibration_independent_estimate_dz_mm": -47.21,
        "agreement_with_sdf_mm": 1.4,
        "interpretation": "User clicks confirm SDF entrance offset within click noise",
    },
}
out = Path.home() / "aic_visible_mouth_calib.json"
out.write_text(json.dumps(result, indent=2))
print(f"Wrote {out}")
print(json.dumps(result, indent=2))
