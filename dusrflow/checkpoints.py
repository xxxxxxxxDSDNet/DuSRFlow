"""Checkpoint compatibility helpers for renamed DuSRFlow modules."""

from collections import OrderedDict


LEGACY_SR_PREFIXES = {
    "spynet.": "duflownet.",
    "corner_match.": "kf_matching.",
    "c2_matching_lv1.": "kf_dcn_warping_lv1.",
    "c2_matching_lv2.": "kf_dcn_warping_lv2.",
    "c2_matching_lv3.": "kf_dcn_warping_lv3.",
}


def migrate_legacy_sr_state_dict(state_dict):
    """Map legacy checkpoint prefixes to current module names."""
    migrated = OrderedDict()
    for key, value in state_dict.items():
        clean_key = key[7:] if key.startswith("module.") else key
        for legacy_prefix, release_prefix in LEGACY_SR_PREFIXES.items():
            if clean_key.startswith(legacy_prefix):
                clean_key = release_prefix + clean_key[len(legacy_prefix) :]
                break
        migrated[clean_key] = value
    return migrated
