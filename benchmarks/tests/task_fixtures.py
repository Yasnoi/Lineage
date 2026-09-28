"""Synthetic transport tasks, not registered project datasets."""
PROFILE = {"base_url": "https://model.invalid", "model_name": "fixture-model"}
def get_case(identifier):
    return {"case_id": identifier, "unit_id": identifier, "version": "fixture-v1", "turns": ["first", "second"], "dimensions": [], "mechanism": None, "settings_overrides": {}}
