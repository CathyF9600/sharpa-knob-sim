"""JSONL transitions with physics metadata kept outside model experience."""
import json
import math


def json_value(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {k: json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class TransitionWriter:
    def __init__(self, path, run_header):
        self.file = open(path, "x", encoding="utf-8")
        self.file.write(json.dumps({"type": "header", **run_header}, allow_nan=False) + "\n")

    def write_batch(self, transition):
        t = json_value(transition)
        for i, condition in enumerate(t["condition_id"]):
            before = {k: v[i] for k, v in t["before"].items()}
            after = {k: v[i] for k, v in t["after"].items()}
            record = {
                "type": "transition", "env_id": i,
                "metadata": {"condition_id": condition, "trial_id": t["trial_id"][i],
                             "physical_settings": t["physical_settings_metadata_only"][i]},
                "experience": {
                    "before": before, "after": after,
                    "proposed_action": t["proposed_action"][i],
                    "issued_target_rad": t["issued_target_rad"][i],
                    "duration_s": t["duration_s"],
                    "terminated": t["terminated"][i], "truncated": t["truncated"][i],
                },
            }
            self.file.write(json.dumps(record, allow_nan=False) + "\n")
        self.file.flush()

    def close(self):
        self.file.close()
