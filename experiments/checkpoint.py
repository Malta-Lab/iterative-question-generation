import json
import os


class CheckpointManager:

    def __init__(self, path):
        self.path = path

        if os.path.exists(path):
            with open(path) as f:
                self.state = json.load(f)
        else:
            self.state = {"processed_ids": []}

    def is_done(self, claim_id):
        return claim_id in self.state["processed_ids"]

    def mark_done(self, claim_id):
        self.state["processed_ids"].append(claim_id)
        self.save()

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

        with open(self.path, "w") as f:
            json.dump(self.state, f, indent=2)