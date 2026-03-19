import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if project_root not in sys.path:
    sys.path.append(project_root)
print(sys.path)

from pipeline.pipeline_factory import pipeline_rule_verdict
from pipeline.context import ClaimContext
from experiment_config import ExperimentConfig
from checkpoint import CheckpointManager
from result_writer import ResultWriter
from tqdm import tqdm
import json

def load_dataset(path):
    with open(path) as f:
        return json.load(f)


def run():

    pipeline = pipeline_rule_verdict()

    for dataset_cfg in ExperimentConfig.DATASETS:

        name = dataset_cfg["name"]
        path = dataset_cfg["path"]

        print(f"\nRunning dataset: {name}")

        data = load_dataset(path)

        ckpt = CheckpointManager(f"outputs/{name}.ckpt")
        writer = ResultWriter()

        total = len(data)

        done_ids = set(ckpt.state["processed_ids"])
        pbar = tqdm(total=total, desc=name)
        pbar.update(len(done_ids))        

        for i, item in enumerate(data):

            claim_id = i  # ou hash do claim

            if ckpt.is_done(claim_id):
                continue

            context = ClaimContext(
                claim_id=claim_id,
                claim_text=item["claim"],
                claim_date=item.get("claim_date")
            )

            result = pipeline.run(context)

            writer.add(item, result)
            ckpt.mark_done(claim_id)

            pbar.update(1)

            if i % 10 == 0:
                writer.save(f"outputs/{name}.json")

        writer.save(f"outputs/{name}.json")
        pbar.close()

if __name__ == "__main__":
    run()