import sys
import os
import json
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if project_root not in sys.path:
    sys.path.append(project_root)

from pipeline.pipeline_factory import pipeline_rule_verdict
from pipeline.context import ClaimContext
from experiment_config import ExperimentConfig
from checkpoint import CheckpointManager
from result_writer import ResultWriter


# 🔧 CONFIG
VERBOSE = True          # liga/desliga debug
PRINT_EVERY = 1         # imprime a cada N exemplos


def load_dataset(path):
    with open(path) as f:
        return json.load(f)


def print_step_debug(step, step_id):
    print(f"\n   🔹 Step {step_id+1}")
    print(f"   Q: {step.get('question')}")
    print(f"   A: {step.get('answer')[:200]}...")

    if step.get("stance"):
        print(f"   Stance: {step.get('stance')}")

    if step.get("evidence"):
        top_ev = step["evidence"][0] if isinstance(step["evidence"], list) else step["evidence"]
        text = top_ev.get("text") if isinstance(top_ev, dict) else str(top_ev)
        print(f"   Evidence (top): {text[:200]}...")


def run():

    pipeline = pipeline_rule_verdict()  # depois podemos passar verbose=True

    for dataset_cfg in ExperimentConfig.DATASETS:

        name = dataset_cfg["name"]
        path = dataset_cfg["path"]

        print(f"\n🚀 Running dataset: {name}")

        data = load_dataset(path)

        ckpt = CheckpointManager(f"outputs/{name}.ckpt")
        writer = ResultWriter()

        total = len(data)

        done_ids = set(ckpt.state["processed_ids"])
        pbar = tqdm(total=total, desc=name)
        pbar.update(len(done_ids))

        for i, item in enumerate(data):

            claim_id = i

            if ckpt.is_done(claim_id):
                continue

            # 🔥 HEADER DO EXEMPLO
            if VERBOSE and i % PRINT_EVERY == 0:
                print("\n" + "="*80)
                print(f"🧾 CLAIM {claim_id}")
                print(item["claim"])

            context = ClaimContext(
                claim_id=claim_id,
                claim_text=item["claim"],
                claim_date=item.get("claim_date")
            )

            result = pipeline.run(context)

            # 🔥 DEBUG DETALHADO
            if VERBOSE and i % PRINT_EVERY == 0:

                steps = getattr(result, "steps", [])

                for j, step in enumerate(steps):
                    print_step_debug(step, j)

                print(f"\n   ✅ FINAL VERDICT: {result.verdict}")

            writer.add(item, result)
            ckpt.mark_done(claim_id)

            pbar.update(1)

            if i % 10 == 0:
                writer.save(f"outputs/{name}.json")

        writer.save(f"outputs/{name}.json")
        pbar.close()


if __name__ == "__main__":
    run()