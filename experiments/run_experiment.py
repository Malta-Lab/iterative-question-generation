import sys
import os
import json
from datetime import datetime
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if project_root not in sys.path:
    sys.path.append(project_root)

from pipeline.pipeline_factory import averitec_pipeline
from pipeline.context import ClaimContext
from experiment_config import ExperimentConfig
from checkpoint import CheckpointManager
from result_writer import ResultWriter


# 🔧 CONFIG
VERBOSE = True
PRINT_EVERY = 1
SAVE_EVERY = 10   # ⬅️ salva a cada N exemplos


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

    # 🧪 cria diretório único do experimento
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join("outputs", timestamp)
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n📁 Run directory: {run_dir}")

    # 🔧 pipeline
    pipeline = averitec_pipeline()

    # 💾 salva config do experimento
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({
            "datasets": ExperimentConfig.DATASETS,
            "model": os.getenv("OLLAMA_MODEL"),
            "temperature": os.getenv("LLM_TEMPERATURE"),
        }, f, indent=2)

    for dataset_cfg in ExperimentConfig.DATASETS:

        name = dataset_cfg["name"]
        path = dataset_cfg["path"]

        print(f"\n🚀 Running dataset: {name}")

        data = load_dataset(path)

        ckpt = CheckpointManager(os.path.join(run_dir, f"{name}.ckpt"))
        writer = ResultWriter()

        total = len(data)

        done_ids = set(ckpt.state["processed_ids"])
        pbar = tqdm(total=total, desc=name)
        pbar.update(len(done_ids))

        for i, item in enumerate(data):

            claim_id = i

            if ckpt.is_done(claim_id):
                continue

            if VERBOSE and i % PRINT_EVERY == 0:
                print("\n" + "="*80)
                print(f"🧾 CLAIM {claim_id}")
                print(item["claim"])

            context = ClaimContext(
                claim_id=claim_id,
                claim_text=item["claim"],
                claim_date=item.get("claim_date"),
                speaker=item.get("speaker")
            )

            result = pipeline.run(context)

            # 🧠 debug
            if VERBOSE and i % PRINT_EVERY == 0:

                steps = getattr(result, "steps", [])

                for j, step in enumerate(steps):
                    print_step_debug(step, j)

                gt = item.get("label")

                correct = (str(result.verdict).lower() == str(gt).lower())
                status = "✅ CORRECT" if correct else "❌ WRONG"

                print(f"\n   {status}")
                print(f"   PRED: {result.verdict}")
                print(f"   GT  : {gt}")

            # 💾 salva resultado
            writer.add(item, result)
            ckpt.mark_done(claim_id)

            # 💾 salvamento incremental
            if i % SAVE_EVERY == 0:
                writer.save(os.path.join(run_dir, f"{name}.json"))

            pbar.update(1)

        # 💾 save final
        writer.save(os.path.join(run_dir, f"{name}.json"))
        pbar.close()


if __name__ == "__main__":
    run()