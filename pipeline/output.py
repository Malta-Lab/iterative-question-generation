def output(result):
    print("\n" + "="*90)
    print("🧠 CLAIM")
    print("="*90)
    print(result.claim)


    print("\n" + "="*90)
    print("🔍 TOP EVIDENCE")
    print("="*90)

    for i, evidence in enumerate(result.evidence, 1):
        print(f"{i:02d}. {evidence}")


    print("\n" + "="*90)
    print("⚖️ STANCE CLASSIFICATION")
    print("="*90)

    # contadores
    support = 0
    refute = 0
    nee = 0

    for i, (evidence, stance) in enumerate(result.stances, 1):

        label = stance.lower()

        # normalização (importante!)
        if label in ["support", "supported"]:
            label_fmt = "SUPPORTED"
            support += 1
            icon = "✅"
        elif label in ["refute", "refuted"]:
            label_fmt = "REFUTED"
            refute += 1
            icon = "❌"
        else:
            label_fmt = "NOT ENOUGH EVIDENCE"
            nee += 1
            icon = "⚪"

        print(f"{i:02d}. {icon} [{label_fmt}]")
        print(f"    ↳ {evidence}\n")


    print("\n" + "="*90)
    print("📊 STANCE SUMMARY")
    print("="*90)
    print(f"✅ Supported: {support}")
    print(f"❌ Refuted: {refute}")
    print(f"⚪ Not Enough Evidence: {nee}")


    print("\n" + "="*90)
    print("❓ QA PAIRS")
    print("="*90)

    for i, qa in enumerate(result.qa_pairs, 1):
        print(f"{i:02d}. Q: {qa['question']}")
        print(f"    A: {qa['answer']}\n")


    print("\n" + "="*90)
    print("🏁 FINAL VERDICT")
    print("="*90)
    print(f"➡️  {result.verdict.upper()}")


    print("\n" + "="*90)
    print("📝 JUSTIFICATION")
    print("="*90)
    print(result.justification)