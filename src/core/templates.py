DEFAULT_TEMPLATES = {
    "baseline": (
        "Q: {question}\n"
        "A:"
    ),

    "apo_rewrite": (
        "Task: Rewrite the user question.\n"
        "Steps:\n"
        "1. Make it precise and unambiguous.\n"
        "2. Add limits or constraints to reduce hallucination.\n"
        "3. Keep it short and clear.\n"
        "4. Output only the rewritten question.\n\n"
        "Input Question:\n{question}\n\n"
        "Rewritten Question:"
    ),

    "apo_target": (
        "Task: Answer using the optimized prompt.\n\n"
        "Optimized Prompt:\n{optimized_prompt}\n\n"
        "Rules:\n"
        "1. Give a short, direct answer.\n"
        "2. Use only verified facts.\n"
        "3. Cite key sources or evidence.\n"
        "4. Do not add extra text outside the answer.\n\n"
        "Final Answer:"
    ),

    "cove": (
        "Task: Fact-check the draft answer.\n\n"
        "Question:\n{question}\n"
        "Draft Answer:\n{prior_answer}\n\n"
        "Steps:\n"
        "1. Extract all factual claims clearly.\n"
        "2. Check each claim independently.\n"
        "3. Mark each claim as Supported, Unsupported, or Uncertain.\n"
        "4. Correct mistakes in the draft.\n"
        "5. Give a final corrected answer.\n\n"
        "Final Answer:"
    ),

    "self_correct": (
        "Task: Correct the answer.\n\n"
        "Question:\n{question}\n"
        "Current Answer:\n{prior_answer}\n\n"
        "Rules:\n"
        "1. Fix factual errors only.\n"
        "2. Do not change style or structure.\n"
        "3. Do not add new content.\n"
        "4. Use temperature 0 (deterministic).\n\n"
        "Corrected Answer:"
    ),

    "judge": (
        "Task: Judge factual correctness.\n\n"
        "Question:\n{question}\n"
        "Answer:\n{answer}\n\n"
        "Rules:\n"
        "1. Check if the answer is factually correct.\n"
        "2. Do not explain reasoning.\n"
        "3. Output only one of the following:\n"
        "   - PASS <p=0.80>\n"
        "   - FAIL <p=0.20>"
    ),

    "gate_judge": (
        "Task: Judge correctness for early exit.\n\n"
        "Question:\n{question}\n"
        "Answer:\n{answer}\n\n"
        "Rules:\n"
        "1. Evaluate factual correctness.\n"
        "2. Output only:\n"
        "   - PASS <p=...>\n"
        "   - FAIL <p=...>"
    ),
}
