DEFAULT_TEMPLATES = {
    "baseline": "Q: {question}\nA:",
    "apo_rewrite": (
        "Rewrite the user question into a concise, specific prompt that reduces ambiguity "
        "and includes constraints to avoid hallucinations. Output only the rewritten prompt.\n\nQ: {question}"
    ),
    "apo_target": "Use this optimized prompt:\n\n{optimized_prompt}\n\nAnswer succinctly and cite key facts.",
   "cove": (
    "Fact-checking mode.\n"
    "Q: {question}\n"
    "Draft: {prior_answer}\n\n"
    "Steps:\n"
    "1. Extract factual claims clearly.\n"
    "2. Check each claim independently for accuracy.\n"
    "3. Produce a corrected final answer.\n\n"
    "Output JSON:\n"
    "{{\n"
    "  \"claims\": [\n"
    "    {{\"text\": \"...\", \"verdict\": \"supported|unsupported|uncertain\"}}\n"
    "  ],\n"
    "  \"final_answer\": \"...\"\n"
    "}}"
),

    "self_correct": (
        "Revise only factual errors at temperature 0.\n"
        "Question: {question}\n"
        "Current answer: {prior_answer}\n"
        "Return a corrected final answer only."
    ),
    "judge": (
        "You are a strict fact-checking judge.\n"
        "Question: {question}\n"
        "Answer: {answer}\n"
        "Return exactly: PASS <p=0.80> or FAIL <p=0.20>."
    ),
    "gate_judge": (
        "Judge correctness for early exit.\n"
        "Question: {question}\n"
        "Answer: {answer}\n"
        "Return PASS <p=...> or FAIL <p=...>."
    )
}


