DEFAULT_TEMPLATES = {
    "baseline": "Answer in one word or phrase only.\n\nQ: {question}\nA:",
    "apo_rewrite": (
        "Rewrite the user question into a concise, specific prompt that reduces ambiguity "
        "and includes constraints to avoid hallucinations. Output only the rewritten prompt.\n\nQ: {question}"
    ),
    "apo_target": "Answer in one word or phrase only.\n\n{optimized_prompt}",
    "cove": (
        "Answer in one word or phrase only. Verify the answer's factuality first.\n"
        "Question: {question}\n"
        "Prior answer: {prior_answer}\n"
        "Give only the corrected final answer."
    ),
    "self_correct": (
        "Answer in one word or phrase only. Fix any factual errors.\n"
        "Question: {question}\n"
        "Current answer: {prior_answer}\n"
        "Return only the corrected answer."
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
