def detect_mode(query: str) -> str:
    q = query.lower()

    if "summarize" in q or "summary" in q:
        return "summarize"
    if "compare" in q:
        return "compare"
    if "action items" in q or "tasks" in q:
        return "extract_actions"
    return "qa"