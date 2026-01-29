import re

DEPARTMENT_KEYWORDS = {
    "finance": [
        "refund", "invoice", "payment", "reimbursement", "expense", "tax", "salary", "payroll", "billing"
    ],
    "hr": [
        "remote", "holiday", "leave", "vacation", "benefits", "policy", "work from home", "wfh", "sick", "hr"
    ],
    "it": [
        "vpn", "password", "email", "laptop", "wifi", "network", "login", "access", "2fa", "mfa", "software"
    ],
}

def route_department(user_message: str) -> str:
    q = user_message.lower()

    # simple scoring
    scores = {dept: 0 for dept in DEPARTMENT_KEYWORDS}
    for dept, keywords in DEPARTMENT_KEYWORDS.items():
        for kw in keywords:
            if kw in q:
                scores[dept] += 1

    # pick best match if any
    best_dept = max(scores, key=scores.get)
    if scores[best_dept] == 0:
        return "general"

    return best_dept
