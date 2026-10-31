"""Instruction pool (10 verifiable, mutually satisfiable system-prompt instructions) and their verifiers.

Each instruction has a category: format / length (the "F/L" group) and persona / style (the "P/S" group).
Checks are deterministic and case-insensitive where a phrase is required, so that S_lower never conflicts with them.
"""
import re

INSTRUCTIONS = {
    "F_title": dict(cat="format", text="Give your response a title wrapped in double angular brackets, such as <<My Title>>."),
    "F_bullets": dict(cat="format", text="Format the body of your response as exactly 3 markdown bullet points that start with \"* \"."),
    "F_end": dict(cat="format", text="End your response with the exact phrase \"Any other questions?\" and write nothing after it."),
    "L_max": dict(cat="length", text="Keep your entire response under 100 words."),
    "P_ahoy": dict(cat="persona", text="You are a cheerful pirate: begin your response with the word \"Ahoy\"."),
    "P_name": dict(cat="persona", text="Introduce yourself by name as Captain Byte in your response."),
    "P_matey": dict(cat="persona", text="Address the user as \"matey\" at least once."),
    "S_lower": dict(cat="style", text="Write your entire response in lowercase letters only; no capital letters are allowed."),
    "S_nocomma": dict(cat="style", text="Do not use any commas anywhere in your response."),
    "S_keyword": dict(cat="style", text="Use the word \"remarkable\" at least twice in your response."),
}
IDS = list(INSTRUCTIONS)
GROUP = {k: ("FL" if v["cat"] in ("format", "length") else "PS") for k, v in INSTRUCTIONS.items()}
FILLER = "Be accurate and helpful."          # neutral, always-satisfied filler line (E4 length control)
PREAMBLE = "You are a helpful assistant. Follow all of these instructions in every response:"

_TITLE = re.compile(r"<<[^<>\n]+>>")
_BULLET = re.compile(r"^\s*[*\-•]\s+\S", re.M)


def _strip_title(t):
    """Drop a leading title line (<<..>> or markdown heading) so persona 'begin with' checks see the body."""
    lines = t.strip().split("\n")
    while lines and (not lines[0].strip() or _TITLE.fullmatch(lines[0].strip().strip("*#_ ")) or lines[0].lstrip().startswith("#")):
        lines.pop(0)
    return "\n".join(lines).strip()


def check(iid, text):
    t = text.strip()
    low = t.lower()
    if iid == "F_title":
        return bool(_TITLE.search(t))
    if iid == "F_bullets":
        return len(_BULLET.findall(t)) == 3
    if iid == "F_end":
        return re.sub(r"[\s*_\"'`]+$", "", low).endswith("any other questions?")
    if iid == "L_max":
        return len(t.split()) < 100
    if iid == "P_ahoy":
        body = _strip_title(t)
        if not body:
            body = t
        m = re.search(r"[a-z]+", body.lower())
        return bool(m) and m.group(0) == "ahoy"
    if iid == "P_name":
        return "captain byte" in low
    if iid == "P_matey":
        return re.search(r"\bmatey\b", low) is not None
    if iid == "S_lower":
        return t == low and any(c.isalpha() for c in t)
    if iid == "S_nocomma":
        return "," not in t and "，" not in t
    if iid == "S_keyword":
        return len(re.findall(r"\bremarkable\b", low)) >= 2
    raise KeyError(iid)


def system_prompt(lines, numbered=False):
    """lines: list of instruction ids (or 'FILLER'); returns the system prompt text.
    numbered=True renders a numbered list (robustness variant)."""
    body = [FILLER if i == "FILLER" else INSTRUCTIONS[i]["text"] for i in lines]
    if numbered:
        body = [f"{n}. {b}" for n, b in enumerate(body, 1)]
    return PREAMBLE + "\n" + "\n".join(body)


if __name__ == "__main__":       # sanity tests of the verifiers
    good = "<<sea tips>>\nahoy matey! i am captain byte.\n* a remarkable tip\n* another remarkable one\n* last\nany other questions?"
    for i in IDS:
        assert check(i, good), i
    bad = "Hello, I am an assistant.\n- one\n\nThanks."
    assert not any(check(i, bad) for i in IDS if i != "L_max"), [i for i in IDS if check(i, bad)]
    assert check("P_ahoy", "**<<Title>>**\n\nAhoy there") and check("F_end", "Bye. **Any other questions?**")
    print("verifier tests ok")
