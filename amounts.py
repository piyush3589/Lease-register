"""
amounts.py
Best-effort conversion of an amount as written into a number.

Why this exists: the extractor deliberately keeps amounts as strings, because
"Rs. 65,000" and "Rs. 65,000/-" and "INR 65000" are all valid and the model's
job is to copy, not to compute. But the register's whole purpose is a rent roll
that someone sums in a spreadsheet, so somewhere the string has to become a
number.

So the original string is never touched. This produces a separate, clearly
derived value that is null whenever the text is not confidently a number. A
wrong rent quietly summed into a total is worse than a blank cell, so every
ambiguous case returns None rather than guessing.

Not a general money parser. It handles Indian lakh/crore grouping, which is the
format these leases actually use, and refuses everything else.
"""

import re

# Rs. / Rs / INR / the rupee sign / the word Rupees.
_CURRENCY = r"(?:rs\.?|inr|₹|rupees)"
_LAKH_CRORE = r"(?:lakhs?|crores?)"

# A number as Indians actually write it: 65,000 / 3,25,000 / 1,20,00,000.
# Indian grouping puts a comma every two digits after the last three, so the
# grouping cannot be validated positionally -- it is simply stripped.
_NUMBER = r"\d{1,3}(?:,\d{2,3})*(?:\.\d+)?|\d+(?:\.\d+)?"

_MULTIPLIER = {"lakh": 100_000, "lakhs": 100_000, "crore": 10_000_000, "crores": 10_000_000}

_CURRENCY_AMOUNT = re.compile(
    rf"{_CURRENCY}\s*:?\s*({_NUMBER})", re.IGNORECASE
)
_LAKH_CRORE_AMOUNT = re.compile(
    rf"({_NUMBER})\s*({_LAKH_CRORE})\b", re.IGNORECASE
)
_PURE_NUMBER = re.compile(rf"^\s*(?:{_NUMBER})\s*(?:/-)?\s*$")

_MULTIPLIER = {
    "lakh": 100_000,
    "lakhs": 100_000,
    "crore": 10_000_000,
    "crores": 10_000_000,
}


def _strip_commas(match: str) -> float:
    return float(match.replace(",", ""))


def parse_amount(text: str | None) -> float | None:
    """Return a numeric amount, or None if the text is not confidently money.

    Returns None rather than guessing for anything ambiguous, including:
      - no numbers at all
      - units that are not money ("11 months", "4500 sq ft")
      - free text with no currency marker ("to be discussed separately")

    Accepts Indian lakh/crore grouping ("Rs. 3,25,000") and lakh/crore words
    ("1.5 lakhs"), and takes the FIRST amount when a lease states both digits
    and the amount in words ("Rs. 4,20,000 (Rupees Four Lakh Twenty Thousand)")
    rather than summing them.
    """
    if not text or not isinstance(text, str):
        return None

    stripped = text.strip()
    if not stripped:
        return None

    # Explicit currency wins, and its number is authoritative. A frequency
    # qualifier like "per month" is expected here and must not disqualify it.
    currency = _CURRENCY_AMOUNT.search(stripped)
    if currency:
        return _strip_commas(currency.group(1))

    # "1.5 lakhs" / "2 crores"
    lakh = _LAKH_CRORE_AMOUNT.search(stripped)
    if lakh:
        base = _strip_commas(lakh.group(1))
        multiplier = _MULTIPLIER[lakh.group(2).lower()]
        return base * multiplier

    # A bare number, but only if the string is nothing but a number. This single
    # rule is what keeps "11 months" and "36 months" from being read as money,
    # without needing a blocklist of unit words -- and without that blocklist
    # wrongly rejecting "Rs. 65,000 per month", the most common rent phrasing.
    if _PURE_NUMBER.match(stripped):
        return _strip_commas(stripped.split("/-")[0].strip())

    return None


def _indian_group(value: int) -> str:
    """Group digits the Indian way: last three, then pairs. 325000 -> 3,25,000.

    Python's `,` format is Western (325,000), which would put a number in the
    register that no one on the team recognises as their own rent.
    """
    digits = str(value)
    if len(digits) <= 3:
        return digits

    head, tail = digits[:-3], digits[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    return ",".join(groups + [tail])


def format_amount(amount: float | None, currency: str = "Rs.") -> str:
    """Render a parsed amount the way these leases write it, for display only."""
    if amount is None:
        return ""
    amount = float(amount)
    if amount.is_integer():
        return f"{currency} {_indian_group(int(amount))}"
    whole, _, fraction = f"{amount:.2f}".partition(".")
    return f"{currency} {_indian_group(int(whole))}.{fraction}"
