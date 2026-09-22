"""Monetary parsing and normalisation.

Extraction returns whatever figure the model read off the page, at whatever
scale the page used. The first demonstration dataset therefore recorded
Alvaro Morata's purchase option as `amount=8, currency=EUR` from the span
"a buy option of EUR 8 million", and Alexander Sorloth's add-ons as
`amount=2` from "a further EUR 2m payable in add-ons". Both are off by six
orders of magnitude, and both look plausible in a spreadsheet.

The evidence span is the authority. A recorded amount is accepted only if some
monetary expression in its own span agrees with it at SOME scale; the value is
then rewritten to full units. If the span carries no figure at all, the amount
did not come from the quotation and is not admissible.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

MULTIPLIERS = {
    "m": 1e6, "mln": 1e6, "mio": 1e6, "mn": 1e6, "million": 1e6, "millions": 1e6,
    "milioni": 1e6, "milione": 1e6, "millones": 1e6, "millon": 1e6, "millón": 1e6,
    "milhoes": 1e6, "milhões": 1e6, "milhao": 1e6, "mil": 1e6, "mio.": 1e6,
    "k": 1e3, "mila": 1e3, "thousand": 1e3, "tys": 1e3, "bin": 1e3,
    "bn": 1e9, "billion": 1e9, "miliardi": 1e9, "milliarden": 1e9, "mld": 1e9,
}

CURRENCIES = {
    "€": "EUR", "eur": "EUR", "euro": "EUR", "euros": "EUR", "eur.": "EUR",
    "£": "GBP", "gbp": "GBP", "pound": "GBP", "pounds": "GBP", "sterling": "GBP",
    "$": "USD", "usd": "USD", "dollar": "USD", "dollars": "USD",
    "₺": "TRY", "try": "TRY", "tl": "TRY", "lira": "TRY",
    "chf": "CHF", "fr.": "CHF",
}

_SYMBOLS = "€£$₺"
_MULT_ALT = "|".join(sorted((re.escape(k) for k in MULTIPLIERS), key=len, reverse=True))
_CUR_WORDS = "|".join(sorted((re.escape(k) for k in CURRENCIES if k.isalpha()),
                             key=len, reverse=True))

MONEY_RE = re.compile(
    rf"(?P<sym>[{_SYMBOLS}])?\s*"
    rf"(?:(?P<curw>{_CUR_WORDS})\s*)?"
    r"(?P<num>\d[\d.,]*)"
    rf"\s*(?P<mult>{_MULT_ALT})?\b"
    rf"\s*(?P<curw2>{_CUR_WORDS})?",
    re.IGNORECASE)


def parse_number(raw: str, scaled: bool = False) -> float | None:
    """A numeral written in any of the conventions our sources use.

    '13,5' is thirteen and a half (Italian), '12,200,000' is twelve million two
    hundred thousand (English), '1.200.000' is the same in German or Portuguese.
    Getting this wrong is how a 13.5 becomes a 135.

    `scaled` says a multiplier word follows, which settles the one genuinely
    ambiguous case: in "EUR 2.825m" the dot is a decimal point, because nobody
    writes 2,825 million rather than 2.8 billion. Read as a thousands
    separator it made Weghorst's EUR 2.825m termination fee 2.8 billion.
    """
    s = (raw or "").strip().rstrip(".,")
    if not s or not s[0].isdigit():
        return None
    if "," in s and "." in s:
        dec = "," if s.rindex(",") > s.rindex(".") else "."
        s = s.replace("." if dec == "," else ",", "").replace(dec, ".")
    elif "," in s:
        parts = s.split(",")
        s = s.replace(",", "." if len(parts) == 2 and len(parts[1]) in (1, 2) else "")
    elif "." in s:
        parts = s.split(".")
        if len(parts) > 2:                       # 1.200.000
            s = s.replace(".", "")
        elif len(parts[1]) == 3 and parts[0] not in ("0", "") and not scaled:   # 1.200
            s = s.replace(".", "")
    try:
        return float(s)
    except ValueError:
        return None


@dataclass(frozen=True)
class Money:
    value: float
    currency: str | None
    text: str

    def __repr__(self) -> str:
        return f"Money({self.value:,.0f} {self.currency or '?'} <- {self.text!r})"


def find_all(text: str) -> list[Money]:
    """Every monetary expression in a piece of text, in full units."""
    out: list[Money] = []
    for m in MONEY_RE.finditer(text or ""):
        mult = (m.group("mult") or "").lower()
        num = parse_number(m.group("num"), scaled=mult in MULTIPLIERS)
        if num is None:
            continue
        # A bare "m" only multiplies when it is a real suffix, not the start of
        # another word; the regex word boundary handles that.
        scale = MULTIPLIERS.get(mult, 1.0)
        cur = None
        for g in ("sym", "curw", "curw2"):
            v = m.group(g)
            if v:
                cur = CURRENCIES.get(v.lower())
                if cur:
                    break
        out.append(Money(num * scale, cur, m.group(0).strip()))
    return out


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= max(1.0, abs(b) * 0.02)


@dataclass
class Normalised:
    amount: float | None
    currency: str | None
    raw_text: str | None
    scale_applied: float = 1.0
    status: str = "ok"          # ok | rescaled | no_figure_in_span | contradicts_span
    note: str = ""

    @property
    def ok(self) -> bool:
        return self.status in ("ok", "rescaled")


def normalise_amount(amount, currency, span: str) -> Normalised:
    """Reconcile a recorded amount with the text it claims to quote.

    Returns the amount in full units when the span agrees at some scale, and
    refuses it otherwise. Refusal is the point: an amount that cannot be found
    in its own quotation was not read off that quotation.
    """
    try:
        a = float(amount)
    except (TypeError, ValueError):
        return Normalised(None, currency, None, status="ok", note="no amount recorded")

    found = find_all(span or "")
    if not found:
        return Normalised(None, currency, None, status="no_figure_in_span",
                          note="evidence span contains no monetary figure")

    for scale in (1.0, 1e6, 1e3, 1e9, 1e-6, 1e-3):
        target = a * scale
        for m in found:
            if _close(target, m.value):
                return Normalised(
                    target, m.currency or currency, m.text, scale,
                    "ok" if scale == 1.0 else "rescaled",
                    "" if scale == 1.0 else
                    f"recorded {a:g} rescaled x{scale:g} to match {m.text!r}")
    return Normalised(None, currency, None, status="contradicts_span",
                      note=(f"recorded {a:g} matches no figure in the span "
                            f"({[m.text for m in found][:4]})"))


def normalise_percentage(pct, span: str) -> tuple[float | None, str]:
    """A percentage must appear in its span. 90 and 90% are the same claim."""
    try:
        p = float(pct)
    except (TypeError, ValueError):
        return None, "no percentage recorded"
    nums = {parse_number(n) for n in re.findall(r"(\d[\d.,]*)\s*(?:%|per ?cent|por ?cento|pour ?cent)",
                                                (span or "").lower())}
    nums |= {parse_number(n) for n in re.findall(r"(\d[\d.,]*)", (span or ""))}
    nums.discard(None)
    if not nums:
        return None, "evidence span contains no percentage"
    if any(abs(p - n) <= 0.51 for n in nums):
        return p, ""
    return None, f"recorded {p:g}% matches no figure in the span"
