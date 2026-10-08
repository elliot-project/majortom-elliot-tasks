"""Check a rule-based caption against the fact sheet's TEXT, independently of its writer.

`caption.caption_claims` returns the paragraphs and a claim for every
proportion it wrote: (sheet key, value, words). This module re-reads the value of each
key from the sheet text itself, checks that the words are true of it (its own table of
what each fraction phrase allows), and checks that every fraction phrase in the caption
is accounted for by a claim, so no proportion goes unchecked. Every number in the
caption must appear in the sheet text, and phrases a blind judge flagged as unsupported
in an earlier version of the template are banned.

Usage:
    problems = check(sheet_text, caption_text, claims)     # [] when clean
"""
from __future__ import annotations

import re

NAMED = {'a tenth': 1 / 10, 'an eighth': 1 / 8, 'a seventh': 1 / 7, 'a sixth': 1 / 6,
         'a fifth': 1 / 5, 'a quarter': 1 / 4, 'three-tenths': 3 / 10, 'a third': 1 / 3,
         'two-fifths': 2 / 5, 'half': 1 / 2, 'three-fifths': 3 / 5, 'two-thirds': 2 / 3,
         'seven-tenths': 7 / 10, 'three-quarters': 3 / 4, 'four-fifths': 4 / 5,
         'nine-tenths': 9 / 10}
ABOUT = ('about', 'roughly', 'around')
OVER = ('a little over', 'just over', 'slightly more than')
UNDER = ('a little under', 'just under', 'slightly less than')
PLAIN = {'all': (0.99, 1.0), 'nearly all': (0.94, 0.995), 'almost all': (0.94, 0.995),
         'a very small part': (0.0, 0.035), 'a sliver': (0.0, 0.035),
         'a small part': (0.025, 0.08), 'a small share': (0.025, 0.08)}

_Q = '|'.join(map(re.escape, sorted(ABOUT + OVER + UNDER, key=len, reverse=True)))
_N = '|'.join(map(re.escape, sorted(NAMED, key=len, reverse=True)))
FRACTION_RE = re.compile(rf'\b(?:(?:{_Q}) (?:{_N})|nearly all|almost all|a very small part|'
                         rf'a sliver|a small part|a small share|all of the|the whole of the)\b',
                         re.IGNORECASE)
NUMBER_RE = re.compile(r'(?<![\w-])-?\d[\d,]*(?:\.\d+)?')

#: Phrases the judge flagged as unsupported or garbled in v1; none may come back.
BANNED = [r'tarmac', r'dry sand', r'calm water', r'mostly sealed', r'sealed, built-up',
          r'\bbarren\b', r'marshland', r'open grassland', r'at the surface',
          r'surface temperature', r'\bmodest\b', r'\bmoderate relief', r'\brugged\b',
          r'mountainous', r'gently', r'\bhealthy\b', r'vigorous', r'\ban university',
          r'unclassified running', r'\ba yes\b', r'\(yes\b', r'class 0',
          r'reflect the radar away', r'little of it dense', r'scattered patches',
          r'\bdominated by\b', r'\ban? (?:unclassified|residential|tertiary)\b(?! (?:road|street|area))']

WC_NAMES = {'tree cover': 10, 'shrubland': 20, 'grassland': 30, 'cropland': 40,
            'built-up': 50, 'bare or sparse vegetation': 60, 'snow and ice': 70,
            'permanent water': 80, 'herbaceous wetland': 90, 'mangroves': 95,
            'moss and lichen': 100}


def sheet_values(sheet: str) -> dict[str, float]:
    """Every proportion the sheet prints, under the keys the template's claims use."""
    v: dict[str, float] = {}

    def grab(key, pat, line):
        m = re.search(pat, line)
        if m:
            v[key] = int(m.group(1)) / 100.0

    for line in sheet.split('\n'):
        if line.startswith('LAND COVER'):
            m0 = re.search(r'class 0 (\d+)%', line)
            if m0:
                v['wc:0'] = int(m0.group(1)) / 100.0
            for name, pct in re.findall(r'([a-z][a-z \-]+?) (\d+)%', line[len('LAND COVER (ESA WorldCover) '):]):
                name = name.strip()
                code = 0 if name.startswith('class') else WC_NAMES.get(name)
                if code is not None:
                    v[f'wc:{code}'] = int(pct) / 100.0
        elif line.startswith('CLOUD'):
            for key, pat in (('clear', r'clear (\d+)%'), ('thick', r'thick cloud (\d+)%'),
                             ('thin', r'thin cloud (\d+)%'), ('shadow', r'shadow (\d+)%')):
                grab(key, pat, line)
            v['cloud'] = v.get('thick', 0) + v.get('thin', 0)
        elif line.startswith('SPECTRAL'):
            for key, pat in (('veg', r'vegetated (\d+)%'), ('dense', r'densely (\d+)%'),
                             ('water', r'open water (\d+)%'),
                             ('bare', r'sealed surfaces\) (\d+)%'),
                             ('snow', r'snow or ice (\d+)%'),
                             ('burn', r'burn-scar signature (\d+)%')):
                grab(key, pat, line)
        elif line.startswith('RADAR'):
            for key, pat in (('s1:specular', r'specular/smooth surfaces (\d+)%'),
                             ('s1:volume', r'volume scattering \(canopy-like\) (\d+)%'),
                             ('s1:bright', r'bright double-bounce \(built-up-like\) (\d+)%')):
                grab(key, pat, line)
    return v


def _num(n: str) -> str:
    n = n.replace(',', '').lstrip('+')
    return '0' if n == '-0' else n


def words_true(words: str, f: float) -> bool:
    w = words.lower()
    if w == 'none':
        return f <= 0.004
    if w == 'some':
        return f > 0
    if w in PLAIN:
        lo, hi = PLAIN[w]
        return lo <= f <= hi
    for q in sorted(ABOUT + OVER + UNDER, key=len, reverse=True):
        if w.startswith(q + ' '):
            t = NAMED.get(w[len(q) + 1:])
            if t is None:
                return False
            if q in ABOUT:
                return abs(f - t) <= 0.015
            if q in OVER:
                return 0 < f - t <= 0.06
            return 0 < t - f <= 0.06
    return False


def check(sheet: str, text: str, claims: list[tuple[str, float, str]]) -> list[str]:
    out = []
    vals = sheet_values(sheet)
    for key, value, words in claims:
        if key not in vals:
            # The sheet omits classes under 1%; anything else missing is a fault.
            if not (key.startswith('wc:') and value < 0.01) and not (value == 0 and words == 'none'):
                out.append(f'claim on {key} has no value in the sheet')
            continue
        tol = 0.015 if key == 'cloud' else 0.011
        if abs(vals[key] - value) > tol:
            out.append(f'{key}: claim value {value:.2f} but the sheet says {vals[key]:.2f}')
        if not words_true(words, vals[key]):
            out.append(f'{key}: "{words}" is not true of {vals[key]:.0%}')
    phrases = [m.group(0).lower() for m in FRACTION_RE.finditer(text)]
    claimed = [w for _, _, w in claims if w not in ('none', 'some')]
    # "all of the tile" / "the whole of the tile" both come from the claim word "all".
    norm = ['all' if p in ('all of the', 'the whole of the') else p for p in phrases]
    if sorted(norm) != sorted(claimed):
        extra = sorted(set(norm) - set(claimed))
        missing = sorted(set(claimed) - set(norm))
        out.append(f'fraction phrases do not match claims (unclaimed {extra}, unwritten {missing})')
    sheet_nums = {_num(n) for n in NUMBER_RE.findall(sheet)}
    for n in NUMBER_RE.findall(text):
        if _num(n) not in sheet_nums:
            out.append(f'number {n} is not in the sheet')
    for pat in BANNED:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            out.append(f'banned phrase: "{m.group(0)}"')
    return out
