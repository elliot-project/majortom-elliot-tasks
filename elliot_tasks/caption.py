"""A rule-based caption of an ELLIOT tile, written from its fact sheet, with seeded wording.

The caption has one paragraph per modality segment (`factsheet.SEGMENTS`: core,
optical, thermal, radar, terrain, sky), so a paragraph whose modality is not in the
sample's input is dropped, exactly as an LLM caption written per segment would be. It
reads only what `factsheet.fact_sheet` measured (`FactSheet.raw`, its location line and
its selected objects), so it never states a fact the sheet does not carry.

This is version 2 of the template. In a blind comparison on 100 stratified tiles,
judged by a large LLM against the image and the sheet, it scored level with a small
LLM captioner given the same sheet, and about one point (of ten) below a large one.
Its rules:

- **Fractions** are matched to the nearest of a dense set of named fractions (tenths,
  sevenths, eighths, ...) on the sheet's own rounded percentage, with "about" only when
  it is within 1.2 points and "a little over/under" otherwise. 28% is "a little under
  three-tenths", not "a third".
- **No word the sheet does not license.** The sheet's own labels are used: "bare ground
  or sealed surfaces" (never "mostly sealed"), "smooth, specular surfaces", WorldCover's
  class names, brightness (not surface) temperature, ERA5 *air* temperature. Relief
  gets no adjective; "dominated" needs 60%. Relief under 40 m is not described.
- **Zero is none.** A 0% dense share is "none of it dense", cloud with no thin part is
  "all thick", and a 0% vegetated fraction with a positive index says exactly that.
- **OSM tags are words, not fields**: an unclassified highway is an "unclassified road",
  `historic=yes` relations are not scene features, a tile-filling island or reserve is
  context ("on", "within"), not a feature.
- **Coverage and salience**: up to six named features chosen by rule across kinds
  (transport, water, settlements, green areas, landmarks, other), lines with their
  direction, each land-cover class with its own share, the units the tile spans, all
  spatial patterns of the optical block in one sentence, radar fractions including the
  ones that are zero, and the radar gradient in a sentence of its own.

Every proportion written is also recorded as a claim (`caption_claims`), so
`caption_check.check` can verify each fraction word against the sheet's text
independently of this code.

Every statement has a bank of paraphrases, sentences within a paragraph are reordered
where their order does not matter, and connectives vary. The choice is seeded per
(cell, seed), so one tile drawn with several seeds gives several wordings of the same
facts, and a given (cell, seed) always gives the same text.

For a monthly or burst tile the caption describes the frame the sheet reads (frame 0
unless `fact_sheet(..., frame=k)`); the series facts are the temporal families' job.

Usage:
    import elliot_tasks as et
    from elliot_tasks import caption as cap
    fs = et.fact_sheet(et.find('284D_496L'))
    cap.caption_segments(fs, seed=0)      # {'core': ..., 'optical': ..., ...}
    cap.caption(fs, seed=1)               # assembled text
    cap.caption(fs, seed=1, available={'s2'})   # only what an S2-only input licenses
    cap.caption_claims(fs, seed=1)        # (segments, [(key, value, phrase), ...])
"""
from __future__ import annotations

import hashlib
import random
import re

import numpy as np

from . import factsheet as ef

VERSION = 2


class _Ctx:
    """The seeded generator plus the claims written so far."""

    def __init__(self, cell: str, seed: int):
        h = hashlib.sha256(f'{cell}|{seed}'.encode()).hexdigest()
        self.rng = random.Random(int(h[:16], 16))
        self.claims: list[tuple[str, float, str]] = []

    def choice(self, seq):
        return self.rng.choice(seq)

    def random(self) -> float:
        return self.rng.random()

    def shuffle(self, seq):
        self.rng.shuffle(seq)


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


def _say(c: _Ctx, phrase: str, frames: list[str]) -> str:
    """Pick one frame and put `phrase` in it: '§' is the phrase, '¶' the phrase
    capitalised. The phrase is built ONCE, before the choice, so the claim it logged
    is the one that is written."""
    return c.choice(frames).replace('¶', _cap(phrase)).replace('§', phrase)


def _join(items: list[str], c: _Ctx) -> str:
    """'a', 'a and b', 'a, b and c' (sometimes with a serial comma)."""
    items = [i for i in items if i]
    if len(items) <= 1:
        return ''.join(items)
    if len(items) == 2:
        return f'{items[0]} and {items[1]}'
    return ', '.join(items[:-1]) + c.choice([' and ', ', and ']) + items[-1]


def _q(f: float) -> float:
    """The sheet's own rounding: it prints `f'{100 * f:.0f}%'`."""
    return int(f'{100 * f:.0f}') / 100.0


# --------------------------------------------------------------------------------
# Quantities in words.
# --------------------------------------------------------------------------------
FRACTIONS = [(1 / 10, 'a tenth'), (1 / 8, 'an eighth'), (1 / 7, 'a seventh'),
             (1 / 6, 'a sixth'), (1 / 5, 'a fifth'), (1 / 4, 'a quarter'),
             (3 / 10, 'three-tenths'), (1 / 3, 'a third'), (2 / 5, 'two-fifths'),
             (1 / 2, 'half'), (3 / 5, 'three-fifths'), (2 / 3, 'two-thirds'),
             (7 / 10, 'seven-tenths'), (3 / 4, 'three-quarters'), (4 / 5, 'four-fifths'),
             (9 / 10, 'nine-tenths')]
ABOUT = ['about', 'roughly', 'around']
OVER = ['a little over', 'just over', 'slightly more than']
UNDER = ['a little under', 'just under', 'slightly less than']
ABOUT_TOL = 0.012
_AREA = ['the tile', 'the scene', 'the area']


def fraction_words(f: float, c: _Ctx) -> str:
    """A proportion in words, on the sheet's rounding. Never a number."""
    f = _q(f)
    if f >= 0.995:
        return 'all'
    if f >= 0.95:
        return c.choice(['nearly all', 'almost all'])
    if f <= 0.0:
        return 'none'
    if f < 0.03:
        return c.choice(['a very small part', 'a sliver'])
    if f < 0.075:
        return c.choice(['a small part', 'a small share'])
    t, w = min(FRACTIONS, key=lambda x: abs(x[0] - f))
    d = f - t
    if abs(d) <= ABOUT_TOL:
        return f'{c.choice(ABOUT)} {w}'
    return f'{c.choice(OVER if d > 0 else UNDER)} {w}'


def prop(f: float, c: _Ctx, key: str, area: bool = True) -> str:
    """`fraction_words` as a noun phrase ("about a third of the tile"), recorded as a
    claim under `key` so a checker can find the sheet value it paraphrases."""
    w = fraction_words(f, c)
    c.claims.append((key, _q(f), w))
    if not area:
        return w
    n = c.choice(_AREA)
    if w == 'all':
        return c.choice([f'all of {n}', f'the whole of {n}'])
    if w == 'none':
        return f'none of {n}'
    return f'{w} of {n}'


# --------------------------------------------------------------------------------
# Vocabulary.
# --------------------------------------------------------------------------------
#: WorldCover's class names, with variants only where they mean the same thing.
WC_WORDS = {
    10: ['tree cover', 'tree-covered land'],
    20: ['shrubland'],
    30: ['grassland'],
    40: ['cropland', 'cultivated land'],
    50: ['built-up land', 'built-up area'],
    60: ['bare or sparsely vegetated ground', 'bare or sparse vegetation'],
    70: ['snow and ice', 'permanent snow and ice'],
    80: ['permanent water', 'open water'],
    90: ['herbaceous wetland'],
    95: ['mangroves'],
    100: ['moss and lichen'],
}

_HIGHWAY = {'motorway': 'motorway', 'trunk': 'trunk road', 'primary': 'primary road',
            'secondary': 'secondary road', 'tertiary': 'tertiary road',
            'unclassified': 'unclassified road', 'residential': 'residential street',
            'service': 'service road', 'track': 'track', 'path': 'path',
            'bridleway': 'bridleway', 'footway': 'footpath', 'cycleway': 'cycle path'}
_RAILWAY = {'rail': 'railway line', 'narrow_gauge': 'narrow-gauge railway',
            'abandoned': 'abandoned railway', 'disused': 'disused railway',
            'light_rail': 'light railway', 'tram': 'tramway'}
_WORDS = {
    ('waterway', 'river'): 'river', ('waterway', 'stream'): 'stream',
    ('waterway', 'canal'): 'canal', ('waterway', 'dam'): 'dam',
    ('waterway', 'waterfall'): 'waterfall', ('waterway', 'drain'): 'drain',
    ('natural', 'water'): 'water body', ('natural', 'bay'): 'bay',
    ('natural', 'wetland'): 'wetland', ('natural', 'fen'): 'fen',
    ('natural', 'beach'): 'beach', ('natural', 'coastline'): 'coastline',
    ('natural', 'glacier'): 'glacier or ice shelf', ('natural', 'peak'): 'peak',
    ('natural', 'volcano'): 'volcano', ('natural', 'wood'): 'wood',
    ('natural', 'scrub'): 'scrubland area', ('natural', 'bare_rock'): 'bare rock area',
    ('natural', 'mountain_range'): 'mountain range', ('natural', 'peninsula'): 'peninsula',
    ('landuse', 'railway'): 'railway yard', ('landuse', 'residential'): 'residential area',
    ('landuse', 'industrial'): 'industrial area', ('landuse', 'retail'): 'retail area',
    ('landuse', 'commercial'): 'commercial area', ('landuse', 'military'): 'military area',
    ('landuse', 'cemetery'): 'cemetery', ('landuse', 'forest'): 'forest',
    ('landuse', 'farmyard'): 'farmyard', ('landuse', 'landfill'): 'landfill',
    ('landuse', 'aquaculture'): 'aquaculture site', ('landuse', 'village_green'): 'village green',
    ('leisure', 'park'): 'park', ('leisure', 'nature_reserve'): 'nature reserve',
    ('leisure', 'golf_course'): 'golf course', ('leisure', 'garden'): 'garden',
    ('leisure', 'stadium'): 'stadium', ('leisure', 'sports_centre'): 'sports centre',
    ('aeroway', 'aerodrome'): 'airfield', ('aeroway', 'airstrip'): 'airstrip',
    ('aeroway', 'apron'): 'airport apron',
    ('amenity', 'school'): 'school', ('amenity', 'university'): 'university campus',
    ('amenity', 'college'): 'college campus', ('amenity', 'hospital'): 'hospital',
    ('amenity', 'grave_yard'): 'graveyard', ('amenity', 'place_of_worship'): 'place of worship',
    ('amenity', 'townhall'): 'town hall', ('amenity', 'animal_breeding'): 'animal-breeding farm',
    ('building', 'school'): 'school building',
    ('man_made', 'works'): 'works', ('man_made', 'wastewater_plant'): 'wastewater plant',
    ('tourism', 'camp_site'): 'campsite', ('tourism', 'caravan_site'): 'caravan site',
    ('place', 'isolated_dwelling'): 'isolated dwelling', ('place', 'city_block'): 'city block',
    ('place', 'island'): 'island', ('place', 'sea'): 'sea',
}
_SETTLEMENT = ['city', 'town', 'suburb', 'village', 'neighbourhood', 'quarter', 'borough',
               'hamlet', 'locality', 'isolated_dwelling', 'city_block', 'plot', 'municipality']
#: Tags that name no kind of place: these are administrative or historical relations
#: and generic buildings, and a caption has nothing honest to call them.
_SKIP = {('historic', 'yes'), ('building', 'yes'), ('boundary', 'administrative')}
#: Features that fill the tile are context for the location sentence, not things in it.
_CONTEXT_WITHIN = {('leisure', 'nature_reserve'), ('boundary', 'national_park'),
                   ('boundary', 'protected_area'), ('natural', 'mountain_range'),
                   ('natural', 'peninsula'), ('natural', 'glacier'), ('place', 'sea')}
_CONTEXT_ON = {('place', 'island'), ('place', 'islet')}


_PLURAL = {'moss and lichen', 'mangroves', 'snow and ice', 'permanent snow and ice'}


def _v(noun: str, singular: str, plural: str) -> str:
    return plural if noun in _PLURAL else singular


def feature_word(m: dict) -> str | None:
    cat, sub = m['category'], m['subtype']
    if (cat, sub) in _SKIP or sub in ('yes', '(none)', ''):
        return None
    if cat == 'highway':
        return _HIGHWAY.get(sub, f'{sub.replace("_", " ")} road')
    if cat == 'railway':
        return _RAILWAY.get(sub, f'{sub.replace("_", " ")} railway')
    if cat == 'place':
        return _WORDS.get((cat, sub), sub.replace('_', ' '))
    return _WORDS.get((cat, sub), sub.replace('_', ' '))


def feature_group(m: dict) -> str:
    cat, sub = m['category'], m['subtype']
    if cat in ('highway', 'railway', 'aeroway') or (cat, sub) == ('landuse', 'railway'):
        return 'transport'
    if cat == 'waterway' and sub in ('river', 'stream', 'canal', 'drain'):
        return 'water'
    if (cat == 'natural' and sub in ('water', 'bay', 'wetland', 'fen', 'beach', 'coastline',
                                     'glacier')) or (cat, sub) in (('place', 'sea'),
                                                                   ('landuse', 'aquaculture')):
        return 'water'
    if (cat == 'place' and sub in _SETTLEMENT) or (cat, sub) == ('landuse', 'residential'):
        return 'settlement'
    if (cat == 'leisure' and sub in ('park', 'nature_reserve', 'golf_course', 'garden')) or \
            (cat, sub) in (('landuse', 'forest'), ('natural', 'wood'), ('natural', 'scrub'),
                           ('landuse', 'village_green')):
        return 'green'
    if (cat == 'natural' and sub in ('peak', 'volcano', 'bare_rock', 'mountain_range',
                                     'peninsula', 'saddle', 'cape')) or \
            (cat == 'waterway' and sub in ('dam', 'waterfall')) or \
            (cat, sub) in (('place', 'island'), ('place', 'islet')):
        return 'landmark'
    return 'other'


def _article(w: str) -> str:
    lw = w.lower()
    if lw.startswith(('uni', 'use', 'eu', 'one', 'once')):
        return 'a'
    if lw.startswith(('hour', 'honest', 'heir')):
        return 'an'
    return 'an' if lw[:1] in 'aeiou' else 'a'


def _name(m: dict) -> str:
    """OSM names joined by ';' are alternatives for one feature, not two features."""
    parts = [p.strip() for p in str(m['name']).split(';') if p.strip()]
    if len(parts) > 1:
        return f'{parts[0]} (also {parts[1]})'
    return parts[0] if parts else str(m['name'])


_COLS = {'western': {'north-west', 'west', 'south-west'},
         'central': {'north', 'centre', 'south'},
         'eastern': {'north-east', 'east', 'south-east'}}
_ROWS = {'northern': {'north-west', 'north', 'north-east'},
         'central': {'west', 'centre', 'east'},
         'southern': {'south-west', 'south', 'south-east'}}


def _compact(cells: list[str]) -> str | None:
    """Five or more compass cells, said as the thirds of the tile they make up."""
    got = set(cells)
    for thirds in (_COLS, _ROWS):
        full = [k for k, v in thirds.items() if v <= got]
        if full and set().union(*(thirds[k] for k in full)) == got:
            return f'across the {_join_plain(full)} thirds of the tile'
    return None


def _join_plain(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ', '.join(items[:-1]) + ' and ' + items[-1]


def _where(m: dict) -> tuple[str, bool]:
    """(position words, is_line) from the sheet's own position phrase.

    The sheet's phrase is the authority; this only rewords it into prose, keeping every
    compass cell it lists."""
    if not m.get('bbox'):
        return '', False
    a, l = m.get('area') or 0, m.get('length') or 0
    p = ef.position_phrase(m['bbox'], bool(l) and not a, m.get('geom'))
    mm = re.match(r'running (.+?), (.+)$', p)
    if not mm:
        if p.startswith('in the ') and p.count(',') >= 3:
            cells = re.split(r', | and ', p[len('in the '):])
            return _compact(cells) or 'across much of the tile', False
        if p == 'spread over most of the tile':
            return 'over most of the tile', False
        return p, False
    bearing, span = mm.groups()
    if ' to ' in span and ' through ' not in span:
        x, y = span.split(' to ', 1)
        return f'{bearing} from the {x} to the {y}', True
    cells = span.split(' through ')
    if len(cells) == 1:
        return f'{bearing} in the {cells[0]}', True
    return f'{bearing} through the ' + ', '.join(cells[:-1]) + f' and {cells[-1]}', True


# --------------------------------------------------------------------------------
# Location and context.
# --------------------------------------------------------------------------------
def _location(fs: ef.FactSheet):
    """(finest unit, region, country, open-water flag, spanned units) from LOCATION."""
    line = next((l for m, l in fs.blocks if l.startswith('LOCATION ')), None)
    if not line:
        return None, None, None, False, []
    body, _, span = line[len('LOCATION '):].partition('  [tile spans ')
    if body.startswith('open water'):
        return None, None, None, True, []
    parts = []
    for p in (p.strip() for p in body.split(',')):
        if p and p not in parts:
            parts.append(p)
    spanned = []
    if span:
        # "2 units at level 8 (A, B); 6 units at level 9 (C, D, E, ...)" -> the
        # coarsest level that splits the tile into two or three named units.
        for n, lvl, names in re.findall(r'(\d+) units at level (\d+) \(([^)]*)\)', span):
            if 2 <= int(n) <= 3 and '...' not in names:
                spanned = [x.strip() for x in names.split(',')][:int(n)]
                break
    if len(parts) == 1:
        return None, None, parts[0], False, spanned
    country = parts[-1]
    region = None
    for cand in reversed(parts[1:-1]):
        if cand.endswith((' Region', 'Federal District')) or cand == country \
                or cand.startswith('Metropolitan '):
            continue
        region = cand
        break
    fine = parts[0] if parts[0] not in (region, country) else None
    if fine in spanned and len(parts) > 2 and parts[1] not in (region, country):
        fine = parts[1]          # the tile spans the finest unit and its neighbour
    elif fine in spanned:
        fine = None
    return fine, region, country, False, spanned


def _wc_shares(fs: ef.FactSheet) -> list[tuple[int, float]]:
    wc = fs.raw.get('worldcover') or {}
    hist = {int(k): v for k, v in wc.items()}
    tot = sum(hist.values()) or 1
    return sorted(((k, v / tot) for k, v in hist.items()), key=lambda kv: -kv[1])


def _core(fs: ef.FactSheet, c: _Ctx) -> str:
    fine, region, country, ocean, spanned = _location(fs)
    shares = _wc_shares(fs)
    classes = [(k, f) for k, f in shares if k in WC_WORDS and _q(f) >= 0.05]
    nodata = sum(f for k, f in shares if k not in WC_WORDS)
    water = sum(f for k, f in shares if k == 80)
    acq = fs.raw.get('acq') or {}

    # Features: the tile-filling ones are context, the rest are chosen by kind.
    objs, seen = [], set()
    context = []
    for m in fs.objects:
        if feature_word(m) is None or _name(m) in seen:
            continue
        seen.add(_name(m))
        key = (m['category'], m['subtype'])
        a = m.get('area') or 0
        if (a >= 0.8 * ef.TILE_AREA_M2 and key in _CONTEXT_WITHIN) or \
                (a >= 0.3 * ef.TILE_AREA_M2 and key in _CONTEXT_ON):
            context.append(m)
            continue
        objs.append(m)

    sents = []
    # --- where ---
    where = ', '.join(x for x in (fine, region, country) if x)
    if ocean:
        loc = c.choice(['The tile lies outside any administrative area.',
                        'No administrative area covers this tile.'])
    elif country and not fine and not region and max(
            water, (fs.raw.get('s2_spectral') or {}).get('frac_water', 0)) >= 0.9:
        loc = c.choice([f'This tile is open water within the boundaries of {country}.',
                        f'The tile lies over water inside the boundaries of {country}.'])
    elif where:
        loc = c.choice([f'This tile lies in the area of {where}.',
                        f'The scene lies within the boundaries of {where}.',
                        f'This image covers part of the area of {where}.'])
        if spanned:
            loc = loc[:-1] + c.choice([f', spanning {_join(spanned, c)}.',
                                       f', across {_join(spanned, c)}.'])
    else:
        loc = None
    if loc:
        sents.append(loc)
    for m in context[:2]:
        key = (m['category'], m['subtype'])
        w = feature_word(m)
        if key in _CONTEXT_ON:
            sents.append(c.choice([f'The land here is part of the {w} of {_name(m)}.',
                                   f'Its land belongs to the {w} of {_name(m)}.']))
        else:
            sents.append(c.choice([f'It lies within {_name(m)}, {_article(w)} {w}.',
                                   f'The tile falls inside {_name(m)}, {_article(w)} {w}.']))

    # --- land cover: every class of 5% or more, each with its own share ---
    if classes:
        names = [c.choice(WC_WORDS[k]) for k, _ in classes]
        (k1, f1) = classes[0]
        if _q(f1) >= 0.95:
            lc = _say(c, prop(f1, c, f'wc:{k1}'), [f'¶ is {names[0]}',
                                                    f'{_cap(names[0])} {_v(names[0], "covers", "cover")} §'])
            if len(classes) > 1:
                lc += c.choice([f', with some {names[1]}', f', with a little {names[1]}'])
                c.claims.append((f'wc:{classes[1][0]}', _q(classes[1][1]), 'some'))
            lc += '.'
        else:
            form = c.choice([0, 1, 2] if _q(f1) >= 0.6 else [0, 1])
            p1 = prop(f1, c, f'wc:{k1}')
            head = (f'{_cap(names[0])} {_v(names[0], "covers", "cover")} {p1}' if form == 0 else
                    f'{_cap(p1)} is {names[0]}' if form == 1 else
                    f'{_cap(names[0])} {_v(names[0], "dominates", "dominate")}, covering {p1}')
            rest = [f'{n} {prop(f, c, f"wc:{k}", area=False)}'
                    for n, (k, f) in zip(names[1:4], classes[1:4])]
            if rest:
                lc = head + c.choice([', ', '; ']) + _join(rest, c) + '.'
            else:
                lc = head + '.'
        sents.append(lc)
    if _q(nodata) >= 0.05:
        sents.append(_say(c, prop(nodata, c, 'wc:0'), ['WorldCover has no data for §.',
                                                        '¶ has no WorldCover class.']))

    hm = acq.get('human_mod')
    if hm is not None:
        band = ('near-pristine, with very little human modification' if hm < 0.1 else
                'lightly modified by human activity' if hm < 0.3 else
                'substantially modified by human activity' if hm < 0.6 else
                'heavily modified by human activity')
        sents.append(c.choice([f'The landscape is {band}.', f'It is {band}.']))

    feats = _feature_sentences(objs, c)
    sents.extend(feats)
    if len(objs) < 2:
        mix = _feature_mix(fs, c)
        if mix:
            sents.append(mix)

    # --- notable weather at overpass, as ERA5 gives it (air temperature) ---
    if acq.get('t2m') is not None:
        t = int(f'{acq["t2m"]:.0f}') + 0
        bits = []
        if (acq.get('precip') or 0) >= 0.5:
            bits.append(f'{acq["precip"]:.1f} mm of precipitation')
        if (acq.get('wind') or 0) >= 30:
            bits.append(f'wind of about {acq["wind"]:.0f} km/h')
        if bits or not -15 <= t <= 35:
            sents.append(c.choice([
                f'At the overpass ERA5 gives an air temperature of about {t} °C'
                + (f', with {_join(bits, c)}' if bits else '') + '.',
                f'ERA5 puts the air temperature at about {t} °C at the time of the image'
                + (f', with {_join(bits, c)}' if bits else '') + '.']))
    return ' '.join(sents)


def _feature_sentences(objs: list[dict], c: _Ctx) -> list[str]:
    """Up to six features, the best of each kind first, then by size."""
    caps = {'transport': 2, 'water': 2, 'settlement': 2, 'green': 2, 'landmark': 3,
            'other': 1}
    by = {}
    for rank, m in enumerate(objs):
        by.setdefault(feature_group(m), []).append((rank, m))
    order = sorted(by, key=lambda g: by[g][0][0])
    chosen = {g: [] for g in order}
    total, MAX = 0, 6
    # Round-robin over kinds in order of their largest member, within caps.
    for depth in range(3):
        for g in order:
            if total >= MAX:
                break
            if depth < len(by[g]) and depth < caps[g]:
                chosen[g].append(by[g][depth][1])
                total += 1
    out = []
    for g in order:
        ms = chosen[g]
        if not ms:
            continue
        if len(ms) > 1 and all(not (m.get('area') or m.get('length')) for m in ms):
            # Peaks and the like are points; group them when they share a position.
            words = {feature_word(m) for m in ms}
            pos = {_where(m)[0] for m in ms}
            if len(words) == 1 and len(pos) == 1:
                w, ps = words.pop(), pos.pop()
                names = _join([_name(m) for m in ms], c)
                pl = w + ('es' if w.endswith(('s', 'sh', 'ch', 'x')) else 's')
                out.append(c.choice([f'The {pl} {names} are mapped {ps}.',
                                     f'{names} are {pl} {ps}.']))
                continue
        clauses = [_feature_clause(m, c) for m in ms]
        if len(clauses) == 1:
            out.append(_cap(clauses[0]) + '.')
        else:
            out.append(_cap(c.choice(['; ', ', while ']).join(clauses)) + '.'
                       if len(clauses) == 2 else _cap('; '.join(clauses)) + '.')
    return out


_COUNT_WORDS = {1: 'a', 2: 'two', 3: 'three'}
_MIX_KEEP = {'transport', 'water', 'green', 'settlement', 'landmark'}


def _feature_mix(fs: ef.FactSheet, c: _Ctx) -> str | None:
    """Where named features are few, the sheet's FEATURE MIX (all OSM features, named
    or not) still says what is mapped: "three tracks, a path and a stream"."""
    line = next((l for _, l in fs.blocks if l.startswith('FEATURE MIX ')), None)
    if not line:
        return None
    items = []
    for cat, sub, n in re.findall(r'([a-z_]+)/([a-z_]+|None) x(\d+)', line):
        m = {'category': cat, 'subtype': sub}
        w = feature_word(m) if sub != 'None' else None
        if not w or feature_group(m) not in _MIX_KEEP or (cat, sub) == ('place', 'region'):
            continue
        n = int(n)
        if (cat, sub) == ('landuse', 'farmland'):
            items.append('farmland plots')
            continue
        pl = w + ('es' if w.endswith(('s', 'sh', 'ch', 'x')) else 's')
        items.append(f'{_article(w)} {w}' if n == 1 else
                     f'{_COUNT_WORDS.get(n, "several" if n < 10 else "many")} {pl}')
        if len(items) == 4:
            break
    if not items:
        return None
    return c.choice([f'OpenStreetMap maps {_join(items, c)} here, mostly unnamed.',
                     f'Mapped features, mostly unnamed, include {_join(items, c)}.'])


def _feature_clause(m: dict, c: _Ctx) -> str:
    w = feature_word(m)
    nm = _name(m)
    pos, is_line = _where(m)
    named_kind = w.split()[-1].lower() in nm.lower()     # "Blainroe Golf Course"
    if named_kind and not (m['category'] == 'place' and m['subtype'] in _SETTLEMENT):
        verb = 'runs' if is_line else ('extends' if pos.startswith(('over', 'across'))
                                       else 'lies')
        return f'{nm} {verb} {pos}' if pos else f'{nm} is mapped in the tile'
    if is_line:
        return c.choice([f'{nm}, {_article(w)} {w}, runs {pos}',
                         f'the {w} {nm} runs {pos}'])
    if m['category'] == 'place' and m['subtype'] in _SETTLEMENT:
        if pos:
            return c.choice([f'the {w} of {nm} is {pos}', f'{nm}, {_article(w)} {w}, lies {pos}'])
        return f'the {w} of {nm} is mapped here'
    if not pos:
        return f'{nm}, {_article(w)} {w}, is mapped in the tile'
    if pos.startswith(('over', 'across')):
        return c.choice([f'{nm}, {_article(w)} {w}, extends {pos}',
                         f'{_article(w)} {w}, {nm}, extends {pos}'])
    return c.choice([f'{nm}, {_article(w)} {w}, lies {pos}',
                     f'{_article(w)} {w}, {nm}, is {pos}'])


# --------------------------------------------------------------------------------
# Sensor paragraphs.
# --------------------------------------------------------------------------------
def _grid_where(g, hi: str, lo: str, spread: float) -> tuple[str, str] | None:
    p = ef._grid_phrase(np.asarray(g, dtype=float), hi, lo, spread) if g is not None else None
    if not p:
        return None
    m = re.match(rf'{hi} in the (.+), {lo} in the (.+)$', p)
    return m.groups() if m else None


def _optical(fs: ef.FactSheet, c: _Ctx) -> str:
    """Fractions first, then every spatial pattern the sheet gives, in one sentence.

    The gating mirrors the sheet's own: a gradient is mentioned exactly when the sheet
    prints it."""
    sp = fs.raw.get('s2_spectral') or {}
    if sp.get('ndvi_mean') is None:
        return ''               # the sky paragraph says the scene is obscured
    fv, fd = sp['frac_veg'], sp['frac_dense_veg']
    fw, fb = sp.get('frac_water', 0), sp.get('frac_swir_bright_nonveg', 0)
    fsn = sp.get('frac_snow', 0)
    sents = []

    if _q(fv) > 0:
        if _q(fd) <= 0:
            c.claims.append(('dense', 0.0, 'none'))
        dense = (c.choice([', none of it dense', ', with no dense vegetation'])
                 if _q(fd) <= 0 else
                 _say(c, prop(fd, c, 'dense'), [', with dense vegetation over §',
                                                 ', dense over §']))
        s = _say(c, prop(fv, c, 'veg'), [f'Vegetation covers §{dense}',
                                          f'¶ is vegetated{dense}'])
    else:
        sign = ('weakly positive' if sp['ndvi_mean'] > 0.005 else
                'near zero' if sp['ndvi_mean'] > -0.005 else 'negative')
        s = c.choice([f'No part of the tile reaches the vegetated threshold, and the '
                      f'vegetation index is {sign} on average',
                      f'None of the tile counts as vegetated; the vegetation index is '
                      f'{sign} on average'])
        c.claims.append(('veg', 0.0, 'none'))
    sents.append(s + '.')

    rest = []
    if _q(fw) > 0:
        rest.append(_say(c, prop(fw, c, 'water'), ['open water covers §', '§ is open water']))
    else:
        rest.append(c.choice(['there is no open water', 'no open water is detected']))
    if _q(fb) > 0:
        rest.append(_say(c, prop(fb, c, 'bare'), [
            'bare ground or sealed surfaces, unvegetated and bright in the shortwave '
            'infrared, cover §',
            '§ is unvegetated and bright in the shortwave infrared (bare ground or sealed '
            'surfaces)']))
    if fsn > 0.02:
        rest.append(_say(c, prop(fsn, c, 'snow'), ['snow or ice covers §',
                                                    '§ has a snow or ice signature']))
    if ef._burn_is_plausible(sp, fs.raw.get('worldcover')):
        rest.append(_say(c, prop(sp['frac_low_nbr'], c, 'burn'), [
            'a burn-scar signature appears over §', '§ shows a burn-scar signature']))
    if len(rest) <= 2:
        sents.append(_cap(c.choice(['; ', ', and ']).join(rest)) + '.')
    else:
        sents.append(_cap('; '.join(rest)) + '.')

    # Spatial patterns, combined.
    pats = []
    g = _grid_where(sp.get('ndvi_grid'), 'greenest', 'least green', 0.10) if fv > 0.10 else None
    if g:
        pats.append(c.choice([f'vegetation is greenest in the {g[0]} and least green in '
                              f'the {g[1]}', f'greenness peaks in the {g[0]} and is lowest '
                              f'in the {g[1]}']))
    g = _grid_where(sp.get('water_grid'), 'most water', 'least', 0.15) if fw > 0.03 else None
    if g:
        pats.append(c.choice([f'water is concentrated in the {g[0]}',
                              f'most of the water is in the {g[0]}']))
    g = (_grid_where(sp.get('swir_bright_grid'), 'most concentrated', 'least', 0.15)
         if fb > 0.03 else None)
    if g:
        pats.append(c.choice([f'the bare or sealed surfaces concentrate in the {g[0]}',
                              f'the bare or sealed surfaces are densest in the {g[0]}']))
    g = _grid_where(sp.get('snow_grid'), 'deepest cover', 'least', 0.15) if fsn > 0.02 else None
    if g:
        pats.append(c.choice([f'snow or ice is most extensive in the {g[0]}',
                              f'the snow or ice signature is strongest in the {g[0]}']))
    if pats:
        sents.append(_cap(c.choice(['; ', ', while '] if len(pats) == 2 else ['; '])
                          .join(pats)) + '.')
    return ' '.join(sents)


def _thermal(fs: ef.FactSheet, c: _Ctx) -> str:
    th = fs.raw.get('l8_thermal') or {}
    if not th:
        return ''
    m, lo, hi = (int(f'{th[k]:.0f}') + 0 for k in ('bt_mean_c', 'bt_p05_c', 'bt_p95_c'))
    if hi - lo < 1:
        s = c.choice([f'On a separate Landsat-8 overpass the at-sensor brightness '
                      f'temperature is uniform, about {m} °C',
                      f'Landsat-8 thermal data from another date give a uniform brightness '
                      f'temperature of about {m} °C'])
    else:
        s = c.choice([f'On a separate Landsat-8 overpass the at-sensor brightness '
                      f'temperature averages about {m} °C, from {lo} to {hi} °C across the tile',
                      f'Landsat-8 thermal data from another date give a brightness '
                      f'temperature of about {m} °C on average, ranging from {lo} to {hi} °C'])
    gw = _grid_where(th.get('bt_grid'), 'warmest', 'coolest', 1.5)
    if gw:
        s += c.choice([f'. The {gw[0]} is warmest and the {gw[1]} coolest',
                       f', warmest in the {gw[0]} and coolest in the {gw[1]}'])
    return s + '.'


def _radar(fs: ef.FactSheet, c: _Ctx) -> str:
    s1 = fs.raw.get('s1') or {}
    if not s1:
        return ''
    mech = [('s1:volume', s1['frac_volume'], 'canopy-like volume scattering'),
            ('s1:bright', s1['frac_bright'], 'bright built-up-like double-bounce returns'),
            ('s1:specular', s1['frac_specular'], 'smooth specular surfaces')]
    mech.sort(key=lambda x: -x[1])
    present = [x for x in mech if _q(x[1]) > 0]
    absent = [x for x in mech if _q(x[1]) <= 0]
    parts = [f'{w} over {prop(f, c, k)}' for k, f, w in present]
    sents = []
    if parts:
        sents.append(c.choice([f'C-band radar from Sentinel-1 shows {_join(parts, c)}',
                               f'Sentinel-1 C-band backscatter shows {_join(parts, c)}']))
    if absent:
        for k, f, w in absent:
            c.claims.append((k, 0.0, 'none'))
        ws = [w for _, _, w in absent]
        none = ws[0] if len(ws) == 1 else ', '.join(ws[:-1]) + ' or ' + ws[-1]
        verb = 'are' if ws[-1].endswith(('returns', 'surfaces')) else 'is'
        sents.append(c.choice([f'no {none} {verb} detected', f'the tile shows no {none}']))
    s = sents[0] + ('; ' + sents[1] if len(sents) > 1 else '') + '.'
    s = _cap(s)
    gw = _grid_where(s1.get('vv_grid'), 'brightest', 'darkest', 2.0)
    if gw:
        s += c.choice([f' Backscatter is brightest in the {gw[0]} and darkest in the {gw[1]}.',
                       f' Overall backscatter is strongest in the {gw[0]} and weakest in '
                       f'the {gw[1]}.'])
    return s


def _terrain(fs: ef.FactSheet, c: _Ctx) -> str:
    """The sheet's range and relief, and the median as a position within the range."""
    dem = fs.raw.get('dem')
    if not dem or 'dem' not in fs.available:
        return ''
    lo, hi, _med = (int(f'{dem[k]:.0f}') + 0 for k in ('min', 'max', 'p50'))
    relief = int(f'{dem["max"] - dem["min"]:.0f}')
    if relief < 40:
        return ''
    r = (dem['p50'] - dem['min']) / max(dem['max'] - dem['min'], 1e-6)
    where = (c.choice([', and most of the tile lies in the lower part of that range',
                       ', with most of the ground near the low end'])
             if r < 0.25 else
             c.choice([', and most of the tile lies in the upper part of that range',
                       ', with most of the ground near the high end'])
             if r > 0.75 else '')
    return c.choice([
        f'Elevation ranges from {lo:,} to {hi:,} m, a relief of {relief:,} m{where}.',
        f'The ground lies between {lo:,} and {hi:,} m, a relief of {relief:,} m{where}.'])


def _sky(fs: ef.FactSheet, c: _Ctx) -> str:
    cl = fs.raw.get('cloud')
    if not cl or 's2' not in fs.available or _q(cl['clear']) >= 0.97:
        return ''
    thick, thin, shadow = _q(cl['thick_cloud']), _q(cl['thin_cloud']), _q(cl['shadow'])
    cloud = thick + thin
    bits = []
    if cloud > 0:
        kind = ('all of it thick' if thin <= 0 else 'all of it thin' if thick <= 0 else
                'mostly thick' if thick >= 2 * thin else 'mostly thin' if thin >= 2 * thick
                else 'a mix of thick and thin')
        bits.append(_say(c, prop(cloud, c, 'cloud'), [f'cloud covers § ({kind})',
                                                       f'§ is under cloud ({kind})']))
    if shadow > 0:
        bits.append(_say(c, prop(shadow, c, 'shadow'), ['shadow covers §', '§ is in shadow']))
    if not bits:
        return ''
    s = _cap('; '.join(bits))
    if _q(cl['clear']) < 0.2:
        s += c.choice(['; the surface is largely obscured', ', so most of the ground is hidden'])
    else:
        s += _say(c, prop(cl['clear'], c, 'clear'), ['; § is clear', ', leaving § clear'])
    return s + '.'


_WRITERS = {'core': _core, 'optical': _optical, 'thermal': _thermal,
            'radar': _radar, 'terrain': _terrain, 'sky': _sky}


def caption_claims(fs: ef.FactSheet, seed: int = 0, available: set[str] | None = None):
    """(segments, claims): the paragraphs, and every proportion stated as
    (sheet key, sheet value on its rounding, words used)."""
    av = set(available) if available is not None else fs.available
    c = _Ctx(fs.cell, seed)
    out = {}
    for k, spec in ef.SEGMENTS.items():
        if spec['requires'] <= av:
            out[k] = _WRITERS[k](fs, c)
    return out, c.claims


def caption_segments(fs: ef.FactSheet, seed: int = 0,
                     available: set[str] | None = None) -> dict[str, str]:
    """One paragraph per segment the sample's modalities license, as the LLM returns."""
    return caption_claims(fs, seed, available)[0]


def caption(fs: ef.FactSheet, seed: int = 0, available: set[str] | None = None,
            sep: str = '\n\n') -> str:
    seg = caption_segments(fs, seed, available)
    av = set(available) if available is not None else fs.available
    return sep.join(t for k, t in seg.items()
                    if t and ef.SEGMENTS[k]['requires'] <= av)


if __name__ == '__main__':
    import sys

    from .data import configure
    configure()                          # roots from ELLIOT_ROOT (and ELLIOT_X_EXT_ROOT)
    for cell in sys.argv[1:] or ['284D_496L']:
        f = ef.fact_sheet(cell)
        for s in (0, 1):
            print(f'--- {cell} seed {s}')
            print(caption(f, s))
