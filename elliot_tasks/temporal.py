"""Series facts: what is true of a STACK and of no single frame in it.

ONE CAPTION PER SERIES, NOT ONE PER FRAME. The encoder is fed the stack, so a
per-frame caption teaches frame-wise description and discards the only thing a series
has that a date does not. This module produces the `temporal` segment of that one
caption, and the facts the temporal task families draw on.

THE CONTRACT EXTENDS, IT DOES NOT CHANGE. `factsheet.SEGMENTS` gains one key,
`temporal`, requiring {'s2', 'series'}; a monotemporal sample never carries `series`,
so `assemble()` drops the paragraph and one corpus serves all three parts. Nothing
else in the sheet moves.

WHAT BELONGS HERE, AND THE ONE THING THAT DOES NOT:

  phenology     the NDVI trajectory -- amplitude, the month of peak and trough, and
                the NUMBER OF GREEN-UP CYCLES. One cycle is a season or a single
                crop; two is double-cropping, and that label is not obtainable from
                any single frame or from any map we hold.
  water         minimum and maximum extent and the month of each: a reservoir drawn
                down in summer, a floodplain inundated in one frame.
  snow          which frames carry cover, and the maximum. A snowline is a temporal
                object.
  disturbance   dNBR between consecutive frames. THIS is where a real burn statement
                lives; the single-date proxy in `factsheet` is deliberately weak, and
                burst's ~5-day cadence exists for fire and flood onset.
  usability     which frames are too cloudy to read. Without it the caption asserts a
                phenology it could not see: if May and June are both under thick
                cloud, "greens up in late spring" is a guess about missing data.
  weather       the ERA5 trajectory, which is what lets a low NDVI be ATTRIBUTED to a
                dry spring rather than merely reported.

  NOT RADAR.    ELLIOT-X-EXT holds one S1 image per S2 frame (a series for
                monthly and burst), but nothing here describes it yet, and a
                temporal paragraph must not imply an S1 trajectory it never read.

MONTHS, NEVER INDICES. Every statement names the month or the date it belongs to.
"Greenest in March" is checkable against the acquisition dates the sheet already
carries; "greenest in frame 3" teaches a model to count tensor slices, which is a
fact about our storage order and not about the Earth.

NEITHER SERIES PART IS WHAT ITS NAME SUGGESTS, and getting it wrong would put a
confident false claim on every one of the 29,166 series tiles. Measured:

  monthly   is NOT a year. Median span 2,442 days -- about six and a half years --
            across 4 to 8 distinct calendar years, with only 0.2% of samples drawn
            from a single year, and typically 9 to 11 distinct months rather than 12
            (some months repeat, some are missing). It is a CLIMATOLOGY: one frame
            per calendar month, each from whichever year gave a clean scene. So
            "vegetation peaks in October" is a statement about the typical seasonal
            cycle, and "two crops in the year" is a statement this data cannot
            support about any particular year. `SeriesKind.CLIMATOLOGY`.

  burst     is NOT a season. Median span 25 days, 93.6% within one calendar year,
            one or two distinct months. Month names are the wrong unit entirely and
            seasonal language is simply false; what a four-week window at a five-day
            cadence is FOR is abrupt change -- flood, fire, harvest, landslide -- so
            that is what it reports, by date. `SeriesKind.BURST`.

The mode is chosen from the DATES, not from the part name, so a future part with a
different cadence is described correctly without editing this module.
"""
from __future__ import annotations

import collections
import datetime
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from . import factsheet as fsheet
from .data import Tile

MONTHS = ('January', 'February', 'March', 'April', 'May', 'June', 'July',
          'August', 'September', 'October', 'November', 'December')

#: A frame cloudier than this cannot support a statement about the ground under it.
#: Stricter than the per-cell gate in `factsheet` because a series can afford to drop
#: a frame: there are eleven others, and a masked mean over 30% of a tile is noisier
#: than the seasonal signal it would be read as.
FRAME_CLEAR_MIN = 0.50

#: Below this NDVI swing there is no season to describe. An evergreen tropical tile
#: runs 0.78 to 0.82 all year, and calling its maximum a "green-up" invents a
#: phenology out of noise.
NDVI_AMPLITUDE_MIN = 0.12

#: A green-up counts only where the rise clears this. Without it, sensor noise on a
#: flat trajectory scores three or four "cycles" and every tile looks double-cropped.
CYCLE_PROMINENCE = 0.15

#: Per-PIXEL dNBR at which USGS calls a burn low-to-moderate severity. It is a
#: threshold on a pixel, and that distinction is the whole point of `dnbr_area`
#: below: comparing it against the TILE MEAN is a category error. A fire covering a
#: tenth of a 111 km2 tile moves the mean by a tenth of its own severity, so such a
#: test can never fire. Measured over 125 burst series, the 99th percentile of the
#: largest mean drop over land is 0.126 -- a 0.20 gate on the mean sits above the
#: entire observed distribution and detects nothing in a part whose stated purpose is
#: floods, fires and landslides.
DNBR_PIXEL_SEVERE = 0.27

#: And the share of the tile that has to clear it before the series says so. Below
#: this it is speckle, co-registration error at field edges, or one cloud shadow the
#: mask missed.
DNBR_AREA_MIN = 0.02

#: An absence claim ("near-constant through the season") needs enough frames to be an
#: absence rather than a gap. Three clear frames out of twelve can see 0.63 to 0.67 of
#: a series that actually runs 0.02 to 0.67, and report it as constant.
MIN_FRAMES_FOR_ABSENCE = 6

#: Above this span, in days, the frames cannot be one season of one year.
CLIMATOLOGY_SPAN_DAYS = 400

#: A vegetation statement needs vegetation. Without this gate a 92%-
#: open-water tile whose NDVI runs -0.33 to -0.06 as "vegetation typically peaks in
#: August, 2 separate green-up periods, consistent with more than one growing
#: season" is what comes out. It is the same failure the single-date sheet guards
#: against over Antarctica: a
#: gradient of something that is not there is noise wearing a fact's clothes. The
#: threshold is on the series MAXIMUM, not the mean, so a genuinely seasonal tile
#: that is bare for half the year still qualifies.
NDVI_PRESENT_MAX = 0.20

#: A burn or clearance statement needs land. Over a 99%-water burst the NBR swings
#: with sun glint and wind roughening, and an ungated 0.28 drop reads as "consistent
#: with fire, flood or clearance".
MAX_WATER_FOR_DISTURBANCE = 0.50


class SeriesKind(str, Enum):
    CLIMATOLOGY = 'climatology'   # one frame per calendar month, mixed years
    BURST = 'burst'               # a few weeks at a short cadence


@dataclass
class SeriesFacts:
    cell: str
    part: str
    dates: list[str] = field(default_factory=list)
    clear: list[float] = field(default_factory=list)
    ndvi: list[float | None] = field(default_factory=list)
    water: list[float | None] = field(default_factory=list)
    snow: list[float | None] = field(default_factory=list)
    nbr: list[float | None] = field(default_factory=list)
    #: (frame index, fraction of the tile whose PER-PIXEL dNBR against the previous
    #: readable frame clears `DNBR_PIXEL_SEVERE`). Empty where fewer than two frames
    #: are readable.
    dnbr_area: list[tuple[int, float]] = field(default_factory=list)
    era5: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def n_frames(self) -> int:
        return len(self.dates)

    def usable(self) -> list[int]:
        return [i for i, c in enumerate(self.clear) if c >= FRAME_CLEAR_MIN]

    def month_of(self, i: int) -> str:
        return MONTHS[int(self.dates[i][5:7]) - 1]

    def day_of(self, i: int) -> str:
        """`5 March 2020`, for a window too short for month names to mean anything."""
        d = self.dates[i]
        return f'{int(d[8:10])} {MONTHS[int(d[5:7]) - 1]} {d[:4]}'

    def label_of(self, i: int) -> str:
        return self.month_of(i) if self.kind is SeriesKind.CLIMATOLOGY \
            else self.day_of(i)

    @property
    def span_days(self) -> int:
        if len(self.dates) < 2:
            return 0
        ds = sorted(datetime.date.fromisoformat(d) for d in self.dates)
        return (ds[-1] - ds[0]).days

    @property
    def kind(self) -> SeriesKind:
        """Read off the DATES, never the part name."""
        return SeriesKind.CLIMATOLOGY if self.span_days > CLIMATOLOGY_SPAN_DAYS \
            else SeriesKind.BURST

    @property
    def n_years(self) -> int:
        return len({d[:4] for d in self.dates})

    def has_vegetation(self) -> bool:
        """Is there anything green here at any point in the series?"""
        vals = [self.ndvi[i] for i in self.usable() if self.ndvi[i] is not None]
        return bool(vals) and max(vals) >= NDVI_PRESENT_MAX

    def is_mostly_land(self) -> bool:
        vals = [self.water[i] for i in self.usable() if self.water[i] is not None]
        if not vals:
            return True
        return float(np.median(vals)) < MAX_WATER_FOR_DISTURBANCE


def series_facts(tile: Tile, read_pixels: bool = True) -> SeriesFacts | None:
    """Per-frame indices for one tile's Sentinel-2 series.

    Returns None for a monotemporal tile: one frame is not a series, and a
    one-element trajectory would make every statement about "the month of peak
    greenness" trivially true.

    Dates and clear fractions are ELLIOT-X-EXT's per-acquisition OmniCloudMask rows;
    the indices are computed from ELLIOT's S2 frames; ERA5 is X-EXT's per-acquisition
    columns.
    """
    tile = fsheet.as_tile(tile)
    if tile.n_frames <= 1:
        return None
    cloud = [r for r in tile.s2_ext_frames if r.get('ml:clear') is not None]
    if not cloud:
        return None
    n = min(tile.n_frames, len(cloud))
    sf = SeriesFacts(cell=tile.cell, part=tile.part.name,
                     dates=[cloud[i]['ml:date'] for i in range(n)],
                     clear=[float(cloud[i]['ml:clear']) for i in range(n)])
    if tile.n_frames != len(cloud):
        sf.notes.append(f'{tile.n_frames} rasters against {len(cloud)} cloud records; '
                        f'using the first {n}')
    if not read_pixels:
        return sf

    stack = tile.read('s2')
    scale = tile.scale('s2')
    bands = [fsheet.S2[k] for k in ('green', 'red', 'nir', 'swir16', 'swir22')]
    prev_nbr = prev_ok = prev_water = prev_snow = None
    for i in range(n):
        if stack is None or i >= stack.shape[0] or stack.shape[1] < 13:
            for lst in (sf.ndvi, sf.water, sf.snow, sf.nbr):
                lst.append(None)
            continue
        cube = stack[i, bands].astype(np.float32) * scale
        green, red, nir, swir16, swir22 = cube
        ok = np.isfinite(cube).all(axis=0) & (red > 0)
        if ok.sum() < 1000:
            for lst in (sf.ndvi, sf.water, sf.snow, sf.nbr):
                lst.append(None)
            continue

        def nd(a, b):
            with np.errstate(invalid='ignore', divide='ignore'):
                return np.where(ok & (np.abs(a + b) > 1e-6), (a - b) / (a + b), np.nan)

        ndvi, ndsi = nd(nir, red), nd(green, swir16)
        mndwi, nbr = nd(green, swir16), nd(nir, swir22)
        # Snow FIRST, then water -- the ordering the single-date sheet also needs, and
        # it matters more over a year: a snow-covered winter frame read as flooding
        # produces a water trajectory that peaks in January.
        snow = (ndsi > fsheet.NDSI_SNOW) & (nir > 0.11)
        water = (mndwi > fsheet.MNDWI_WATER) & ~snow
        sf.ndvi.append(float(np.nanmean(ndvi)))
        sf.water.append(float(np.nanmean(water)))
        sf.snow.append(float(np.nanmean(snow)))
        sf.nbr.append(float(np.nanmean(nbr)))

        # PER-PIXEL dNBR against the previous readable frame. Only the previous
        # array is held. Water and snow are excluded on BOTH dates: a
        # frozen-then-thawed lake and a glinting sea both produce large dNBR and
        # neither has burned.
        if prev_nbr is not None and sf.clear[i] >= FRAME_CLEAR_MIN:
            land = ok & prev_ok & ~water & ~snow & ~prev_water & ~prev_snow
            if land.sum() >= 1000:
                d = np.where(land, prev_nbr - nbr, np.nan)
                sf.dnbr_area.append(
                    (i, float(np.nanmean(d >= DNBR_PIXEL_SEVERE))))
        if sf.clear[i] >= FRAME_CLEAR_MIN:
            prev_nbr, prev_ok = nbr, ok
            prev_water, prev_snow = water, snow

    sf.era5 = era5_series(tile, n)
    return sf


def era5_series(tile: Tile, n: int) -> list[dict]:
    """ERA5 at each S2 acquisition, from ELLIOT-X-EXT. `{}` where a frame has none."""
    out = []
    for r in tile.s2_ext_frames[:n]:
        t, p, h = (r.get('ml:era5_temperature_2m_c'), r.get('ml:era5_precipitation_mm'),
                   r.get('ml:era5_relative_humidity_pct'))
        out.append({} if t is None or p is None else {
            'temperature_2m_c': float(t), 'precipitation_mm': float(p),
            'relative_humidity_pct': None if h is None else float(h)})
    return out + [{}] * (n - len(out))


# --------------------------------------------------------------------------------
# Derived statements.
# --------------------------------------------------------------------------------
def green_up_cycles(ndvi: list[float], prominence: float = CYCLE_PROMINENCE) -> int:
    """How many distinct green-ups the trajectory contains.

    One is a season or a single crop; two is double-cropping, and that label is what
    this module exists for. Counted as rises clearing `prominence` above a preceding
    trough, NOT as local maxima: a noisy flat series has four local maxima and no
    cycles at all.

    The scan WRAPS, because a twelve-frame calendar year does: a crop that greens up
    in November and peaks in February is one cycle, and a linear scan reports two
    halves of one.
    """
    v = [x for x in ndvi if x is not None]
    if len(v) < 4:
        return 0
    cycles, trough, rising = 0, v[0], False
    for x in v[1:] + v[:1]:
        if x < trough:
            trough, rising = x, False
        elif x - trough >= prominence and not rising:
            cycles += 1
            rising = True
        if rising and x < trough + prominence / 2:
            trough, rising = x, False
    return cycles


def temporal_block(sf: SeriesFacts) -> str | None:
    """The `temporal` fact-sheet line for one series, or None when there is nothing.

    Two vocabularies, chosen by `sf.kind`, because the two series parts describe
    different things. A CLIMATOLOGY speaks in months and in typical behaviour; a
    BURST speaks in dates over a named number of days and never says "season".

    Every clause is gated on the frames supporting it actually being clear, and the
    coverage note comes FIRST, because a phenology read off four visible frames out
    of twelve is a different claim from one read off twelve and the caption has to
    be able to say which it is.
    """
    usable = sf.usable()
    if len(usable) < 3:
        return None
    clim = sf.kind is SeriesKind.CLIMATOLOGY
    lab = sf.label_of
    bits = []

    # ---- what this series IS. Stated every time, because the reader cannot tell a
    # six-year climatology from a year from twelve numbers.
    if clim:
        head = (f'a seasonal composite: {sf.n_frames} acquisitions, roughly one per '
                f'calendar month but drawn from {sf.n_years} different years, so it '
                f'describes the TYPICAL year and not any single one')
    else:
        head = (f'a burst: {sf.n_frames} acquisitions over {sf.span_days} days')
    bits.append(head)

    if len(usable) < sf.n_frames:
        # Deduplicated with counts: two frames in the same month is normal in the
        # climatology, and "January, January" reads as a bug rather than as a fact.
        hidden = collections.Counter(lab(i) for i in range(sf.n_frames) if i not in usable)
        txt = ', '.join(f'{k} (x{v})' if v > 1 else k for k, v in hidden.items())
        bits.append(f'{len(usable)} of {sf.n_frames} frames are clear enough to read '
                    f'({txt} obscured)')

    nd = [(i, sf.ndvi[i]) for i in usable if sf.ndvi[i] is not None]
    if len(nd) >= 3 and not sf.has_vegetation():
        # Say the absence rather than describing a trajectory of nothing. An open
        # water body still has an NDVI series and it still has a maximum.
        bits.append('no vegetation signal at any point in the series')
    elif len(nd) >= 3:
        vals = [v for _, v in nd]
        hi = max(nd, key=lambda p: p[1])[0]
        lo = min(nd, key=lambda p: p[1])[0]
        if max(vals) - min(vals) >= NDVI_AMPLITUDE_MIN:
            if clim:
                bits.append(f'vegetation typically peaks in {lab(hi)} and is lowest '
                            f'in {lab(lo)} (NDVI {min(vals):+.2f} to {max(vals):+.2f})')
                cycles = green_up_cycles(vals)
                if cycles >= 2:
                    # "in the year" is NOT available here: the frames are from
                    # different years, so two peaks in the seasonal cycle is what can
                    # be said, and double-cropping is what it is CONSISTENT with.
                    bits.append(f'{cycles} separate green-up periods in the seasonal '
                                f'cycle, consistent with more than one growing season')
            else:
                bits.append(f'vegetation is highest on {lab(hi)} and lowest on '
                            f'{lab(lo)} (NDVI {min(vals):+.2f} to {max(vals):+.2f}) '
                            f'over these {sf.span_days} days')
        elif len(nd) >= MIN_FRAMES_FOR_ABSENCE:
            # An ABSENCE claim, and only with enough frames to be an absence rather
            # than a gap: three clear frames of twelve can show 0.63-0.67 as
            # "near-constant" for a series that actually runs 0.02 to 0.67.
            bits.append(f'vegetation is near-constant across the series '
                        f'(NDVI {min(vals):+.2f} to {max(vals):+.2f})')

    wt = [(i, sf.water[i]) for i in usable if sf.water[i] is not None]
    if len(wt) >= 3:
        vals = [v for _, v in wt]
        if max(vals) - min(vals) >= 0.05:
            hi = max(wt, key=lambda p: p[1])[0]
            bits.append(f'open water is most extensive {"in" if clim else "on"} '
                        f'{lab(hi)} ({100 * max(vals):.0f}% of the tile against '
                        f'{100 * min(vals):.0f}% at its lowest)')

    snowy = [(i, sf.snow[i]) for i in usable
             if sf.snow[i] is not None and sf.snow[i] > 0.05]
    if snowy:
        peak = max(snowy, key=lambda p: p[1])
        names = collections.Counter(lab(i) for i, _ in snowy)
        bits.append(f'snow is present {"in" if clim else "on"} '
                    f'{", ".join(names)}, deepest {"in" if clim else "on"} '
                    f'{lab(peak[0])} at {100 * peak[1]:.0f}% of the tile')

    # AREA, not mean. "8% of the tile dropped past the severity threshold" is a
    # claim somebody can go and check on the pixels; "the tile mean fell 0.28" is
    # not, and over a 111 km2 footprint it is dominated by whatever the other 92%
    # was doing.
    if sf.dnbr_area and sf.is_mostly_land():
        worst = max(sf.dnbr_area, key=lambda p: p[1])
        if worst[1] >= DNBR_AREA_MIN:
            where = f'{100 * worst[1]:.0f}% of the tile'
            if clim:
                bits.append(f'{where} is much darker in the burn ratio in '
                            f'{lab(worst[0])} than in the preceding month, a '
                            f'seasonal contrast rather than a dated event')
            else:
                bits.append(f'{where} dropped past the burn-severity threshold by '
                            f'{lab(worst[0])}, consistent with fire, flood or '
                            f'clearance inside this window')

    if sf.era5 and any(sf.era5):
        temps = [(i, e['temperature_2m_c']) for i, e in enumerate(sf.era5)
                 if e and i in usable]
        if len(temps) >= 3:
            hi, lo = max(temps, key=lambda p: p[1]), min(temps, key=lambda p: p[1])
            if hi[1] - lo[1] >= 2.0:
                bits.append(f'air temperature at overpass runs {lo[1]:.0f} C '
                            f'{"in" if clim else "on"} {lab(lo[0])} to {hi[1]:.0f} C '
                            f'{"in" if clim else "on"} {lab(hi[0])}')
        rain = [(i, e['precipitation_mm']) for i, e in enumerate(sf.era5)
                if e and i in usable]
        if rain:
            wettest = max(rain, key=lambda p: p[1])
            if wettest[1] >= 0.5:
                bits.append(f'rain was falling at the {lab(wettest[0])} overpass '
                            f'({wettest[1]:.1f} mm in that hour)')

    if len(bits) <= 1:          # the header alone is not a description
        return None
    return 'SERIES (Sentinel-2) ' + '; '.join(bits)


#: The segment to add to `factsheet.SEGMENTS` for a sample that carries a series.
#: Declared here rather than written into that module so a monotemporal-only consumer
#: never imports the raster-reading path.
TEMPORAL_SEGMENT = {
    'requires': {'s2', 'series'},
    'brief': ('What only the stack shows: when vegetation peaks and troughs, whether '
              'there is more than one green-up, how water and snow move through the '
              'year, and any abrupt change. Name MONTHS, never frame numbers. Omit '
              'anything the cloudy frames cannot support.'),
}
