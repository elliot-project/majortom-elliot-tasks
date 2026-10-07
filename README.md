# majortom-elliot-tasks

Measured fact sheets and on-the-fly vision-language tasks over
**Major TOM ELLIOT-Pretrain** tiles, read together with the extension
**ELLIOT-X-EXT**.

For every 10.56 km tile the package builds:

- a **fact sheet**: every measured fact about the tile, each tagged with the modality
  that licenses it (S2, L8, S1, DEM, land cover, or none). Drop a modality from the
  encoder's input and the facts it licensed disappear with it;
- **series facts** for the monthly and burst parts: phenology, water and snow
  dynamics, abrupt change, cloud usability and weather. The facts are worded by what the
  series actually is, read off its dates: a multi-year climatology for `monthly`, a
  few weeks at five-day cadence for `burst`;
- **16 task families**, generated on demand with a deterministic seed per
  (cell, family, epoch), with paraphrase banks split 80/20 into train and eval
  templates by index;
- **scorers** for every answer shape;
- a thin **PyTorch dataset** that yields `(arrays, conversation)`.

Nothing is stored. Tasks are rebuilt from the data each time they are asked for, so
the wording changes from epoch to epoch.

## Data

| Dataset | Where | Licence |
|---|---|---|
| Major TOM ELLIOT-Pretrain | Source Cooperative [`major-tom/elliot-pretrain`](https://source.coop/major-tom/elliot-pretrain) (`https://data.source.coop/major-tom/elliot-pretrain`) | CC-BY-SA-4.0 |
| ELLIOT-X-EXT | Hugging Face `isp-uv-es/elliot-x-ext` | CC-BY-SA-4.0; `osm.parquet` and `admin.parquet` ODbL-1.0 (© OpenStreetMap contributors) |

Both come in three parts, `monotemporal` (250,000 tiles, one date), `monthly`
(12,500 tiles, 12 frames) and `burst` (16,666 tiles, 6 frames). The two datasets are
**row-aligned**: tile `i` of a part is the same Major TOM cell in both.

- **ELLIOT-Pretrain** provides Sentinel-2 L1C (13 bands), Landsat-8/9 (11 bands,
  including TIRS brightness temperature), Copernicus DEM and ESA WorldCover, plus
  per-tile context (admin names, human modification, climate).
- **ELLIOT-X-EXT** provides an OmniCloudMask cloud/shadow mask and its fractions for
  every S2 and L8 acquisition, ERA5 weather at every acquisition, one Sentinel-1 RTC
  frame per S2 frame, the tile's OpenStreetMap features and administrative units as
  GeoParquet, a WorldCover histogram and DEM statistics.

All reads go through the TACO reader (`taco.ml.Dataset`). A root holds the three
part folders, and each part folder can be either of two layouts:

- a TACO FOLDER container: `COLLECTION.json`, `METADATA/`, `DATA/`;
- TACO zip parts with their `.tacocat/` catalog. This is the layout of the Hugging
  Face release of ELLIOT-X-EXT.

A root can be:

| Root | Example | What is fetched |
|---|---|---|
| local directory | `/data/elliot-x-ext` | nothing |
| Hugging Face | `hf://isp-uv-es/elliot-x-ext` | a part's catalog when the part is opened, then only the zip part (or the files) holding a sample when it is read |
| HTTP mirror of a FOLDER layout | `https://data.source.coop/major-tom/elliot-pretrain` | a part's metadata, then a sample's files on first read |

Remote files are cached under `~/.cache/elliot-tasks`, or under `ELLIOT_TASKS_CACHE`
if that is set.

### ELLIOT-Pretrain needs corrected metadata

The current Source Cooperative release of ELLIOT-Pretrain cannot be read as it is
published, for two reasons.

- Its `COLLECTION.json` has no `ml:contract`, so a typed reader does not know what
  its bands, units and classes are.
- Its frame metadata has known errors. The worst is in `burst`, where nearly every
  tile's cell identity and acquisition rows belong to another tile.

So the package reads the release's **pixels** together with **corrected metadata**,
supplied as a metadata overlay. The overlay holds, for each part,
`<part>/COLLECTION.json` and `<part>/METADATA/*.parquet`, and nothing else. It is
published as `elliot-pretrain-metadata-overlay.zip` (96 MB) in the Hugging Face
dataset `isp-uv-es/elliot-x-ext`.

You do not need to pass the overlay. Whenever an ELLIOT-Pretrain root's own
`COLLECTION.json` has no `ml:contract`, the package downloads it once, uses it, and
warns that it did:

```python
et.configure(elliot='https://data.source.coop/major-tom/elliot-pretrain',
             ext='hf://isp-uv-es/elliot-x-ext')
# UserWarning: ... this release of Major TOM ELLIOT-Pretrain has no ml:contract and
# carries the old frame metadata, so its corrected metadata is used instead, from
# hf://isp-uv-es/elliot-x-ext/elliot-pretrain-metadata-overlay.zip ...
```

To use another overlay, pass `elliot_metadata=` (or set `ELLIOT_METADATA`). It can
be a directory or a zip file, local or `hf://<org>/<repo>/<path>`. A root whose
metadata already has an `ml:contract` is read as it is.

### What the ELLIOT-Pretrain release would need to change

The overlay becomes unnecessary once the release's own metadata matches it. Only
metadata changes; no raster has to be rewritten. For each part (`monotemporal`,
`monthly`, `burst`):

1. **`COLLECTION.json`**
   - Add `"taco:version": "3.0.0"`.
   - Add an `ml:contract` declaring four input slots:
     - `s2`: 13 bands, `scale_factor` 1e-4, TOA reflectance.
     - `l8`: 11 bands, `scale_factor` 1e-4. B1-B9 are reflectance and B10-B11
       brightness temperature in kelvin. The slot is optional in `monthly`, where
       312 L8 rows are entirely null.
     - `dem`: metres. No nodata value; 0 is a real elevation.
     - `land_cover`: ESA WorldCover classes, 0 = no data.
   - Rename the metadata levels from `collection`, `sample/s2`, `sample/l8` to
     `sample`, `children`, `children/s2`, `children/l8`, `children/dem`,
     `children/lc`.
   - Correct the descriptions of five columns:
     - `climate:temperature` is kelvin, not °C.
     - `climate:precipitation` is metres per year, not mm per year.
     - `soil:carbon`, `soil:bulk_density` and `soil:ph` are OpenLandMap's integer
       encodings (×5 g/kg, ×10 kg/m³, pH ×10).
     - `socio:gdp` is not per capita.
   - Monotemporal only: say that the `lc` rasters write WorldCover's no data as 80
     (permanent water), where `monthly` and `burst` write 0.
2. **`METADATA/` file names**, following the TACO v3 levels:
   - `collection.parquet` becomes `sample.parquet`;
   - `sample.parquet` becomes `children.parquet`;
   - `sample__{s2,l8,dem,lc}.parquet` become `children__{s2,l8,dem,lc}.parquet`.
3. **Columns**
   - Rename `geometry` to `ml:geometry`, in the sample level and in the S2 and L8
     frame levels.
   - In the S2 and L8 frame levels, rename `stac:crs`, `stac:geotransform`,
     `stac:tensor_shape`, `stac:time_start` and `stac:time_end` to `ml:crs`,
     `ml:geotransform`, `ml:tensor_shape`, `ml:time_start` and `ml:time_end`.
   - Add `ml:tensor_shape` to the DEM and land-cover levels.
4. **Values**
   - **`burst`, cell order.** The cell block of the sample level (`majortom:code_*`,
     `ml:geometry`, and every soil, climate, socio, terrain and admin column) is
     joined to the wrong `DATA/` directories: 16,665 of 16,666 rows. The `window:*`
     columns are permuted the same way.
   - **`burst`, frame rows.** Every per-acquisition column of the S2 and L8 frame
     rows (geometry, times, dates, product ids, scores, nodata, angles) is permuted
     the same way: 99,822 of 99,996 S2 rows carry another tile's dates. The fix
     moves them, frame for frame. CRS, geotransform and tensor shape stay, because
     they are read off the files.
   - **`monotemporal`, S2 timestamps.** 15,116 S2 `time_start`/`time_end` values are
     in microseconds, not milliseconds. Divide them by 1,000.
   - **`monotemporal`, frame geometry.** S2 geotransforms are wrong on 247,040 of
     250,000 frames, with origins hundreds of km off. Take the CRS, geotransform and
     tensor shape of every S2 and L8 frame from the file's own header.
   - **`monotemporal`, paths and order.** The land-cover level's paths point to
     `wc/data.tif`, a byte-identical duplicate of `lc/data.tif` (verified on all
     250,000 tiles); point them to `lc/data.tif`. The L8 level is in scrambled row
     order (row 0 is tile 225241); order it like its parent level.
   - **`monthly` and `burst`, paths.** The DEM and land-cover levels give directory
     paths (`0/dem`, `0/lc`) where the row is the file (`0/dem/main.tif`,
     `0/lc/main.tif`).
   - **All parts, soil.** Where the five soil columns are all 0 (exactly the rows
     with pH 0: open water, Antarctica, Greenland), store null.

Once a release carries this metadata, `taco.ml.Dataset` opens it directly and the
package stops using the overlay automatically. The corrected files in the overlay
can be adopted as they are: each `COLLECTION.json` lists these corrections in
`ml:metadata_revision`.

## Install

```bash
pip install "majortom-elliot-tasks[torch] @ git+https://github.com/elliot-project/majortom-elliot-tasks"
# or, from a checkout
pip install -e ".[torch,examples,test]"
```

The package needs Python 3.12 or later. It reads through the `taco.ml` reader of the
TACO fork (`taco-eo[ml]` from `OscarPellicer/taco`, `main`). That reader is proposed
upstream to [asterisk-labs/taco](https://github.com/asterisk-labs/taco), and the
dependency will point there once it is merged.

pip builds the fork from source, which needs:

- **CMake** (3.21 or later) and **Ninja**, on `PATH`;
- **a C++20 compiler**, such as GCC 11+ or Clang 14+. Point `CC` and `CXX` at it if
  the system default is older:

```bash
CC=/path/to/gcc CXX=/path/to/g++ pip install -e ".[torch,test]"
```

## Quick start

```python
import elliot_tasks as et

et.configure(elliot='/path/to/elliot-pretrain', ext='/path/to/elliot-x-ext')
# or set ELLIOT_ROOT, ELLIOT_X_EXT_ROOT (and optionally ELLIOT_METADATA)

tile = et.find('284D_496L')               # by cell code, or et.tile('monthly', 352)
sheet = et.fact_sheet(tile)
print(sheet.text({'s2', 'dem'}))          # only what S2 and the DEM license

for ex in et.examples_for(tile, epoch=0):
    print(ex.family, '|', ex.question, '->', ex.target)

score = et.score(ex, model_answer)        # Score(parsed, wellformed, score, details)
```

```python
from torch.utils.data import DataLoader
from elliot_tasks.torch import ElliotTaskDataset, collate

ds = ElliotTaskDataset('monthly', available=('s2', 's1', 'dem'))
loader = DataLoader(ds, batch_size=4, num_workers=4, collate_fn=collate)
for epoch in range(3):
    ds.set_epoch(epoch)
    for arrays, conversations in loader:
        ...
```

`available` sets both which arrays are loaded and which tasks are asked: a model fed
S2 and the DEM is never asked about radar or thermal. Arrays are float32 with nodata
as NaN: S2/L8 reflectance (×1e-4 applied; L8 bands 9 and 10 in kelvin), S1 linear
gamma-0, DEM in metres. Land cover and cloud masks are int64 class codes. The
per-acquisition modalities carry a leading frame axis.

[`examples/tour.ipynb`](examples/tour.ipynb) shows each part, plots every modality,
shows how the fact sheet shrinks, generates every family (including the temporal ones
on a monthly and a burst tile), packs a multi-turn conversation, and scores answers.

## Task families

| Family | Answer shape | Requires | Frozen-decoder feasibility |
|---|---|---|---|
| `caption` | prose | - | high |
| `dominant_cover` | word | lc | high |
| `which_present` | choice | - | high |
| `locate_point` | point | - | high |
| `locate_bbox` | bbox | - | high |
| `features_all` | points_multi | - | low |
| `ndvi_grid` | grid | s2 | low |
| `cover_grid` | grid | lc | low |
| `vegetated_fraction` | derived (number_array -> number) | s2 | high |
| `acquisition` | record | - | high |
| `relief` | number | dem | medium |
| `peak_greenness` | choice | s2, series | high |
| `green_up_count` | choice | s2, series | high |
| `clearest_frame` | choice | s2, series | high |
| `ndvi_series` | number_array | s2, series | medium |
| `disturbance` | derived (number_array -> choice) | s2, series | medium |

- `series` is a pseudo-modality: a stack of more than one acquisition.
- Coordinates are integers on a 0-1000 grid.
- A grid prompt always states its size and its cell-coverage threshold.
- A derived answer gives its evidence before its answer.
- `examples_for(..., max_feasibility=Feasibility.HIGH)` keeps only the shapes that a
  frozen language model emits reliably. `shapes.py` gives the measurement behind
  each rating.

## Scoring

Each scorer returns `parsed`, `wellformed` and `score` (0 to 1). An answer that does
not parse scores 0.

| Shape | Rule |
|---|---|
| word, choice | exact match after Unicode, case and punctuation folding; well formed when it is one of the offered options |
| number | the first number, within tolerance (relief ±10 m or ±10%) |
| number_array | share of positions within tolerance; a null target matches only a null |
| point | inside the reference object's box (else within 100 grid units) |
| bbox | IoU (hit at ≥ 0.5) |
| points_multi | F1 of a one-to-one greedy match within 100 grid units, order-free |
| grid | share of cells right: exact class for land cover, ±5 NDVI×100 for NDVI; the shape must match |
| record | share of keys right (date exact, solar elevation ±2°) |
| derived | the answer's score; the evidence's score is reported in `details` |
| prose | token F1 against the reference |

The tolerances are in `scoring.TOLERANCE`.

## What has no public source

Every fact sheet field and every family was ported from an internal version that
read private hydration databases. Facts now come only from the two public datasets.
Measured parity: on 27 random monotemporal tiles, 16 fact sheets are byte-identical.
On every other tile the only differences are the two listed below:

- **Solar elevation** is now 90° minus the Sentinel-2 granule's mean solar zenith. It
  differs from the internal value by at most 1°.
- **ERA5 weather** is ELLIOT-X-EXT's, read at the hour nearest the S2 acquisition. It
  now exists for every tile, and on some tiles its values differ slightly from the
  internal ones.

On three tiles every generated task was identical. On four monthly tiles the series
facts were identical. On burst tiles they differ only in the dates, which now come
from the corrected acquisition dates.

Dropped, or left to the caller:

- **Captions.** There is no public caption set. The `caption` family produces an
  example only when you pass a caption (`examples_for(tile, caption=...)`, or
  `captions={cell: text}` to the dataset).
- **Population density** (`socio:population`) is deliberately not stated. Read as
  people per km² it is implausible for cities, and its unit is not documented.
- **Mean annual precipitation** (`climate:precipitation`) is not stated, because its
  unit is not documented in the release.
- **Referring expressions** are not implemented, as in the internal version.

Changes in coverage:

- Cloud fractions and masks now exist for all three parts. Previously they existed
  only for monotemporal, so monthly and burst single-date facts are now cloud-masked
  too.
- For monthly and burst tiles the single-date facts describe acquisition 0 by default
  (`fact_sheet(tile, frame=k)`).

## Tests

```bash
pytest                                              # data tests skip without data
ELLIOT_ROOT=... ELLIOT_X_EXT_ROOT=... pytest        # all 45
```

- Without data, the tests check the scorers and the registry.
- With data, they read one city tile, one monthly tile and one burst tile.
  - They check that every family's own target scores 1.0.
  - They check how the fact sheet shrinks and that the seeds are deterministic.
  - They exercise the PyTorch dataset.
  - They build a six-tile subset as TACO zip parts with a `.tacocat/` and check that
    the zip layout, a stubbed Hugging Face download (which fetches only the zip part
    it needs), a metadata overlay and a zip overlay used by default for a release
    without `ml:contract` all give identical results.

## Licence and credits

The code is MIT-licensed (see `LICENSE`). The data keep their own licences (see
above). If you use the data, cite Major TOM:

```bibtex
@inproceedings{francis2024majortom,
  title={Major TOM: Expandable Datasets for Earth Observation},
  author={Francis, Alistair and Czerkawski, Mikolaj},
  booktitle={IGARSS 2024},
  pages={2935--2940},
  year={2024},
  doi={10.1109/IGARSS53475.2024.10640760}
}
```

Developed by the Image and Signal Processing Group (ISP), Universitat de València,
within the [ELLIOT](https://elliot-earth.eu) project (European Commission, Horizon
Europe, Grant 101214398).
