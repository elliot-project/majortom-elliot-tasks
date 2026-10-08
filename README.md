# majortom-elliot-tasks

Vision-language tasks over [Major TOM ELLIOT-Pretrain](https://source.coop/major-tom/elliot-pretrain)
and its extension [ELLIOT-X-EXT](https://huggingface.co/datasets/isp-uv-es/elliot-x-ext).

For each 10.56 km tile the package measures a **fact sheet** (land cover, spectral
indices, thermal, radar, terrain, cloud, weather, named OpenStreetMap features), and
from it generates **captions** and **16 task families** (grounding, grids, counts,
phenology, change), with **scorers** and a **PyTorch dataset**. Tasks are built on the
fly, with new wording every epoch, and only from the modalities the model is given.

## Data

| Dataset | Content | Link | Licence |
|---|---|---|---|
| ELLIOT-Pretrain | Sentinel-2 L1C, Landsat-8/9, Copernicus DEM, ESA WorldCover | [source.coop/major-tom/elliot-pretrain](https://source.coop/major-tom/elliot-pretrain) | CC-BY-SA-4.0 |
| ELLIOT-X-EXT | cloud masks, ERA5, Sentinel-1 RTC, OpenStreetMap, admin units | [huggingface.co/datasets/isp-uv-es/elliot-x-ext](https://huggingface.co/datasets/isp-uv-es/elliot-x-ext) | CC-BY-SA-4.0; OSM layers ODbL-1.0 |
| Metadata overlay | corrected metadata for ELLIOT-Pretrain | [elliot-pretrain-metadata-overlay.zip](https://huggingface.co/datasets/isp-uv-es/elliot-x-ext/blob/main/elliot-pretrain-metadata-overlay.zip) | CC-BY-SA-4.0 |

Both datasets have three parts, row-aligned (tile `i` is the same cell in both):
`monotemporal` (250,000 tiles), `monthly` (12,500 tiles × 12 frames) and `burst`
(16,666 tiles × 6 frames).

ELLIOT-Pretrain is read with the metadata overlay, which corrects its published
metadata (the pixels are unchanged). The package downloads and applies the overlay
automatically.

## Install

```bash
pip install "majortom-elliot-tasks[torch] @ git+https://github.com/elliot-project/majortom-elliot-tasks"
```

Python 3.12+. The data are read with the TACO reader from
[OscarPellicer/taco](https://github.com/OscarPellicer/taco), which pip builds from
source: it needs CMake, Ninja and a C++20 compiler (set `CC`/`CXX` if the default is
older).

## Get the data

Streaming needs no download; files are cached under `~/.cache/elliot-tasks` as they
are read:

```python
import elliot_tasks as et
et.configure(elliot='https://data.source.coop/major-tom/elliot-pretrain',
             ext='hf://isp-uv-es/elliot-x-ext')
```

To work from local copies, download ELLIOT-X-EXT (one part shown) and the overlay:

```bash
huggingface-cli download isp-uv-es/elliot-x-ext --repo-type dataset \
    --include "burst/*" "elliot-pretrain-metadata-overlay.zip" --local-dir elliot-x-ext
```

and ELLIOT-Pretrain from its [Source Cooperative page](https://source.coop/major-tom/elliot-pretrain).
Then:

```python
et.configure(elliot='/data/elliot-pretrain', ext='/data/elliot-x-ext',
             elliot_metadata='/data/elliot-x-ext/elliot-pretrain-metadata-overlay.zip')
```

The same three settings can be given as `ELLIOT_ROOT`, `ELLIOT_X_EXT_ROOT` and
`ELLIOT_METADATA`.

## Use

```python
tile = et.find('284D_496L')                # by cell, or et.tile('monthly', 352)
sheet = et.fact_sheet(tile)
print(sheet.text({'s2', 'dem'}))           # the facts S2 and the DEM support

from elliot_tasks import caption
print(caption.caption(sheet, seed=0))      # a rule-based caption; the seed picks the wording

for ex in et.examples_for(tile, available={'s2', 's1', 'dem'}, epoch=0):
    print(ex.family, '|', ex.question, '->', ex.target)
score = et.score(ex, model_answer)         # 0 to 1
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

`available` sets both the arrays loaded and the tasks asked: a model given only S2
and the DEM is never asked about radar or thermal. Arrays are float32 with NaN for
no data (reflectance scaled, L8 thermal in kelvin, S1 linear gamma-0, DEM in metres);
land cover and cloud masks are integer classes.

[`examples/tour.ipynb`](examples/tour.ipynb) walks through every modality and task.

## Tasks

| Family | Answer | Needs |
|---|---|---|
| `caption` | prose | - |
| `dominant_cover` | word | lc |
| `which_present` | choice | - |
| `locate_point`, `locate_bbox` | point, box | - |
| `features_all` | points | - |
| `ndvi_grid`, `cover_grid` | grid | s2, lc |
| `vegetated_fraction` | number | s2 |
| `acquisition` | record | - |
| `relief` | number | dem |
| `peak_greenness`, `green_up_count`, `clearest_frame` | choice | s2, series |
| `ndvi_series` | numbers | s2, series |
| `disturbance` | choice | s2, series |

`series` means a stack of several acquisitions (`monthly`, `burst`). Coordinates are
integers on a 0-1000 grid. Pass your own captions with `examples_for(tile, caption=...)`
or `ElliotTaskDataset(..., captions={cell: text})`.

## Tests

```bash
ELLIOT_ROOT=... ELLIOT_X_EXT_ROOT=... pytest     # without the data, only the data-free tests run
```

## Licence and citation

Code: MIT. Data: see the table above. If you use the data, cite Major TOM:

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
within the [ELLIOT](https://elliot-ai.eu) project (Horizon Europe, grant 101214398).
