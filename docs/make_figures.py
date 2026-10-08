"""Draw the README figures from the release, read through `taco.ml`.

    python docs/make_figures.py ROOT [--burst I] [--monthly I] [--monotemporal I]

ROOT is the release root (a local copy or https://data.source.coop/major-tom/elliot-pretrain).
Writes docs/images/{burst,monthly,monotemporal}.jpg.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import taco.ml
from matplotlib.colors import ListedColormap

OUT = Path(__file__).parent / 'images'
MASK = ListedColormap(['#2a9d8f', '#f4f1de', '#a8dadc', '#3d405b'])
WORLDCOVER = {10: '#006400', 20: '#ffbb22', 30: '#ffff4c', 40: '#f096ff', 50: '#fa0000',
              60: '#b4b4b4', 70: '#f0f0f0', 80: '#0064c8', 90: '#0096a0', 95: '#00cf75',
              100: '#fae6a0'}


def rgb(stack, scale):
    """True colour (red, green, blue = bands 3, 2, 1): each band's haze (its 1st
    percentile) removed, then one stretch for all three, so colours stay natural."""
    x = stack[[3, 2, 1]].astype(np.float32) * scale
    x = x - np.nanpercentile(x, 1, axis=(1, 2), keepdims=True)
    return np.clip(x / max(float(np.nanpercentile(x, 99)), 1e-6), 0, 1).transpose(1, 2, 0)


def db(vv):
    """VV backscatter in dB, stretched to its 2-98 percentiles."""
    x = 10 * np.log10(np.where(vv > 0, vv, np.nan))
    if np.isnan(x).all():
        return x
    lo, hi = np.nanpercentile(x, (2, 98))
    return np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)


def worldcover(lc):
    out = np.zeros((*lc.shape, 3))
    for code, colour in WORLDCOVER.items():
        out[lc == code] = matplotlib.colors.to_rgb(colour)
    return out


def scale(ds, name):
    return next(s for s in ds.contract.inputs if s.name == name).scale_factor or 1.0


def dates(times):
    return [pd.Timestamp(t, unit='ms') if isinstance(t, (int, np.integer)) else pd.Timestamp(t)
            for t in times]


def place(ds, index):
    row = ds.metadata(index)
    return f"{row['majortom:code_10km']}, {row.get('admin:state')}, {row.get('admin:country')}"


def series(ds, index, out, title):
    s = ds.read(index, slots=['s2', 'cloud_mask_s2', 's1'])
    n = s['s2'].array.shape[0]
    when = dates(s['s2'].times)
    fig, axes = plt.subplots(3, n, figsize=(2.6 * n, 8))
    for k in range(n):
        axes[0, k].imshow(rgb(s['s2'].array[k], scale(ds, 's2')))
        axes[0, k].set_title(when[k].strftime('%Y-%m-%d'), fontsize=9)
        axes[1, k].imshow(s['cloud_mask_s2'].array[k], cmap=MASK, vmin=0, vmax=3,
                          interpolation='nearest')
        axes[2, k].imshow(db(s['s1'].array[k, 0]), cmap='gray')
    for ax, label in zip(axes[:, 0], ('Sentinel-2', 'cloud mask', 'Sentinel-1 VV')):
        ax.set_ylabel(label)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f'{title}: {place(ds, index)}')
    fig.tight_layout()
    fig.savefig(out, dpi=80, pil_kwargs={'quality': 85})
    plt.close(fig)


def single(ds, index, out):
    s = ds.read(index)
    # Monotemporal layers are single rasters, (bands, H, W), not one-frame series.
    panels = [
        ('Sentinel-2', rgb(s['s2'].array, scale(ds, 's2')), {}),
        ('Landsat', rgb(s['l8'].array, scale(ds, 'l8')), {}),
        ('Sentinel-1 VV', db(s['s1'].array[0]), {'cmap': 'gray'}),
        ('cloud mask', np.squeeze(s['cloud_mask_s2'].array), {'cmap': MASK, 'vmin': 0, 'vmax': 3,
                                                              'interpolation': 'nearest'}),
        ('Copernicus DEM', np.squeeze(s['dem'].array), {'cmap': 'terrain'}),
        ('ESA WorldCover', worldcover(s['land_cover'].array), {'interpolation': 'nearest'}),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(3 * len(panels), 3.6))
    for ax, (title, image, kw) in zip(axes, panels):
        ax.imshow(image, **kw)
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    fig.suptitle(f'Monotemporal: {place(ds, index)}')
    fig.tight_layout()
    fig.savefig(out, dpi=80, pil_kwargs={'quality': 85})
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('root')
    ap.add_argument('--burst', type=int, default=120)
    ap.add_argument('--monthly', type=int, default=8000)
    ap.add_argument('--monotemporal', type=int, default=140510)
    ap.add_argument('--parts', default='burst,monthly,monotemporal')
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    root = a.root.rstrip('/')
    for part in a.parts.split(','):
        ds = taco.ml.Dataset(f'{root}/{part}')
        index = getattr(a, part)
        if part == 'monotemporal':
            single(ds, index, OUT / f'{part}.jpg')
        else:
            series(ds, index, OUT / f'{part}.jpg', part.capitalize())
        print(f'{part} {index} -> {OUT / part}.jpg')


if __name__ == '__main__':
    main()
