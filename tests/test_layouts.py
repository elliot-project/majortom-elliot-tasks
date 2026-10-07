"""The same tiles read from every supported layout give the same facts and tasks.

Builds, from the real data, a six-sample burst subset of ELLIOT-Pretrain (FOLDER) and
of ELLIOT-X-EXT as two TACO zip parts with a `.tacocat/` -- the layout of the
Hugging Face release -- then reads it locally, through a stubbed Hugging Face
download, and with a metadata overlay.
"""
import fnmatch
import os
import shutil
from pathlib import Path

import pytest
from conftest import needs_data

pytestmark = needs_data
N = 6


@pytest.fixture(autouse=True)
def restore_roots():
    yield
    import elliot_tasks
    elliot_tasks.configure()                 # back to the environment's roots


def _signature(et, tile):
    return (tile.cell, et.fact_sheet(tile).text(),
            [(e.family, e.question, e.target) for e in et.examples_for(tile)])


@pytest.fixture(scope='module')
def subset(tmp_path_factory):
    import pyarrow as pa
    import taco
    root = tmp_path_factory.mktemp('layouts')
    e_root, x_root = os.environ['ELLIOT_ROOT'], os.environ['ELLIOT_X_EXT_ROOT']
    taco.export(f'{e_root}/burst', root / 'e' / 'burst',
                samples=pa.table({'sample_id': list(range(N))}))
    for k, (a, b) in enumerate([(0, N // 2), (N // 2, N)]):
        taco.export(f'{x_root}/burst', root / 'x' / 'burst' / f'elliot-x-ext-burst.{k:04d}.zip',
                    samples=pa.table({'sample_id': list(range(a, b))}))
    taco.consolidate(sorted((root / 'x' / 'burst').glob('*.zip')))
    return root


@pytest.fixture(scope='module')
def reference(subset):
    import elliot_tasks as et
    et.configure()
    return {i: _signature(et, et.tile('burst', i)) for i in (1, N - 1)}


def test_tacocat_zip_parts(subset, reference):
    import elliot_tasks as et
    et.configure(elliot=str(subset / 'e'), ext=str(subset / 'x'))
    assert len(et.open_part('burst')) == N
    for i, want in reference.items():
        assert _signature(et, et.tile('burst', i)) == want


def test_hugging_face_fetches_one_zip_part(subset, reference, tmp_path, monkeypatch):
    import huggingface_hub

    import elliot_tasks as et
    calls = []

    def fake_snapshot_download(repo_id, repo_type, allow_patterns, local_dir):
        calls.append(list(allow_patterns))
        repo = subset / 'x'
        for f in repo.rglob('*'):
            rel = f.relative_to(repo).as_posix()
            if f.is_file() and any(fnmatch.fnmatch(rel, p) for p in allow_patterns):
                dest = Path(local_dir) / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dest)
        return local_dir

    monkeypatch.setattr(huggingface_hub, 'snapshot_download', fake_snapshot_download)
    et.configure(elliot=str(subset / 'e'), ext='hf://example-org/elliot-x-ext',
                 cache_dir=str(tmp_path))
    assert _signature(et, et.tile('burst', 1)) == reference[1]
    zips = sorted(p.name for p in tmp_path.rglob('*.zip'))
    assert zips == ['elliot-x-ext-burst.0000.zip']          # only the part holding row 1
    assert _signature(et, et.tile('burst', N - 1)) == reference[N - 1]
    assert len(list(tmp_path.rglob('*.zip'))) == 2


def test_metadata_overlay(subset, reference, tmp_path):
    import elliot_tasks as et
    et.configure(elliot=str(subset / 'e'), ext=str(subset / 'x'),
                 elliot_metadata=str(subset / 'e'), cache_dir=str(tmp_path))
    assert _signature(et, et.tile('burst', 1)) == reference[1]
    assert any(p.is_symlink() for p in tmp_path.rglob('DATA'))
