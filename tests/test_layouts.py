"""The same tiles read from every supported layout give the same facts and tasks.

These tests are of the two-dataset layout (ELLIOT_ROOT and ELLIOT_X_EXT_ROOT both set).
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

pytestmark = [needs_data, pytest.mark.skipif(
    not os.environ.get('ELLIOT_X_EXT_ROOT'), reason='the two-dataset layout needs ELLIOT_X_EXT_ROOT')]
N = 6


@pytest.fixture(autouse=True)
def restore_roots():
    yield
    import elliot_tasks
    elliot_tasks.configure()                 # back to the environment's roots


def _signature(et, tile):
    return (tile.cell, et.fact_sheet(tile).text(),
            [(e.family, e.question, e.target) for e in et.examples_for(tile)])


def _export(source, output, first: int, stop: int) -> None:
    """Copy samples [first, stop) with the TACO writer, in either export API: a SQL
    selection (taco 0.14 and later) or a table of sample ids (earlier)."""
    import inspect

    import taco
    if 'sql' in inspect.signature(taco.export).parameters:
        taco.export(source, output, sql=f'SELECT * FROM sample WHERE "taco:sample_index" '
                                        f'>= {first} AND "taco:sample_index" < {stop}')
    else:
        import pyarrow as pa
        taco.export(source, output, samples=pa.table({'sample_id': list(range(first, stop))}))


@pytest.fixture(scope='module')
def subset(tmp_path_factory):
    pytest.importorskip('cozip')                 # the TACO writer's zip backend
    import taco
    root = tmp_path_factory.mktemp('layouts')
    e_root, x_root = os.environ['ELLIOT_ROOT'], os.environ['ELLIOT_X_EXT_ROOT']
    _export(f'{e_root}/burst', root / 'e' / 'burst', 0, N)
    for k, (a, b) in enumerate([(0, N // 2), (N // 2, N)]):
        _export(f'{x_root}/burst', root / 'x' / 'burst' / f'elliot-x-ext-burst.{k:04d}.zip', a, b)
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
    huggingface_hub = pytest.importorskip('huggingface_hub')

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


def _as_published(subset, tmp_path):
    """The subset as the ELLIOT-Pretrain release is published: no ml:contract."""
    import json
    rel = tmp_path / 'release' / 'burst'
    rel.mkdir(parents=True)
    coll = json.loads((subset / 'e' / 'burst' / 'COLLECTION.json').read_text())
    coll.pop('ml:contract')
    (rel / 'COLLECTION.json').write_text(json.dumps(coll))
    for name in ('DATA', 'METADATA'):
        (rel / name).symlink_to(subset / 'e' / 'burst' / name)
    return tmp_path / 'release'


def _overlay_zip(subset, path):
    import zipfile
    src = subset / 'e' / 'burst'
    with zipfile.ZipFile(path, 'w') as z:
        z.write(src / 'COLLECTION.json', 'burst/COLLECTION.json')
        for f in sorted((src / 'METADATA').glob('*.parquet')):
            z.write(f, f'burst/METADATA/{f.name}')
    return path


def test_release_without_contract_uses_the_default_zip_overlay(subset, reference, tmp_path,
                                                               monkeypatch):
    import elliot_tasks as et
    from elliot_tasks import data
    monkeypatch.setattr(data, 'ELLIOT_METADATA_OVERLAY',
                        str(_overlay_zip(subset, tmp_path / 'overlay.zip')))
    et.configure(elliot=str(_as_published(subset, tmp_path)), ext=str(subset / 'x'),
                 cache_dir=str(tmp_path / 'cache'))
    with pytest.warns(UserWarning, match='no ml:contract'):
        assert _signature(et, et.tile('burst', 1)) == reference[1]


def test_release_without_contract_and_no_overlay_says_why(subset, tmp_path, monkeypatch):
    import elliot_tasks as et
    from elliot_tasks import data
    monkeypatch.setattr(data, 'ELLIOT_METADATA_OVERLAY', None)
    et.configure(elliot=str(_as_published(subset, tmp_path)), ext=str(subset / 'x'),
                 cache_dir=str(tmp_path / 'cache'))
    with pytest.raises(ValueError, match='metadata overlay'):
        et.open_part('burst')
