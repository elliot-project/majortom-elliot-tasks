"""Reading ELLIOT and ELLIOT-X-EXT, fact sheets, tasks and the loader, on real tiles."""
import numpy as np
import pytest
from conftest import needs_data

pytestmark = needs_data


def test_parts_are_row_aligned(et):
    for name in et.PARTS:
        p = et.open_part(name)
        assert len(p.elliot) == len(p.ext)
        assert p.elliot.table.column('majortom:code_10km').to_pylist()[:50] == \
            p.ext.table.column('majortom:code_10km').to_pylist()[:50]


def test_tile_arrays_have_a_frame_axis(city, monthly):
    assert city.read('s2').shape == (1, 13, 1056, 1056)
    assert city.read('cloud_mask_s2').shape == (1, 1056, 1056)
    assert city.read('dem').shape == (1056, 1056)
    assert monthly.read('s2').shape[0] == 12 == monthly.n_frames
    assert monthly.read('s1').shape == (12, 2, 1056, 1056)
    assert len(monthly.s2_ext_frames) == 12


def test_fact_sheet_shrinks_with_modalities(et, city):
    fs = et.fact_sheet(city)
    assert {'s2', 'l8', 's1', 'dem', 'lc'} <= fs.available
    full = fs.text()
    for head in ('GRID CELL 284D_496L', 'LOCATION', 'Curitiba', 'SPECTRAL', 'THERMAL',
                 'RADAR', 'ELEVATION', 'NAMED FEATURES'):
        assert head in full
    s2_only = fs.text({'s2'})
    assert 'THERMAL' not in s2_only and 'RADAR' not in s2_only and 'ELEVATION' not in s2_only
    assert 'NAMED FEATURES' in fs.text(set())


def test_series_facts(et, monthly, burst):
    m = et.series_facts(monthly)
    assert m.kind is et.SeriesKind.CLIMATOLOGY and m.n_frames == 12
    b = et.series_facts(burst)
    assert b.kind is et.SeriesKind.BURST and b.n_frames == 6
    assert 'burst' in et.temporal_block(b)


@pytest.mark.parametrize('which', ['city', 'monthly', 'burst'])
def test_every_task_scores_its_own_target(et, request, which):
    tile = request.getfixturevalue(which)
    exs = et.examples_for(tile, caption='A caption supplied by the caller, long enough to count as prose here.')
    assert len(exs) >= 8
    for e in exs:
        s = et.score(e, e.target)
        assert s.parsed and s.wellformed and s.score == 1.0, (e.family, s)


def test_temporal_families_and_determinism(et, monthly, burst):
    fm = {e.family for e in et.examples_for(monthly)}
    assert {'peak_greenness', 'green_up_count', 'ndvi_series'} <= fm
    assert 'disturbance' in {e.family for e in et.examples_for(burst)}
    a = [e.question for e in et.examples_for(burst, epoch=3)]
    assert a == [e.question for e in et.examples_for(burst, epoch=3)]
    assert a != [e.question for e in et.examples_for(burst, epoch=4)]


def test_available_drops_families(et, city):
    fams = {e.family for e in et.examples_for(city, available=set())}
    assert not fams & {'ndvi_grid', 'cover_grid', 'relief', 'dominant_cover'}
    assert 'locate_bbox' in fams


def test_torch_dataset(et):
    pytest.importorskip('torch')
    from elliot_tasks.torch import ElliotTaskDataset
    ds = ElliotTaskDataset('burst', [5], available=('s2', 'dem'))
    arrays, conv = ds[0]
    assert set(arrays) == {'s2', 'dem'} and tuple(arrays['s2'].shape) == (6, 13, 1056, 1056)
    assert np.isfinite(arrays['s2'].numpy()).any()
    assert conv and conv[0]['role'] == 'user' and len(conv) % 2 == 0
    text = ' '.join(t['content'] for t in conv)
    assert 'Sentinel-1' not in text and 'thermal' not in text.lower()
    assert ds[0][1] == conv
