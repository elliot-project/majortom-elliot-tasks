"""The facts store gives the same tasks and captions as the pixels."""
from conftest import needs_data

from elliot_tasks import caption as cap
from elliot_tasks import factstore as F
from elliot_tasks import tasks as T


def _signature(examples):
    return [(e.family, e.question, e.target, repr(e.meta)) for e in examples]


@needs_data
def test_a_stored_record_rebuilds_every_task(city, monthly):
    for tile in (city, monthly):
        record = F.decode(F.encode(F.compute(tile)))
        stored = T.TileFacts(cell=record['cell'], sheet=record['sheet'], series=record['series'],
                             tile=None, grids=record['grids'])
        live = T.tile_facts(tile)
        for epoch in range(3):
            for available in (None, {'s2'}, {'s2', 'dem', 'lc'}):
                assert _signature(T.examples_for(facts=stored, available=available, epoch=epoch)) == \
                    _signature(T.examples_for(facts=live, available=available, epoch=epoch))
        assert cap.caption(stored.sheet, 1) == cap.caption(live.sheet, 1)


@needs_data
def test_facts_reads_a_store(tmp_path, city):
    import pyarrow as pa
    import pyarrow.parquet as pq
    record = F.compute(city)
    path = F.shard_path(tmp_path, city.part.name, city.index)
    path.parent.mkdir(parents=True)
    pq.write_table(pa.table({'index': [city.index], 'cell': [record['cell']],
                             'facts': [F.encode(record)], 'error': [None]}), path)
    got = F.facts(city.part.name, city.index, root=str(tmp_path))
    assert got.cell == city.cell
    assert got.tile is None
    assert got.sheet.text() == T.tile_facts(city).sheet.text()
