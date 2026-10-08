"""The rule-based captioner: wording, modality licensing and the fact checker."""
from conftest import needs_data

from elliot_tasks import caption as cap
from elliot_tasks import caption_check as cc
from elliot_tasks import tasks as T


def test_fraction_words_follow_the_sheet_rounding():
    c = cap._Ctx('x', 0)
    assert cap.fraction_words(1.0, c) == 'all'
    assert cap.fraction_words(0.0, c) == 'none'
    assert cap.fraction_words(0.28, c).endswith('under three-tenths')
    assert cap.fraction_words(0.25, c).endswith('a quarter')


@needs_data
def test_caption_is_seeded_and_checker_clean(city):
    fs = T.fsheet.fact_sheet(city)
    texts = set()
    for seed in range(4):
        segments, claims = cap.caption_claims(fs, seed)
        text = '\n\n'.join(t for t in segments.values() if t)
        assert cc.check(fs.text(), text, claims) == []
        assert cap.caption(fs, seed) == cap.caption(fs, seed)
        texts.add(text)
    assert len(texts) == 4


@needs_data
def test_caption_keeps_only_licensed_paragraphs(city):
    fs = T.fsheet.fact_sheet(city)
    s2_only = cap.caption_segments(fs, 0, available={'s2'})
    assert set(s2_only) <= {'core', 'optical', 'sky'}
    assert 'thermal' in cap.caption_segments(fs, 0)


@needs_data
def test_caption_family_writes_one_without_a_stored_caption(city):
    facts = T.tile_facts(city, with_series=False)
    ex = {e.family: e for e in T.examples_for(facts=facts, available={'s2'})}
    assert ex['caption'].target
    assert 'Landsat' not in ex['caption'].target
    stored = T.examples_for(facts=facts, caption='A stored caption.')
    assert {e.family: e for e in stored}['caption'].target == 'A stored caption.'
