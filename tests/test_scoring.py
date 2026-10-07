"""Scorers on hand-made examples; no data needed."""
import json

from elliot_tasks import scoring as S
from elliot_tasks.shapes import Shape
from elliot_tasks.tasks import Example


def ex(family, shape, target, **meta):
    return Example(family, 'q', target if isinstance(target, str) else json.dumps(target),
                   shape, meta)


def test_parse_json_tolerates_fences_and_chatter():
    assert S.parse_json('```json\n{"a": 1}\n```') == {'a': 1}
    assert S.parse_json('Sure! {"a": [1, 2]} hope that helps') == {'a': [1, 2]}
    assert S.parse_json('no json here') is None


def test_choice_is_normalised_and_checks_options():
    e = ex('dominant_cover', Shape.WORD, 'tree cover', options=['tree cover', 'cropland'])
    assert S.score(e, 'Tree cover.').score == 1.0
    s = S.score(e, 'forest')
    assert s.score == 0.0 and not s.wellformed


def test_number_tolerance_is_per_family():
    e = ex('relief', Shape.NUMBER, '400')
    assert S.score(e, '430').correct                       # within 10%
    assert not S.score(e, '460').correct
    assert not S.score(e, 'about 400 m').wellformed        # parsed, but not bare
    assert not S.score(e, 'flat').parsed


def test_point_inside_reference_box():
    e = ex('locate_point', Shape.POINT, {'point_2d': [500, 500], 'label': 'x'},
           bbox_2d=[400, 400, 900, 600])
    assert S.score(e, '{"point_2d":[850,450],"label":"x"}').score == 1.0
    assert S.score(e, '{"point_2d":[500,700],"label":"x"}').score == 0.0
    assert not S.score(e, '{"point_2d":[500,1500]}').wellformed


def test_bbox_iou():
    e = ex('locate_bbox', Shape.BBOX, {'bbox_2d': [0, 0, 100, 100], 'label': 'x'})
    s = S.score(e, '{"bbox_2d":[0,0,100,50],"label":"x"}')
    assert abs(s.score - 0.5) < 1e-9 and s.details['hit']
    assert not S.score(e, '{"bbox_2d":[100,0,0,100]}').wellformed


def test_points_multi_is_order_free():
    t = {'points_2d': [[100, 100], [900, 900]], 'label': 'named features'}
    e = ex('features_all', Shape.POINTS_MULTI, t)
    assert S.score(e, '{"points_2d":[[905,890],[110,95]]}').score == 1.0
    assert S.score(e, '{"points_2d":[[100,100]]}').details['recall'] == 0.5


def test_grid_categorical_numeric_and_nulls():
    cat = ex('cover_grid', Shape.GRID, {'variable': 'land cover', 'shape': [2, 2],
                                        'grid': [[0, 1], [None, 3]]})
    assert S.score(cat, json.dumps({'variable': 'land cover', 'shape': [2, 2],
                                    'grid': [[0, 1], [None, 2]]})).score == 0.75
    assert not S.score(cat, '{"shape":[3,3],"grid":[[0]]}').wellformed
    num = ex('ndvi_grid', Shape.GRID, {'variable': 'ndvi_x100', 'shape': [1, 2],
                                       'grid': [[50, 60]]})
    s = S.score(num, '{"variable":"ndvi_x100","shape":[1,2],"grid":[[54,70]]}')
    assert s.score == 0.5 and s.details['mae'] == 7


def test_number_array_nulls_and_length():
    e = ex('ndvi_series', Shape.NUMBER_ARRAY, [10, None, 30])
    assert S.score(e, '[12, null, 29]').score == 1.0
    assert abs(S.score(e, '[12, 20, 29]').score - 2 / 3) < 1e-9
    assert not S.score(e, '[1, 2]').wellformed


def test_record_per_key():
    e = ex('acquisition', Shape.RECORD, {'date': '2020-05-01', 'solar_elevation_deg': 60})
    assert S.score(e, '{"date":"2020-05-01","solar_elevation_deg":61}').score == 1.0
    assert S.score(e, '{"date":"2020-05-02","solar_elevation_deg":61}').score == 0.5


def test_derived_scores_answer_and_reports_evidence():
    e = ex('vegetated_fraction', Shape.DERIVED,
           '{"ndvi_x100":[40,50,60],"vegetated_percent":70.0}')
    s = S.score(e, '{"ndvi_x100":[40,50,90],"vegetated_percent":75}')
    assert s.score == 1.0 and abs(s.details['evidence_score'] - 2 / 3) < 1e-9
    d = ex('disturbance', Shape.DERIVED, '{"changed_percent":[1.0,8.0],"date":"5 May 2020"}',
           options=['1 May 2020', '5 May 2020'])
    assert S.score(d, '{"changed_percent":[1.0,8.0],"date":"5 May 2020"}').correct
    assert S.score(d, '{"changed_percent":[1.0,8.0],"date":"1 May 2020"}').score == 0.0


def test_prose_token_f1():
    e = ex('caption', Shape.PROSE, 'a city on a river with forest to the north')
    assert S.score(e, 'a city on a river with forest to the north').score == 1.0
    assert 0 < S.score(e, 'a small city by a river').score < 1
