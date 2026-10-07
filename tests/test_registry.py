"""The task registry and the shape contract; no data needed."""
import pytest

from elliot_tasks import tasks as T
from elliot_tasks.shapes import SHAPES, Feasibility, Shape, derived_spec


def test_sixteen_families_each_with_prompts():
    assert len(T.FAMILIES) == 16
    for name, fam in T.FAMILIES.items():
        assert len(T.PROMPTS[name]) == 10
        assert fam.shape is Shape.DERIVED or fam.shape in SHAPES


@pytest.mark.parametrize('family', sorted(T.PROMPTS))
def test_template_split_is_disjoint_and_complete(family):
    train, ev = T.template_indices(family, 'train'), T.template_indices(family, 'eval')
    assert not set(train) & set(ev)
    assert sorted(train + ev) == list(T.template_indices(family, 'all'))


def test_derived_promotes_a_bare_number():
    assert derived_spec(Shape.NUMBER_ARRAY, Shape.NUMBER).feasibility is Feasibility.HIGH
    assert derived_spec(Shape.GRID, Shape.NUMBER).feasibility is Feasibility.LOW


def test_every_shape_used_has_a_format_instruction():
    for fam in T.FAMILIES.values():
        if fam.shape is not Shape.DERIVED:
            assert fam.shape in T.FORMAT_INSTRUCTIONS
