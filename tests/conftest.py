"""Tests that need ELLIOT data read it from ELLIOT_ROOT (the merged release) or from
ELLIOT_ROOT and ELLIOT_X_EXT_ROOT (the two datasets), and are skipped without it."""
import os
from pathlib import Path

import pytest

ROOTS = [r for r in (os.environ.get('ELLIOT_ROOT'), os.environ.get('ELLIOT_X_EXT_ROOT')) if r]
HAVE_DATA = bool(ROOTS) and all(r.startswith(('hf://', 'http')) or Path(r).is_dir() for r in ROOTS)

needs_data = pytest.mark.skipif(not HAVE_DATA, reason='set ELLIOT_ROOT (and ELLIOT_X_EXT_ROOT for the two datasets)')

#: Tiles the data tests use: a city with every modality, a seasonal climatology
#: with two growing seasons' worth of frames, and a burst with an abrupt change.
CITY = ('monotemporal', '284D_496L')
MONTHLY = ('monthly', 352)
BURST = ('burst', 5)


@pytest.fixture(scope='session')
def et():
    import elliot_tasks
    elliot_tasks.configure()
    return elliot_tasks


@pytest.fixture(scope='session')
def city(et):
    return et.find(CITY[1], CITY[0])


@pytest.fixture(scope='session')
def monthly(et):
    return et.tile(*MONTHLY)


@pytest.fixture(scope='session')
def burst(et):
    return et.tile(*BURST)
