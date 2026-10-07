"""Tests that need ELLIOT data read it from ELLIOT_ROOT and ELLIOT_X_EXT_ROOT and
are skipped when either is unset or missing."""
import os
from pathlib import Path

import pytest

ROOTS = (os.environ.get('ELLIOT_ROOT'), os.environ.get('ELLIOT_X_EXT_ROOT'))
HAVE_DATA = all(r and (r.startswith('hf://') or Path(r).is_dir()) for r in ROOTS)

needs_data = pytest.mark.skipif(not HAVE_DATA, reason='set ELLIOT_ROOT and ELLIOT_X_EXT_ROOT')

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
