"""Test the old numpy pickler, compatibility version."""

import shutil
import os
import random
import sys
import tempfile
import glob
import nose

from tempfile import mkdtemp
from nose import SkipTest
from joblib.test.common import np, with_numpy

# numpy_pickle is not a drop-in replacement of pickle, as it takes
# filenames instead of open files as arguments.
from joblib import numpy_pickle, numpy_pickle_compat, numpy_pickle_utils
from joblib.test import data


###############################################################################
# Test fixtures

env = dict()


def setup_module():
    """Test setup."""
    env['dir'] = mkdtemp()
    env['filename'] = os.path.join(env['dir'], 'test.pkl')
    print(80 * '_')
    print('setup numpy_pickle')
    print(80 * '_')


def teardown_module():
    """Test teardown."""
    shutil.rmtree(env['dir'])
    print(80 * '_')
    print('teardown numpy_pickle')
    print(80 * '_')


def test_z_file():
    """Test saving and loading data with Zfiles."""
    filename = env['filename'] + str(random.randint(0, 1000))
    data = numpy_pickle_utils.asbytes('Foo, \n Bar, baz, \n\nfoobar')
    with open(filename, 'wb') as f:
        numpy_pickle_compat.write_zfile(f, data)
    with open(filename, 'rb') as f:
        data_read = numpy_pickle_compat.read_zfile(f)
    nose.tools.assert_equal(data, data_read)


@with_numpy
def test_load_compatibility_function():
    """Test load compatibility function."""
    obj = [np.ones((10, 10)), np.matrix([0, 1, 2])]

    test_data_dir = os.path.dirname(os.path.abspath(data.__file__))
    data_filenames = glob.glob(os.path.join(test_data_dir, 'test_*.gz'))
    data_filenames += glob.glob(os.path.join(test_data_dir, 'test_*.pkl'))

    for filename in data_filenames:
        try:
            obj_read = numpy_pickle.load(filename)
        except ValueError:
            raise SkipTest("Skipped as this version of python (%s), doesn't "
                           "support the required pickle format.")
        else:
            nose.tools.assert_equal(len(obj_read), 2)

            np.testing.assert_array_equal(obj[0], obj_read[0])
            np.testing.assert_array_equal(obj[1], obj_read[1])
