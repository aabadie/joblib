"""Test the numpy pickler as a replacement of the standard pickler."""

from tempfile import mkdtemp
import copy
import shutil
import os
import random
import sys
import re
import tempfile
import glob
import warnings
import nose

from joblib.test.common import np, with_numpy, with_memory_usage, memory_used

# numpy_pickle is not a drop-in replacement of pickle, as it takes
# filenames instead of open files as arguments.
from joblib import numpy_pickle
from joblib.test import data

###############################################################################
# Define a list of standard types.
# Borrowed from dill, initial author: Micheal McKerns:
# http://dev.danse.us/trac/pathos/browser/dill/dill_test2.py

typelist = []

# testing types
_none = None
typelist.append(_none)
_type = type
typelist.append(_type)
_bool = bool(1)
typelist.append(_bool)
_int = int(1)
typelist.append(_int)
try:
    _long = long(1)
    typelist.append(_long)
except NameError:
    # long is not defined in python 3
    pass
_float = float(1)
typelist.append(_float)
_complex = complex(1)
typelist.append(_complex)
_string = str(1)
typelist.append(_string)
try:
    _unicode = unicode(1)
    typelist.append(_unicode)
except NameError:
    # unicode is not defined in python 3
    pass
_tuple = ()
typelist.append(_tuple)
_list = []
typelist.append(_list)
_dict = {}
typelist.append(_dict)
try:
    _file = file
    typelist.append(_file)
except NameError:
    pass # file does not exists in Python 3
try:
    _buffer = buffer
    typelist.append(_buffer)
except NameError:
    # buffer does not exists in Python 3
    pass
_builtin = len
typelist.append(_builtin)


def _function(x):
    yield x


class _class:
    def _method(self):
        pass


class _newclass(object):
    def _method(self):
        pass


typelist.append(_function)
typelist.append(_class)
typelist.append(_newclass)  # <type 'type'>
_instance = _class()
typelist.append(_instance)
_object = _newclass()
typelist.append(_object)  # <type 'class'>


###############################################################################
# Test fixtures

env = dict()


def setup_module():
    """ Test setup.
    """
    env['dir'] = mkdtemp()
    env['filename'] = os.path.join(env['dir'], 'test.pkl')
    print(80 * '_')
    print('setup numpy_pickle')
    print(80 * '_')


def teardown_module():
    """ Test teardown.
    """
    shutil.rmtree(env['dir'])
    #del env['dir']
    #del env['filename']
    print(80 * '_')
    print('teardown numpy_pickle')
    print(80 * '_')


###############################################################################
# Tests

def test_standard_types():
    # Test pickling and saving with standard types.
    filename = env['filename']
    for compress in [0, 1]:
        for member in typelist:
            # Change the file name to avoid side effects between tests
            this_filename = filename + str(random.randint(0, 1000))
            numpy_pickle.dump(member, this_filename, compress=compress)
            _member = numpy_pickle.load(this_filename)
            # We compare the pickled instance to the reloaded one only if it
            # can be compared to a copied one
            if member == copy.deepcopy(member):
                yield nose.tools.assert_equal, member, _member


def test_value_error():
    # Test inverting the input arguments to dump
    nose.tools.assert_raises(ValueError, numpy_pickle.dump, 'foo',
                             dict())


@with_numpy
def test_numpy_persistence():
    filename = env['filename']
    rnd = np.random.RandomState(0)
    a = rnd.random_sample((10, 2))
    for compress in (False, True, 0, 3):
        # We use 'a.T' to have a non C-contiguous array.
        for index, obj in enumerate(((a,), (a.T,), (a, a), [a, a, a])):
            # Change the file name to avoid side effects between tests
            this_filename = filename + str(random.randint(0, 1000))

            # Check warning if cache size is set
            filenames = numpy_pickle.dump(obj, this_filename,
                                          compress=compress)

            # All is cached in one file
            nose.tools.assert_equal(len(filenames), 1)
            # Check that one file was created per array
            if not compress:
                nose.tools.assert_equal(filenames[0], this_filename)
            # Check that this file does exist
            nose.tools.assert_true(
                    os.path.exists(os.path.join(env['dir'], filenames[0])))

            # Unpickle the object
            obj_ = numpy_pickle.load(this_filename)
            # Check that the items are indeed arrays
            for item in obj_:
                nose.tools.assert_true(isinstance(item, np.ndarray))
            # And finally, check that all the values are equal.
            nose.tools.assert_true(np.all(np.array(obj) == np.array(obj_)))

        # Now test with array subclasses
        for obj in (
                    np.matrix(np.zeros(10)),
                    np.core.multiarray._reconstruct(np.memmap, (), np.float)
                   ):
            this_filename = filename + str(random.randint(0, 1000))
            filenames = numpy_pickle.dump(obj, this_filename,
                                          compress=compress)
            # All is cached in one file
            nose.tools.assert_equal(len(filenames), 1)

            obj_ = numpy_pickle.load(this_filename)
            if (type(obj) is not np.memmap and
                    hasattr(obj, '__array_prepare__')):
                # We don't reconstruct memmaps
                nose.tools.assert_true(isinstance(obj_, type(obj)))
                np.testing.assert_array_equal(obj_, obj)

        # Test with an object containing multiple numpy arrays
        obj = ComplexTestObject()
        filenames = numpy_pickle.dump(obj, this_filename,
                                      compress=compress)
        # All is cached in one file
        nose.tools.assert_equal(len(filenames), 1)

        obj_loaded = numpy_pickle.load(this_filename)
        nose.tools.assert_true(isinstance(obj_loaded, type(obj)))
        np.testing.assert_array_equal(obj_loaded.array_float, obj.array_float)
        np.testing.assert_array_equal(obj_loaded.array_int, obj.array_int)
        np.testing.assert_array_equal(obj_loaded.array_obj, obj.array_obj)


@with_numpy
def test_memmap_persistence():
    rnd = np.random.RandomState(0)
    a = rnd.random_sample(10)
    filename = env['filename'] + str(random.randint(0, 1000))
    numpy_pickle.dump(a, filename)
    b = numpy_pickle.load(filename, mmap_mode='r')

    nose.tools.assert_true(isinstance(b, np.memmap))

    # Test with an object containing multiple numpy arrays
    filename = env['filename'] + str(random.randint(0, 1000))
    obj = ComplexTestObject()
    numpy_pickle.dump(obj, filename)
    obj_loaded = numpy_pickle.load(filename, mmap_mode='r')
    nose.tools.assert_true(isinstance(obj_loaded, type(obj)))
    nose.tools.assert_true(isinstance(obj_loaded.array_float, np.memmap))
    nose.tools.assert_false(obj_loaded.array_float.flags.writeable)
    nose.tools.assert_true(isinstance(obj_loaded.array_int, np.memmap))
    nose.tools.assert_false(obj_loaded.array_int.flags.writeable)
    # Memory map not allowed for numpy object arrays
    nose.tools.assert_true(isinstance(obj_loaded.array_obj,
                                      type(obj.array_obj)))
    np.testing.assert_array_equal(obj_loaded.array_float,
                                  obj.array_float)
    np.testing.assert_array_equal(obj_loaded.array_int,
                                  obj.array_int)
    np.testing.assert_array_equal(obj_loaded.array_obj,
                                  obj.array_obj)

    # Test we can write in memmaped arrays
    obj_loaded = numpy_pickle.load(filename, mmap_mode='r+')
    nose.tools.assert_true(obj_loaded.array_float.flags.writeable)
    obj_loaded.array_float[0:10] = 10.0
    nose.tools.assert_true(obj_loaded.array_int.flags.writeable)
    obj_loaded.array_int[0:10] = 10

    obj_reloaded = numpy_pickle.load(filename, mmap_mode='r')
    np.testing.assert_array_equal(obj_reloaded.array_float,
                                  obj_loaded.array_float)
    np.testing.assert_array_equal(obj_reloaded.array_int,
                                  obj_loaded.array_int)

    # Test w+ mode is caught and the mode has switched to r+
    obj_mode_w = numpy_pickle.load(filename, mmap_mode='w+')
    nose.tools.assert_true(obj_loaded.array_int.flags.writeable)
    nose.tools.assert_equal(obj_loaded.array_int.mode, 'r+')
    nose.tools.assert_true(obj_loaded.array_float.flags.writeable)
    nose.tools.assert_equal(obj_loaded.array_float.mode, 'r+')


@with_numpy
def test_memmap_persistence_mixed_dtypes():
    # loading datastructures that have sub-arrays with dtype=object
    # should not prevent memmaping on fixed size dtype sub-arrays.
    rnd = np.random.RandomState(0)
    a = rnd.random_sample(10)
    b = np.array([1, 'b'], dtype=object)
    construct = (a, b)
    filename = env['filename'] + str(random.randint(0, 1000))
    numpy_pickle.dump(construct, filename)
    a_clone, b_clone = numpy_pickle.load(filename, mmap_mode='r')

    # the floating point array has been memory mapped
    nose.tools.assert_true(isinstance(a_clone, np.memmap))

    # the object-dtype array has been loaded in memory
    nose.tools.assert_false(isinstance(b_clone, np.memmap))


@with_numpy
def test_masked_array_persistence():
    # The special-case picker fails, because saving masked_array
    # not implemented, but it just delegates to the standard pickler.
    rnd = np.random.RandomState(0)
    a = rnd.random_sample(10)
    a = np.ma.masked_greater(a, 0.5)
    filename = env['filename'] + str(random.randint(0, 1000))
    numpy_pickle.dump(a, filename)
    b = numpy_pickle.load(filename, mmap_mode='r')
    nose.tools.assert_true(isinstance(b, np.ma.masked_array))


@with_numpy
def test_compress_mmap_mode_warning():
    # Test the warning in case of compress + mmap_mode
    rnd = np.random.RandomState(0)
    a = rnd.random_sample(10)
    this_filename = env['filename'] + str(random.randint(0, 1000))
    numpy_pickle.dump(a, this_filename, compress=1)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        numpy_pickle.load(this_filename, mmap_mode='r+')
        nose.tools.assert_equal(len(w), 1)
        for warn in w:
            nose.tools.assert_equal(warn.category, DeprecationWarning)
            nose.tools.assert_equal(warn.message.args[0],
                                    'File "%(filename)s" appears to be '
                                    'compressed, this is not compatible with '
                                    'mmap_mode "%(mmap_mode)s" flag passed' %
                                    {'filename': this_filename,
                                     'mmap_mode': 'r+'})


@with_numpy
def test_cache_size_warning():
    # Check deprecation warning raised when cache size is not None
    filename = env['filename'] + str(random.randint(0, 1000))
    rnd = np.random.RandomState(0)
    a = rnd.random_sample((10, 2))

    for cache_size in (None, 0, 10):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            filenames = numpy_pickle.dump(a, filename,
                                          cache_size=cache_size)
            nose.tools.assert_equal(len(w),
                                    1 if cache_size is not None else 0)
            for warn in w:
                nose.tools.assert_equal(warn.category, DeprecationWarning)
                nose.tools.assert_equal(warn.message.args[0],
                                        'Cache size is deprecated and '
                                        'will be ignored.')


@with_numpy
@with_memory_usage
def test_memory_usage():
    """Verify memory stays within expected bounds."""
    filename = env['filename']
    small_array = np.ones((10, 10))
    big_array = np.ones((10000, 10000))
    big_matrix = np.matrix([i for i in range(1000000)])
    print("")
    for compress in (True, False):
        for obj in (small_array, big_array, big_matrix):
            size = obj.nbytes / 1e6
            obj_filename = filename + str(np.random.randint(0, 1000))
            mem_used = memory_used(numpy_pickle.dump,
                                   obj, obj_filename, compress=compress)

            # The memory used to dump the object shouldn't exceed the buffer
            # size used to write array chunks (16MB).
            write_buf_size = 16 * 1024 ** 2 / 1e6
            nose.tools.assert_true(mem_used <= write_buf_size)

            mem_used = memory_used(numpy_pickle.load, obj_filename)
            # memory used should be less than array size + buffer size used to
            # read the array chunk by chunk.
            read_buf_size = 32  # MiB
            nose.tools.assert_true(mem_used < size + read_buf_size)


@with_numpy
def test_compressed_pickle_dump_and_load():
    # XXX: temporarily disable this test on non little-endian machines
    if sys.byteorder != 'little':
        raise nose.SkipTest('Skipping this test on non little-endian machines')

    expected_list = [np.arange(5, dtype=np.dtype('<i8')),
                     np.arange(5, dtype=np.dtype('<f8')),
                     np.array([1, 'abc', {'a': 1, 'b': 2}], dtype='O'),
                     # .tostring actually returns bytes and is a
                     # compatibility alias for .tobytes which was
                     # added in 1.9.0
                     np.arange(256, dtype=np.uint8).tostring(),
                     # np.matrix is a subclass of nd.array, here we want
                     # to verify this type of object is correctly unpickled
                     # among versions.
                     np.matrix([0, 1, 2]),
                     u"C'est l'\xe9t\xe9 !"]

    with tempfile.NamedTemporaryFile(suffix='.gz', dir=env['dir']) as f:
        fname = f.name

    # Need to test both code branches (whether array size is greater
    # or smaller than cache_size)
    try:
        dumped_filenames = numpy_pickle.dump(expected_list, fname, compress=1)
        result_list = numpy_pickle.load(fname)
        for result, expected in zip(result_list, expected_list):
            if isinstance(expected, np.ndarray):
                nose.tools.assert_equal(result.dtype, expected.dtype)
                np.testing.assert_equal(result, expected)
            else:
                nose.tools.assert_equal(result, expected)
    finally:
        os.remove(fname)


def _check_pickle(filename, expected_list):
    """Helper function to test joblib pickle content.

    Note: currently only pickles containing an iterable are supported
    by this function.
    """
    version_match = re.match(r'.+py(\d)(\d).+', filename)
    py_version_used_for_writing = int(version_match.group(1))
    py_version_used_for_reading = sys.version_info[0]

    py_version_to_default_pickle_protocol = {2: 2, 3: 3}
    pickle_reading_protocol = py_version_to_default_pickle_protocol.get(
        py_version_used_for_reading, 4)
    pickle_writing_protocol = py_version_to_default_pickle_protocol.get(
        py_version_used_for_writing, 4)
    if pickle_reading_protocol >= pickle_writing_protocol:
        try:
            with warnings.catch_warnings(record=True) as catched_warnings:
                warnings.simplefilter("always")
                result_list = numpy_pickle.load(filename)
                nose.tools.assert_equal(len(catched_warnings),
                                        1 if ("0.9" in filename or
                                              "0.8.4" in filename) else 0)
            for warn in catched_warnings:
                nose.tools.assert_equal(warn.category, DeprecationWarning)
                nose.tools.assert_equal(warn.message.args[0],
                                        "The file '%s' has been generated with "
                                        "a joblib version less than 0.10. "
                                        "Please regenerate this pickle file." %
                                        filename)
            for result, expected in zip(result_list, expected_list):
                if isinstance(expected, np.ndarray):
                    nose.tools.assert_equal(result.dtype, expected.dtype)
                    np.testing.assert_equal(result, expected)
                else:
                    nose.tools.assert_equal(result, expected)
        except Exception as exc:
            # When trying to read with python 3 a pickle generated
            # with python 2 we expect a user-friendly error
            if (py_version_used_for_reading == 3 and
                    py_version_used_for_writing == 2):
                nose.tools.assert_true(isinstance(exc, ValueError))
                message = ('You may be trying to read with '
                           'python 3 a joblib pickle generated with python 2.')
                nose.tools.assert_true(message in str(exc))
            else:
                raise
    else:
        # Pickle protocol used for writing is too high. We expect a
        # "unsupported pickle protocol" error message
        try:
            numpy_pickle.load(filename)
            raise AssertionError('Numpy pickle loading should '
                                 'have raised a ValueError exception')
        except ValueError as e:
            message = 'unsupported pickle protocol: {0}'.format(
                pickle_writing_protocol)
            nose.tools.assert_true(message in str(e.args))


@with_numpy
def test_joblib_pickle_across_python_versions():
    # XXX: temporarily disable this test on non little-endian machines
    if sys.byteorder != 'little':
        raise nose.SkipTest('Skipping this test on non little-endian machines')

    # We need to be specific about dtypes in particular endianness
    # because the pickles can be generated on one architecture and
    # the tests run on another one. See
    # https://github.com/joblib/joblib/issues/279.
    expected_list = [np.arange(5, dtype=np.dtype('<i8')),
                     np.arange(5, dtype=np.dtype('<f8')),
                     np.array([1, 'abc', {'a': 1, 'b': 2}], dtype='O'),
                     # .tostring actually returns bytes and is a
                     # compatibility alias for .tobytes which was
                     # added in 1.9.0
                     np.arange(256, dtype=np.uint8).tostring(),
                     # np.matrix is a subclass of nd.array, here we want
                     # to verify this type of object is correctly unpickled
                     # among versions.
                     np.matrix([0, 1, 2]),
                     u"C'est l'\xe9t\xe9 !"]

    # Testing all the *.gz and *.pkl (compressed and non compressed
    # pickles) in joblib/test/data. These pickles were generated by
    # the joblib/test/data/create_numpy_pickle.py script for the
    # relevant python, joblib and numpy versions.
    test_data_dir = os.path.dirname(os.path.abspath(data.__file__))
    data_filenames = glob.glob(os.path.join(test_data_dir, 'joblib_*.gz'))
    data_filenames += glob.glob(os.path.join(test_data_dir, 'joblib_*.pkl'))

    for fname in data_filenames:
        _check_pickle(fname, expected_list)

################################################################################
# Test dumping array subclasses
if np is not None:

    class SubArray(np.ndarray):

        def __reduce__(self):
            return _load_sub_array, (np.asarray(self), )

    def _load_sub_array(arr):
        d = SubArray(arr.shape)
        d[:] = arr
        return d

    class ComplexTestObject:
        """A complex object containing numpy arrays as attributes."""

        def __init__(self):
            self.array_float = np.arange(100, dtype='float64')
            self.array_int = np.ones(100, dtype='int32')
            self.array_obj = np.array(['a', 10, 20.0], dtype='object')


@with_numpy
def test_numpy_subclass():
    filename = env['filename']
    a = SubArray((10,))
    numpy_pickle.dump(a, filename)
    c = numpy_pickle.load(filename)
    nose.tools.assert_true(isinstance(c, SubArray))
    np.testing.assert_array_equal(c, a)
