"""Utilities for fast persistence of big data, with optional compression."""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# Copyright (c) 2009 Gael Varoquaux
# License: BSD Style, 3 clauses.

import pickle
import os
import sys
import gzip
import warnings
import io
from ctypes import c_int64
from contextlib import closing

from .numpy_pickle_utils import PY3, _ZFILE_PREFIX, _MEGA
from .numpy_pickle_utils import Unpickler, Pickler
from .numpy_pickle_compat import load_compatibility, NDArrayWrapper
from ._compat import _basestring

try:
    import numpy as np
    from numpy.compat import isfileobj, asstr, asbytes

    # Those globals depends on numpy import
    MAGIC_PREFIX = asbytes('\x93NUMPY')
    MAGIC_LEN = len(MAGIC_PREFIX) + 2
    BUFFER_SIZE = 2**18  # size of buffer for reading npz files in bytes
except ImportError:
    np = None

_GZIP_PREFIX = b'\x1f\x8b'


###############################################################################
# Cache file utilities

def _read_magic(file_handle):
    """Utility to check the magic signature of a file.

    _ZFILE_PREFIX is used to determine the number of bytes to read to determine
    the magic number.

    Parameters
    ----------
    file_handle: file_like

    """
    magic = file_handle.read(len(_ZFILE_PREFIX))
    # Pickling needs file-handles at the beginning of the file
    file_handle.seek(0)
    return magic


def _check_filetype(filename, magic):
    """Utility function opening the right fileobject from a filename.

    The magic number is used to choose between the type of file object to open:
    * regular file object (default)
    * gzip file object is magic matches _GZIP_PREFIX

    Parameters
    ----------
    filename: str
        The full path string of the file
    magic: bytes
        The bytes representation of the magic number of the file

    Returns
    -------
        a file like object

    """
    if magic == _GZIP_PREFIX:
        return gzip.GzipFile(filename, 'rb')

    return open(filename, 'rb')


# difference between version 1.0 and 2.0 is a 4 byte (I) header length
# instead of 2 bytes (H) allowing storage of large structured arrays

def _check_version(version):
    """Utility that check if the version is supported.

    An exception is raised if version is not supported.

    Taken from numpy 1.10.

    Parameters
    ----------
    version: tuple
        2 element tuple containing the major and minor versions.

    Raises
    ------
    ValueError
        If the version is invalid.

    """
    if version not in [(1, 0), (2, 0), None]:
        msg = "we only support format version (1,0) and (2, 0), not %s"
        raise ValueError(msg % (version,))


def _read_bytes(fp, size, error_template="ran out of data"):
    """Read from file-like object until size bytes are read.

    Raises ValueError if not EOF is encountered before size bytes are read.
    Non-blocking objects only supported if they derive from io objects.

    Required as e.g. ZipExtFile in python 2.6 can return less data than
    requested.

    Taken from numpy 1.10.

    Parameters
    ----------
    fp: file-like object
    size: int
    error_template: str

    Returns
    -------
    a bytes object
        The data read in bytes.

    """
    data = bytes()
    while True:
        # io files (default in python3) return None or raise on
        # would-block, python2 file will truncate, probably nothing can be
        # done about that.  note that regular files can't be non-blocking
        try:
            r = fp.read(size - len(data))
            data += r
            if len(r) == 0 or len(data) == size:
                break
        except io.BlockingIOError:
            pass
    if len(data) != size:
        msg = "EOF: reading %s, expected %d bytes got %d"
        raise ValueError(msg % (error_template, size, len(data)))
    else:
        return data


def _read_numpy_magic(fp):
    """Read the magic string to get the version of the file format.

    Taken from numpy 1.10.

    Parameters
    ----------
    fp : file_like

    Returns
    -------
    major : int
    minor : int

    """
    magic_str = _read_bytes(fp, MAGIC_LEN, "magic string")
    if magic_str[:-2] != MAGIC_PREFIX:
        msg = "the magic string is not correct; expected %r, got %r"
        raise ValueError(msg % (MAGIC_PREFIX, magic_str[:-2]))
    if sys.version_info[0] < 3:
        major, minor = map(ord, magic_str[-2:])
    else:
        major, minor = magic_str[-2:]
    return major, minor


def _filter_header(s):
    """Clean up 'L' in npz header ints.

    Cleans up the 'L' in strings representing integers. Needed to allow npz
    headers produced in Python2 to be read in Python3.

    Taken from numpy 1.10.

    Parameters
    ----------
    s : byte string
        Npy file header.

    Returns
    -------
    header : str
        Cleaned up header.

    """
    import tokenize
    if sys.version_info[0] >= 3:
        from io import StringIO
    else:
        from StringIO import StringIO

    tokens = []
    last_token_was_number = False
    for token in tokenize.generate_tokens(StringIO(asstr(s)).read):
        token_type = token[0]
        token_string = token[1]
        if (last_token_was_number and
                token_type == tokenize.NAME and
                token_string == "L"):
            continue
        else:
            tokens.append(token)
        last_token_was_number = (token_type == tokenize.NUMBER)
    return tokenize.untokenize(tokens)


def _read_array_header(fp, version):
    """Read array header in file.

    Taken from numpy 1.10.

    """
    # Read an unsigned, little-endian short int which has the length of the
    # header.
    import struct
    if version == (1, 0):
        hlength_str = _read_bytes(fp, 2, "array header length")
        header_length = struct.unpack('<H', hlength_str)[0]
        header = _read_bytes(fp, header_length, "array header")
    elif version == (2, 0):
        hlength_str = _read_bytes(fp, 4, "array header length")
        header_length = struct.unpack('<I', hlength_str)[0]
        header = _read_bytes(fp, header_length, "array header")
    else:
        raise ValueError("Invalid version %r" % version)

    # The header is a pretty-printed string representation of a literal
    # Python dictionary with trailing newlines padded to a 16-byte
    # boundary. The keys are strings.
    #   "shape" : tuple of int
    #   "fortran_order" : bool
    #   "descr" : dtype.descr
    header = _filter_header(header)
    try:
        d = np.safe_eval(header)
    except SyntaxError as e:
        msg = "Cannot parse header: %r\nException: %r"
        raise ValueError(msg % (header, e))
    if not isinstance(d, dict):
        msg = "Header is not a dictionary: %r"
        raise ValueError(msg % d)
    keys = sorted(d.keys())
    if keys != ['descr', 'fortran_order', 'shape']:
        msg = "Header does not contain the correct keys: %r"
        raise ValueError(msg % (keys,))

    # Sanity-check the values.
    if (not isinstance(d['shape'], tuple) or
            not np.all([isinstance(x, (int, np.long)) for x in d['shape']])):
        msg = "shape is not valid: %r"
        raise ValueError(msg % (d['shape'],))
    if not isinstance(d['fortran_order'], bool):
        msg = "fortran_order is not a valid bool: %r"
        raise ValueError(msg % (d['fortran_order'],))
    try:
        dtype = np.dtype(d['descr'])
    except TypeError as e:
        msg = "descr is not a valid dtype descriptor: %r"
        raise ValueError(msg % (d['descr'],))

    return d['shape'], d['fortran_order'], dtype


def open_memmap(filename, array_offset=0, mode='r+', dtype=None, shape=None,
                fortran_order=False, version=None):
    """
    Open a .npy file as a memory-mapped array.

    This may be used to read an existing file or create a new one.

    Taken from numpy 1.10

    Parameters
    ----------
    filename : str
        The name of the file on disk.  This may *not* be a file-like
        object.
    array_offset : int
        The offset in the file object where the array is serialized
    mode : str, optional
        The mode in which to open the file; the default is 'r+'.  In
        addition to the standard file modes, 'c' is also accepted to mean
        "copy on write."  See `memmap` for the available mode strings.
    dtype : data-type, optional
        The data type of the array if we are creating a new file in "write"
        mode, if not, `dtype` is ignored.  The default value is None, which
        results in a data-type of `float64`.
    shape : tuple of int
        The shape of the array if we are creating a new file in "write"
        mode, in which case this parameter is required.  Otherwise, this
        parameter is ignored and is thus optional.
    fortran_order : bool, optional
        Whether the array should be Fortran-contiguous (True) or
        C-contiguous (False, the default) if we are creating a new file in
        "write" mode.
    version : tuple of int (major, minor) or None
        If the mode is a "write" mode, then this is the version of the file
        format used to create the file.  None means use the oldest
        supported version that is able to store the data.  Default: None

    Returns
    -------
    marray : memmap
        The memory-mapped array.

    Raises
    ------
    ValueError
        If the data or the mode is invalid.
    IOError
        If the file is not found or cannot be opened correctly.

    See Also
    --------
    memmap

    """
    # Read the header of the array first.
    try:
        fp = open(filename, mode+'b')
        fp.seek(array_offset)
        version = _read_numpy_magic(fp)
        _check_version(version)

        shape, fortran_order, dtype = _read_array_header(fp, version)
        if dtype.hasobject:
            msg = "Array can't be memory-mapped: Python objects in dtype."
            raise ValueError(msg)
        offset = fp.tell()
    finally:
        fp.close()

    if fortran_order:
        order = 'F'
    else:
        order = 'C'

    # We need to change a write-only mode to a read-write mode since we've
    # already written data to the file.
    if mode == 'w+':
        mode = 'r+'

    marray = np.memmap(filename, dtype=dtype, shape=shape, order=order,
                       mode=mode, offset=offset)

    return marray


def _read_array(fp):
    """Read an array from an NPY file.

    Taken from numpy 1.10.

    Parameters
    ----------
    fp : file_like object
        If this is not a real file object, then this may take extra memory
        and time.

    Returns
    -------
    array : ndarray
        The array from the data on disk.

    Raises
    ------
    ValueError
        If the data is invalid.

    """
    version = _read_numpy_magic(fp)
    _check_version(version)
    shape, fortran_order, dtype = _read_array_header(fp, version)
    if len(shape) == 0:
        count = 1
    else:
        count = np.multiply.reduce(shape)

    # Now read the actual data.
    if dtype.hasobject:
        # The array contained Python objects. We need to unpickle the data.
        array = pickle.load(fp)
    else:
        if isfileobj(fp):
            # We can use the fast fromfile() function.
            array = np.fromfile(fp, dtype=dtype, count=count)
        else:
            # This is not a real file. We have to read it the
            # memory-intensive way.
            # crc32 module fails on reads greater than 2 ** 32 bytes,
            # breaking large reads from gzip streams. Chunk reads to
            # BUFFER_SIZE bytes to avoid issue and reduce memory overhead
            # of the read. In non-chunked case count < max_read_count, so
            # only one read is performed.

            max_read_count = BUFFER_SIZE // min(BUFFER_SIZE, dtype.itemsize)

            array = np.empty(count, dtype=dtype)
            for i in range(0, count, max_read_count):
                read_count = min(max_read_count, count - i)
                read_size = int(read_count * dtype.itemsize)
                data = _read_bytes(fp, read_size, "array data")
                array[i:i+read_count] = np.frombuffer(data, dtype=dtype,
                                                      count=read_count)

        if fortran_order:
            array.shape = shape[::-1]
            array = array.transpose()
        else:
            array.shape = shape

    return array
###############################################################################
# Utility objects for persistence.


class NPArrayWrapper(object):
    """An object to be persisted instead of numpy arrays.

    The only thing this object does, is to carry the filename in which
    the array has been persisted, and the array subclass.

    Attributes
    ----------
    subclass: numpy.ndarray subclass
        Determine the subclass of the wrapped array.
    allow_mmap: bool
        Determine if memory mapping is allowed on the wrapped array.
    offset: int
        Contains the offset in file where the wrapped array can be read.
    """

    def __init__(self, subclass, allow_mmap=True, offset=c_int64(-1)):
        """Constructor. Store the useful information for later."""
        self.subclass = subclass
        self.allow_mmap = allow_mmap
        self.offset = offset

    def read(self, unpickler):
        """Read the array corresponging to this wrapper.

        Use the unpickler to get all information to correctly read the array.

        Parameters
        ----------
        unpickler: NumpyUnpickler

        Returns
        -------
        array: numpy.ndarray or numpy.memmap or numpy.matrix

        """
        # The first pickled array should have an offset set
        if self.offset.value != -1:
            unpickler.current_offset = self.offset.value

        if unpickler.current_offset == -1:
            raise ValueError("Invalid numpy unpickling offset")

        # Get the current position in the file
        offset = unpickler.file_handle.tell()

        # Move to the offset position => beginning of array to read
        unpickler.file_handle.seek(unpickler.current_offset)

        # Now we read the array stored at current offset
        # position in file handle
        if self.allow_mmap and unpickler.mmap_mode is not None:
            array = open_memmap(unpickler.filename,
                                unpickler.current_offset,
                                mode=unpickler.mmap_mode)
        else:
            array = _read_array(unpickler.file_handle)

        # Next offset position is at the end of the array we just read and
        # before the next array if there's one
        unpickler.current_offset = unpickler.file_handle.tell()

        # Go back to unpickler position in file
        unpickler.file_handle.seek(offset)

        # Manage array subclass case
        if (hasattr(array, '__array_prepare__') and
            self.subclass not in (unpickler.np.ndarray,
                                  unpickler.np.memmap)):
            # We need to reconstruct another subclass
            new_array = unpickler.np.core.multiarray._reconstruct(
                                    self.subclass, (0,), 'b')
            new_array.__array_prepare__(array)
            array = new_array

        return array

###############################################################################
# Pickler classes


class NumpyPickler(Pickler):
    """A pickler to persist of big data efficiently.

    The main features of this object are:
    * persistence of numpy arrays in separate .npy files, for which
    I/O is fast.

    * optional compression using Zlib, with a special care on avoid
    temporaries.

    Attributes
    ----------
    fp: file
        File object handle used for serializing the input object.
    cache_size: int
        The maximum object size until the default pickle is used.
    protocol: int
        Pickle protocol used. Default is pickle.DEFAULT_PROTOCOL under
        python 3, pickle.HIGHEST_PROTOCOL otherwise.
    offset: int
        Position from where it is safe to write in the target file after
        pickle serialization completes. This is the position where the
        first numpy array starts to be written in the file.
    """

    dispatch = Pickler.dispatch.copy()

    def __init__(self, fp, cache_size=10, protocol=None, offset=c_int64(-1)):
        """Constructor. Store the useful information for later."""
        self.file = fp
        self.cache_size = cache_size
        self.compress = isinstance(self.file, gzip.GzipFile)

        # store temporarily the arrays
        self.arrays = []

        # By default we want a pickle protocol that only changes with
        # the major python version and not the minor one
        if protocol is None:
            protocol = (pickle.DEFAULT_PROTOCOL if PY3
                        else pickle.HIGHEST_PROTOCOL)

        Pickler.__init__(self, self.file, protocol=protocol)
        # delayed import of numpy, to avoid tight coupling
        try:
            import numpy as np
        except ImportError:
            np = None
        self.np = np
        self.file_offset = offset

    def _create_array_wrapper(self, array):
        """Create and returns a numpy array wrapper from a numpy array.

        Parameters
        ----------
        array: numpy.ndarray

        Returns
        -------
        wrapper: NPArrayWrapper:
            The numpy array wrapper.
        """
        allow_mmap = not array.dtype.hasobject and not self.compress
        offset = c_int64(-1) if len(self.arrays) != 1 else self.file_offset
        wrapper = NPArrayWrapper(type(array),
                                 allow_mmap=allow_mmap,
                                 offset=offset)

        return wrapper

    def save(self, obj):
        """Subclass the Pickler `save` method.

        To save ndarray subclasses in npy
        files, rather than pickling them. Of course, this is a
        total abuse of the Pickler class.

        """
        if self.np is not None and type(obj) in (self.np.ndarray,
                                                 self.np.matrix,
                                                 self.np.memmap):
            size = obj.size * obj.itemsize
            if self.compress and size < self.cache_size * _MEGA:
                # When compressing, as we are not writing directly to the
                # disk, it is more efficient to use standard pickling
                if type(obj) is self.np.memmap:
                    # Pickling doesn't work with memmapped arrays
                    obj = self.np.asarray(obj)
                return Pickler.save(self, obj)

            # We store a ref to arrays during the dump, those arrays
            # will be written at the end of the pickled file
            self.arrays.append(obj)

            # This converts on the fly the array in a wrapper
            obj = self._create_array_wrapper(obj)

        return Pickler.save(self, obj)


class NumpyUnpickler(Unpickler):
    """A subclass of the Unpickler to unpickle our numpy pickles.

    Attributes
    ----------
    mmap_mode: str
        The memorymap mode to use for reading numpy arrays.
    file_handle: file_like
        File object to unpickle from.
    filename: str
        Name of the file to unpickle from. It should correspond to file_handle.
    np: module
        Contain pointer to imported numpy module (if available).
    current_offset: c_int64
        Offset in file of the next array to read.

    """

    dispatch = Unpickler.dispatch.copy()

    def __init__(self, filename, file_handle, mmap_mode=None):
        """Constructor."""
        # The 2 next lines are for backward compatibility
        self._filename = os.path.basename(filename)
        self._dirname = os.path.dirname(filename)

        self.mmap_mode = mmap_mode
        self.file_handle = file_handle
        self.filename = filename
        Unpickler.__init__(self, self.file_handle)
        try:
            import numpy as np
        except ImportError:
            np = None
        self.np = np
        self.current_offset = c_int64(-1)

    def load_build(self):
        """Called to set the state of a newly created object.

        We capture it to replace our place-holder objects,
        NDArrayWrapper, by the array we are interested in. We
        replace them directly in the stack of pickler.
        """
        Unpickler.load_build(self)

        # For back backward compatibility
        if isinstance(self.stack[-1], (NDArrayWrapper, NPArrayWrapper)):
            if self.np is None:
                raise ImportError("Trying to unpickle an ndarray, "
                                  "but numpy didn't import correctly")
            nd_array_wrapper = self.stack.pop()
            array = nd_array_wrapper.read(self)
            self.stack.append(array)

    # Be careful to register our new method.
    if PY3:
        dispatch[pickle.BUILD[0]] = load_build
    else:
        dispatch[pickle.BUILD] = load_build


###############################################################################
# Utility functions

def dump(value, filename, compress=0, cache_size=100, protocol=None):
    """Persist an arbitrary Python object.

    Fast persistence of an arbitrary Python object into one file with
    dedicated storage for numpy arrays.

    Parameters
    -----------
    value: any Python object
        The object to store to disk
    filename: string
        The name of the file in which it is to be stored
    compress: integer for 0 to 9, optional
        Optional compression level for the data. 0 is no compression.
        Higher means more compression, but also slower read and
        write times. Using a value of 3 is often a good compromise.
        See the notes for more details.
    cache_size: positive number, optional
        Fixes the order of magnitude (in megabytes) of the cache used
        for in-memory compression. Note that this is just an order of
        magnitude estimate and that for big arrays, the code will go
        over this value at dump and at load time.
    protocol: positive int
        Pickle protocol, see pickle.dump documentation for more details.

    Returns
    -------
    filenames: list of strings
        The list of file names in which the data is stored. If
        compress is false, each array is stored in a different file.

    See Also
    --------
    joblib.load : corresponding loader

    Notes
    -----
    Memmapping on load cannot be used for compressed files. Thus
    using compression can significantly slow down loading. In
    addition, compressed files take extra extra memory during
    dump and load.

    """
    if not isinstance(filename, _basestring):
        # People keep inverting arguments, and the resulting error is
        # incomprehensible
        raise ValueError(
            'Second argument should be a filename, %s (type %s) was given'
            % (filename, type(filename))
        )

    try:
        if compress > 0:
            fp = gzip.GzipFile(filename, 'wb', compresslevel=compress)
        else:
            fp = open(filename, 'wb')
        pickler = NumpyPickler(fp,
                               cache_size=cache_size,
                               protocol=protocol)
        pickler.dump(value)
        # Arrays were found in the pickled object, we replay the dump
        # in order to set the offset in the first pickled array wrapper
        if len(pickler.arrays) > 0:
            fp.flush()
            offset = c_int64(fp.tell())
            fp.close()
            # Now that we now the offset needed to write after the pickle
            # we process another pickle
            if compress > 0:
                fp = gzip.GzipFile(filename, 'wb', compresslevel=compress)
            else:
                fp = open(filename, 'wb')
            pickler = NumpyPickler(fp,
                                   cache_size=cache_size,
                                   protocol=protocol,
                                   offset=offset)
            pickler.dump(value)
    finally:
        if 'pickler' in locals() and hasattr(pickler, 'file'):
            fp.flush()
            for array in pickler.arrays:
                pickler.np.save(fp, array)

            fp.close()
    return [filename]


def load(filename, mmap_mode=None):
    """Reconstruct a Python object from a file persisted with joblib.dump.

    Parameters
    -----------
    filename: string
        The name of the file from which to load the object
    mmap_mode: {None, 'r+', 'r', 'w+', 'c'}, optional
        If not None, the arrays are memory-mapped from the disk. This
        mode has no effect for compressed files. Note that in this
        case the reconstructed object might not longer match exactly
        the originally pickled object.

    Returns
    -------
    result: any Python object
        The object stored in the file.

    See Also
    --------
    joblib.dump : function to save an object

    Notes
    -----

    This function can load numpy array files saved separately during the
    dump. If the mmap_mode argument is given, it is passed to np.load and
    arrays are loaded as memmaps. As a consequence, the reconstructed
    object might not match the original pickled object. Note that if the
    file was saved with compression, the arrays cannot be memmaped.
    """
    # Determine
    magic = b''
    with open(filename, 'rb') as file_handle:
        magic = _read_magic(file_handle)

    # Backward compatibility with old compression strategy
    if magic == _ZFILE_PREFIX:
        return load_compatibility(filename)

    with closing(_check_filetype(filename, magic)) as file_handle:
        if isinstance(file_handle, gzip.GzipFile) and mmap_mode is not None:
            warnings.warn('file "%(filename)s" appears to be a zip, '
                          'ignoring mmap_mode "%(mmap_mode)s" flag passed'
                          % locals(), Warning, stacklevel=2)

        # We are careful to open the file handle early and keep it open to
        # avoid race-conditions on renames. That said, if data are stored in
        # companion files, moving the directory will create a race when
        # joblib tries to access the companion files.
        unpickler = NumpyUnpickler(filename,
                                   file_handle,
                                   mmap_mode=mmap_mode)
        try:
            obj = unpickler.load()
        except UnicodeDecodeError as exc:
            # More user-friendly error message
            if PY3:
                new_exc = ValueError(
                    'You may be trying to read with '
                    'python 3 a joblib pickle generated with python 2. '
                    'This feature is not supported by joblib.')
                new_exc.__cause__ = exc
                raise new_exc
        finally:
            if hasattr(unpickler, 'file_handle'):
                unpickler.file_handle.close()
        return obj
