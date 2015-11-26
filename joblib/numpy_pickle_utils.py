"""Utilities for fast persistence of big data, with optional compression."""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# Copyright (c) 2009 Gael Varoquaux
# License: BSD Style, 3 clauses.

import pickle
import traceback
import sys
import os
import warnings

from ._compat import _basestring


PY3 = sys.version_info[0] >= 3
PY26 = sys.version_info[0] == 2 and sys.version_info[1] == 6

if PY3:
    Unpickler = pickle._Unpickler
    Pickler = pickle._Pickler
    xrange = range

    def _asbytes(s):
        if isinstance(s, bytes):
            return s
        return s.encode('latin1')

else:
    Unpickler = pickle.Unpickler
    Pickler = pickle.Pickler
    _asbytes = str


def hex_str(an_int):
    """Convert an int to an hexadecimal string."""
    return '{0:#x}'.format(an_int)

_MEGA = 2 ** 20

# Compressed pickle header format: _ZFILE_PREFIX followed by
# bytes which contains the length of the zlib compressed data as an
# hexadecimal string. For example: 'ZF0x139              '
_ZFILE_PREFIX = _asbytes('ZF')
_GZIP_PREFIX = b'\x1f\x8b'

try:
    import numpy as np
    from numpy.compat import asstr, asbytes

    # Those globals depends on numpy import
    MAGIC_PREFIX = asbytes('\x93NUMPY')
    MAGIC_LEN = len(MAGIC_PREFIX) + 2
    BUFFER_SIZE = 2**18  # size of buffer for reading npz files in bytes
except ImportError:
    np = None


def gzip_file_factory(f, mode='rb', compresslevel=0):
    """Factory to produce the class so that we can do a lazy import on gzip."""
    import gzip
    from gzip import WRITE

    class GzipFile(gzip.GzipFile):

        def _read_eof(self):
            pass

        def _init_read(self):
            self.crc = 0xffffffff
            self.size = 0

        def _add_read_data(self, data):
            self.crc = 0xffffffff
            offset = self.offset if PY26 else self.offset - self.extrastart
            self.extrabuf = self.extrabuf[offset:] + data
            self.extrasize = self.extrasize + len(data)
            self.extrastart = self.offset
            self.size = self.size + len(data)

    f = GzipFile(f, mode, compresslevel=compresslevel)

    return f


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
        return gzip_file_factory(filename)

    return open(filename, 'rb')


def _open_memmap(filename, array_offset=0, mode='r+', dtype=None, shape=None,
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
        fp = open(filename, 'rb')
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

    # update the offset so that it corresponds to the end of the read array
    offset += marray.nbytes

    return marray, offset


# The following functions are imported from nympy 1.10 to support numpy
# versions previous to 1.9

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
