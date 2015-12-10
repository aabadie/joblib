"""Utilities for fast persistence of big data, with optional compression."""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# Copyright (c) 2009 Gael Varoquaux
# License: BSD Style, 3 clauses.

import pickle
import sys
import io

PY3 = sys.version_info[0] >= 3
PY26 = sys.version_info[0] == 2 and sys.version_info[1] == 6

if PY3:
    Unpickler = pickle._Unpickler
    Pickler = pickle._Pickler
    xrange = range
else:
    Unpickler = pickle.Unpickler
    Pickler = pickle.Pickler


def hex_str(an_int):
    """Convert an int to an hexadecimal string."""
    return '{0:#x}'.format(an_int)

try:
    import numpy as np
    from numpy.compat import asbytes
except ImportError:
    if PY3:
        def asbytes(s):
            if isinstance(s, bytes):
                return s
            return s.encode('latin1')
    else:
        asbytes = str

    np = None

_MEGA = 2 ** 20

# Compressed pickle header format: _ZFILE_PREFIX followed by
# bytes which contains the length of the zlib compressed data as an
# hexadecimal string. For example: 'ZF0x139              '
_ZFILE_PREFIX = asbytes('ZF')
_GZIP_PREFIX = b'\x1f\x8b'


def gzip_file_factory(f, mode='rb', compresslevel=0):
    """Factory to produce the class so that we can do a lazy import on gzip."""
    import gzip
    from gzip import WRITE

    class GzipFile(gzip.GzipFile):

        def seek(self, offset, whence=0):
            # figure out new position (we can only seek forwards)
            if whence == 1:
                offset = self.offset + offset

            if whence not in [0, 1]:
                raise IOError("Illegal argument")

            if offset < self.offset:
                # for negative seek, rewind and do positive seek
                self.rewind()
                count = offset - self.offset
                for i in range(count // 1024):
                    self.read(1024)
                self.read(count % 1024)

        def tell(self):
            return self.offset

        def _init_write(self, filename):
            self.name = filename
            self.crc = 0xffffffff
            self.size = 0
            self.writebuf = []
            self.bufsize = 0

        def write(self, data):
            try:
                # works with Python < 3.5
                self._check_closed()
            except AttributeError:
                pass
            if self.mode != WRITE:
                import errno
                raise OSError(errno.EBADF,
                              "write() on read-only GzipFile object")

            if self.fileobj is None:
                raise ValueError("write() on closed GzipFile object")

            # Convert data type if called by io.BufferedWriter.
            if not PY26 and isinstance(data, memoryview):
                data = data.tobytes()

            if len(data) > 0:
                self.size = self.size + len(data)
                self.crc = 0xffffffff
                self.fileobj.write(self.compress.compress(data))
                try:
                    # works with Python < 3.5
                    self.offset += len(data)
                except AttributeError:
                    pass

            return len(data)

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


# Import utility functions/variables from numpy required for writing arrays.
# We need at least the functions introduced in version 1.9 of numpy.
if (np is not None and (int(np.version.version.split('.')[0]) > 1 or
                           (int(np.version.version.split('.')[0]) == 1 and
                            int(np.version.version.split('.')[1]) == 9))):
    from numpy.lib.format import BUFFER_SIZE
    from numpy.lib.format import _read_bytes
else:
    # For older versions of numpy, we use the ones from numpy 1.10.
    BUFFER_SIZE = 2**18  # size of buffer for reading npz files in bytes

    def _read_bytes(fp, size, error_template="ran out of data"):
        """Read from file-like object until size bytes are read.

        Raises ValueError if not EOF is encountered before size bytes are read.
        Non-blocking objects only supported if they derive from io objects.

        Required as e.g. ZipExtFile in python 2.6 can return less data than
        requested.

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
