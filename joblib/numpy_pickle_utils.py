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


def gzip_file_factory(f, mode='rb', compresslevel=0):
    """Factory to produce the class so that we can do a lazy import on gzip."""
    import gzip
    from gzip import WRITE

    class GzipFile(gzip.GzipFile):

        def _init_write(self, filename):
            self.name = filename
            self.crc = 0xffffffff
            self.size = 0
            self.writebuf = []
            self.bufsize = 0

        def write(self,data):
            self._check_closed()
            if self.mode != WRITE:
                import errno
                raise OSError(errno.EBADF, "write() on read-only GzipFile object")

            if self.fileobj is None:
                raise ValueError("write() on closed GzipFile object")

            # Convert data type if called by io.BufferedWriter.
            if isinstance(data, memoryview):
                data = data.tobytes()

            if len(data) > 0:
                self.size = self.size + len(data)
                self.crc = 0xffffffff
                self.fileobj.write(self.compress.compress(data))
                self.offset += len(data)

            return len(data)

        def _read_eof(self):
            pass

        def _init_read(self):
            self.crc = 0xffffffff
            self.size = 0

        def _add_read_data(self, data):
            self.crc = 0xffffffff
            offset = self.offset - self.extrastart
            self.extrabuf = self.extrabuf[offset:] + data
            self.extrasize = self.extrasize + len(data)
            self.extrastart = self.offset
            self.size = self.size + len(data)

    f = GzipFile(f, mode, compresslevel=compresslevel)

    return f
