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
