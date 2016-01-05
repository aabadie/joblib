"""Utilities for fast persistence of big data, with optional compression."""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# Copyright (c) 2009 Gael Varoquaux
# License: BSD Style, 3 clauses.

import pickle
import sys
import io
import zlib
import gzip
import bz2

try:
    from threading import RLock
except ImportError:
    from dummy_threading import RLock

PY3 = sys.version_info[0] >= 3
PY26 = sys.version_info[:2] == (2, 6)
PY27 = sys.version_info[:2] == (2, 7)

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

if PY3:
    def asbytes(s):
        if isinstance(s, bytes):
            return s
        return s.encode('latin1')
else:
    asbytes = str

try:
    import numpy as np
except ImportError:
    np = None

try:
    import lzma
except ImportError:
    lzma = None


# Magic numbers of supported compression file formats.        '
_ZFILE_PREFIX = asbytes('ZF')
_GZIP_PREFIX = b'\037\213'
_BZ2_PREFIX = asbytes('BZ')
_LZMA_PREFIX = b'\x5d\x00'

# The magic number of the file format used by joblib.dump with zlib compression.
_JOBLIB_PREFIX = asbytes('JLZF')

# Buffer size used in io.BufferedReader and io.BufferedWriter
_IO_BUFFER_SIZE = 10 * 1024 ** 2


###############################################################################
# Cache file utilities


def _check_magic(filename, magic):
    """Utility to check the magic signature of a file.

    _ZFILE_PREFIX is used to determine the number of bytes to read to determine
    the magic number.

    Parameters
    ----------
    file_handle: file_like

    """
    with open(filename, 'rb') as f:
        read_magic = f.read(len(magic))

    return magic == read_magic


def _buffered_read_file(fobj, buffer_size):
    """Return a buffered version of a file object."""
    if PY26 or (PY27 and isinstance(fobj, bz2.BZ2File)):
        # Python 2.6 doesn't fully support io.BufferedReader.
        # Python 2.7 doesn't work with BZ2File through a buffer: no attribute
        # 'readable'.
        # In those cases, we directly return the file object.
        return fobj
    else:
        return io.BufferedReader(fobj, buffer_size=buffer_size)


def _check_filetype(filename, buffer_size=_IO_BUFFER_SIZE):
    """Utility function opening the right fileobject from a filename.

    The magic number is used to choose between the type of file object to open:
    * regular file object (default)
    * gzip file object if magic matches _GZIP_PREFIX
    * bz2 file object if magic matches _BZ2_PREFIX
    * lzma file object if magic matches _LZMA_PREFIX

    Parameters
    ----------
    filename: str
        The full path string of the file
    buffer_size: positive int
        The memory buffer size used for reading the file.

    Returns
    -------
        a file like object

    """
    if _check_magic(filename, _JOBLIB_PREFIX):
        return _buffered_read_file(JoblibZFile(filename, 'rb'), buffer_size)
    elif _check_magic(filename, _GZIP_PREFIX):
        return _buffered_read_file(gzip.GzipFile(filename, 'rb'), buffer_size)
    elif _check_magic(filename, _BZ2_PREFIX):
        return _buffered_read_file(bz2.BZ2File(filename, 'rb'), buffer_size)
    elif _check_magic(filename, _LZMA_PREFIX):
        if lzma:
            return _buffered_read_file(lzma.LZMAFile(filename), buffer_size)
        else:
            raise NotImplementedError("Lzma decompression is not available for "
                                      "this version of python {0}, joblib "
                                      "can't read file '{1}'."
                                      .format(sys.version, filename))

    return open(filename, 'rb')


_MODE_CLOSED = 0
_MODE_READ = 1
_MODE_READ_EOF = 2
_MODE_WRITE = 3
_BUFFER_SIZE = 8192


class JoblibZFile(io.BufferedIOBase):
    """A file object providing transparent zlib (de)compression.

    A JoblibZFile can act as a wrapper for an existing file object, or refer
    directly to a named file on disk.

    Note that JoblibZFile provides only a *binary* file interface: data read is
    returned as bytes, and data to be written should be given as bytes.


    This object is an adapation of the BZ2File object and is compatible with
    versions of python >= 2.6.
    """

    def __init__(self, filename, mode="r", compresslevel=9):
        """Open a zlib-compressed joblib file.

        If filename is a str or bytes object, it gives the name
        of the file to be opened. Otherwise, it should be a file object,
        which will be used to read or write the compressed data.

        mode can be 'r' for reading (default) or 'w' for (over)writing

        If mode is 'w', compresslevel can be a number between 1
        and 9 specifying the level of compression: 1 produces the least
        compression, and 9 (default) produces the most compression.

        If mode is 'r', the input file may be the concatenation of
        multiple compressed streams.
        """
        # This lock must be recursive, so that BufferedIOBase's
        # readline(), readlines() and writelines() don't deadlock.
        self._lock = RLock()
        self._fp = None
        self._closefp = False
        self._mode = _MODE_CLOSED
        self._pos = 0
        self._size = -1
        self.compresslevel = compresslevel

        if not isinstance(compresslevel, int) or not (1 <= compresslevel <= 9):
            raise ValueError("compresslevel must be between 1 and 9")

        try:
            import blosc
        except ImportError:
            blosc = None
        self.blosc = blosc

        if mode in ("", "r", "rb"):
            mode = "rb"
            mode_code = _MODE_READ
            self._decompressor = zlib.decompressobj(-zlib.MAX_WBITS)
            self._buffer = b""
            self._buffer_offset = 0
        elif mode in ("w", "wb"):
            mode = "wb"
            mode_code = _MODE_WRITE
            self._compressor = zlib.compressobj(compresslevel,
                                                zlib.DEFLATED,
                                                -zlib.MAX_WBITS,
                                                zlib.DEF_MEM_LEVEL,
                                                0)
        else:
            raise ValueError("Invalid mode: %r" % (mode,))

        if isinstance(filename, (str, bytes)):
            self._fp = open(filename, mode)
            self._closefp = True
            self._mode = mode_code
        elif hasattr(filename, "read") or hasattr(filename, "write"):
            self._fp = filename
            self._mode = mode_code
        else:
            raise TypeError("filename must be a str or bytes object, or a file")

        if self._mode == _MODE_WRITE:
            self._fp.write(_JOBLIB_PREFIX)
        if self._mode == _MODE_READ:
            self._fp.read(len(_JOBLIB_PREFIX))

    def close(self):
        """Flush and close the file.

        May be called more than once without error. Once the file is
        closed, any other operation on it will raise a ValueError.
        """
        with self._lock:
            if self._mode == _MODE_CLOSED:
                return
            try:
                if self._mode in (_MODE_READ, _MODE_READ_EOF):
                    self._decompressor = None
                elif self._mode == _MODE_WRITE:
                    self._fp.write(self._compressor.flush())
                    self._compressor = None
            finally:
                try:
                    if self._closefp:
                        self._fp.close()
                finally:
                    self._fp = None
                    self._closefp = False
                    self._mode = _MODE_CLOSED
                    self._buffer = b""
                    self._buffer_offset = 0

    @property
    def closed(self):
        """True if this file is closed."""
        return self._mode == _MODE_CLOSED

    def fileno(self):
        """Return the file descriptor for the underlying file."""
        self._check_not_closed()
        return self._fp.fileno()

    def seekable(self):
        """Return whether the file supports seeking."""
        return self.readable() and self._fp.seekable()

    def readable(self):
        """Return whether the file was opened for reading."""
        self._check_not_closed()
        return self._mode in (_MODE_READ, _MODE_READ_EOF)

    def writable(self):
        """Return whether the file was opened for writing."""
        self._check_not_closed()
        return self._mode == _MODE_WRITE

    # Mode-checking helper functions.

    def _check_not_closed(self):
        if self.closed:
            raise ValueError("I/O operation on closed file")

    def _check_can_read(self):
        if self._mode not in (_MODE_READ, _MODE_READ_EOF):
            self._check_not_closed()
            raise io.UnsupportedOperation("File not open for reading")

    def _check_can_write(self):
        if self._mode != _MODE_WRITE:
            self._check_not_closed()
            raise io.UnsupportedOperation("File not open for writing")

    def _check_can_seek(self):
        if self._mode not in (_MODE_READ, _MODE_READ_EOF):
            self._check_not_closed()
            raise io.UnsupportedOperation("Seeking is only supported "
                                          "on files open for reading")
        if not self._fp.seekable():
            raise io.UnsupportedOperation("The underlying file object "
                                          "does not support seeking")

    # Fill the readahead buffer if it is empty. Returns False on EOF.
    def _fill_buffer(self):
        if self._mode == _MODE_READ_EOF:
            return False
        # Depending on the input data, our call to the decompressor may not
        # return any data. In this case, try again after reading another block.
        while self._buffer_offset == len(self._buffer):
            try:
                rawblock = (self._decompressor.unused_data or
                            self._fp.read(_BUFFER_SIZE))

                if not rawblock:
                    raise EOFError
            except EOFError:
                # End-of-stream marker and end of file. We're good.
                self._mode = _MODE_READ_EOF
                self._size = self._pos
                return False
            else:
                if self.blosc is None:
                    self._buffer = self._decompressor.decompress(rawblock)
                else:
                    self._buffer = self.blosc.decompress(rawblock)
            self._buffer_offset = 0
        return True

    # Read data until EOF.
    # If return_data is false, consume the data without returning it.
    def _read_all(self, return_data=True):
        # The loop assumes that _buffer_offset is 0. Ensure that this is true.
        self._buffer = self._buffer[self._buffer_offset:]
        self._buffer_offset = 0

        blocks = []
        while self._fill_buffer():
            if return_data:
                blocks.append(self._buffer)
            self._pos += len(self._buffer)
            self._buffer = b""
        if return_data:
            return b"".join(blocks)

    # Read a block of up to n bytes.
    # If return_data is false, consume the data without returning it.
    def _read_block(self, n, return_data=True):
        # If we have enough data buffered, return immediately.
        end = self._buffer_offset + n
        if end <= len(self._buffer):
            data = self._buffer[self._buffer_offset: end]
            self._buffer_offset = end
            self._pos += len(data)
            return data if return_data else None

        # The loop assumes that _buffer_offset is 0. Ensure that this is true.
        self._buffer = self._buffer[self._buffer_offset:]
        self._buffer_offset = 0

        blocks = []
        while n > 0 and self._fill_buffer():
            if n < len(self._buffer):
                data = self._buffer[:n]
                self._buffer_offset = n
            else:
                data = self._buffer
                self._buffer = b""
            if return_data:
                blocks.append(data)
            self._pos += len(data)
            n -= len(data)
        if return_data:
            return b"".join(blocks)

    def peek(self, n=0):
        """Return buffered data without advancing the file position.

        Always returns at least one byte of data, unless at EOF.
        The exact number of bytes returned is unspecified.
        """
        with self._lock:
            self._check_can_read()
            if not self._fill_buffer():
                return b""
            return self._buffer[self._buffer_offset:]

    def read(self, size=-1):
        """Read up to size uncompressed bytes from the file.

        If size is negative or omitted, read until EOF is reached.
        Returns b'' if the file is already at EOF.
        """
        with self._lock:
            self._check_can_read()
            if size == 0:
                return b""
            elif size < 0:
                return self._read_all()
            else:
                return self._read_block(size)

    def readinto(self, b):
        """Read up to len(b) bytes into b.

        Returns the number of bytes read (0 for EOF).
        """
        with self._lock:
            return io.BufferedIOBase.readinto(self, b)

    def write(self, data):
        """Write a byte string to the file.

        Returns the number of uncompressed bytes written, which is
        always len(data). Note that due to buffering, the file on disk
        may not reflect the data written until close() is called.
        """
        with self._lock:
            self._check_can_write()
            # Convert data type if called by io.BufferedWriter.
            if not PY26 and isinstance(data, memoryview):
                data = data.tobytes()

            if self.blosc is None:
                compressed = self._compressor.compress(data)
            else:
                compressed = self.blosc.compress(data, typesize=8,
                                                 clevel=self.compresslevel,
                                                 shuffle=False,
                                                 cname='zlib')
            self._fp.write(compressed)
            self._pos += len(data)
            return len(data)

    # Rewind the file to the beginning of the data stream.
    def _rewind(self):
        self._fp.seek(0, 0)
        self._mode = _MODE_READ
        self._pos = 0
        self._decompressor = zlib.decompressobj(-zlib.MAX_WBITS)
        self._buffer = b""
        self._buffer_offset = 0

    def seek(self, offset, whence=0):
        """Change the file position.

        The new position is specified by offset, relative to the
        position indicated by whence. Values for whence are:

            0: start of stream (default); offset must not be negative
            1: current stream position
            2: end of stream; offset must not be positive

        Returns the new file position.

        Note that seeking is emulated, so depending on the parameters,
        this operation may be extremely slow.
        """
        with self._lock:
            self._check_can_seek()

            # Recalculate offset as an absolute file position.
            if whence == 0:
                pass
            elif whence == 1:
                offset = self._pos + offset
            elif whence == 2:
                # Seeking relative to EOF - we need to know the file's size.
                if self._size < 0:
                    self._read_all(return_data=False)
                offset = self._size + offset
            else:
                raise ValueError("Invalid value for whence: %s" % (whence,))

            # Make it so that offset is the number of bytes to skip forward.
            if offset < self._pos:
                self._rewind()
            else:
                offset -= self._pos

            # Read and discard data until we reach the desired position.
            self._read_block(offset, return_data=False)

            return self._pos

    def tell(self):
        """Return the current file position."""
        with self._lock:
            self._check_not_closed()
            return self._pos


# Utility functions/variables from numpy required for writing arrays.
# We need at least the functions introduced in version 1.9 of numpy. Here,
# we use the ones from numpy 1.10.
BUFFER_SIZE = 2**18  # size of buffer for reading npz files in bytes


def _read_bytes(fp, size, error_template="ran out of data"):
    """Read from file-like object until size bytes are read.

    Raises ValueError if not EOF is encountered before size bytes are read.
    Non-blocking objects only supported if they derive from io objects.

    Required as e.g. ZipExtFile in python 2.6 can return less data than
    requested.

    This function was taken from numpy/lib/format.py in version 1.10.

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
