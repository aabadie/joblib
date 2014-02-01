"""
Utilities for fast persistence of big data, with optional compression.
"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# Copyright (c) 2009 Gael Varoquaux
# License: BSD Style, 3 clauses.

import pickle
import traceback
import sys
import os
import zlib
import gzip
import warnings
from contextlib import closing

from .numpy_pickle_utils import PY3, _ZFILE_PREFIX, _MEGA
from .numpy_pickle_utils import load_compatibility
from .numpy_pickle_utils import asbytes, hex_str
from .numpy_pickle_utils import Unpickler, Pickler, NDArrayWrapper
from ._compat import _basestring

from io import BytesIO

_GZIP_PREFIX = b'\x1f\x8b'

###############################################################################
# Cache file utilities

def _read_magic(file_handle):
    """ Utility to check the magic signature of a file identifying it as a
        Zfile
    """
    magic = file_handle.read(len(_ZFILE_PREFIX))
    # Pickling needs file-handles at the beginning of the file
    file_handle.seek(0)
    return magic

def _check_filetype(filename, magic):
    """ Utility function opening and returning a fileobject from a filename
    given it's magic number (compressed using gzip or not compressed)"""

    if magic == _GZIP_PREFIX:
        return gzip.GzipFile(filename, 'rb')

    return open(filename, 'rb')

###############################################################################
# Utility objects for persistence.

class NPArrayWrapper(object):
    """ An object to be persisted instead of numpy arrays.

        The only thing this object does, is to carry the filename in which
        the array has been persisted, and the array subclass.
    """
    def __init__(self, filename, subclass, allow_mmap=True, compress=False):
        "Store the useful information for later"
        self.filename = filename
        self.subclass = subclass
        self.allow_mmap = allow_mmap
        self.compress = compress

    def read(self, unpickler):
        "Reconstruct the array"
        filename = os.path.join(unpickler._dirname, self.filename)
        if self.compress:
            # Closing is required on GzipFile with python 2.6
            with closing(gzip.GzipFile(filename, 'rb')) as fz:
                array =  unpickler.np.load(fz)
        else:
            # Load the array from the disk
            # use getattr instead of self.allow_mmap to ensure backward compat
            # with NDArrayWrapper instances pickled with joblib < 0.9.0
            allow_mmap = getattr(self, 'allow_mmap', True)
            memmap_kwargs = ({} if not allow_mmap
                            else {'mmap_mode': unpickler.mmap_mode})
            array = unpickler.np.load(filename, **memmap_kwargs)

        # Reconstruct subclasses. This does not work with old
        # versions of numpy
        if (hasattr(array, '__array_prepare__')
                and self.subclass not in (unpickler.np.ndarray,
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
    """
    dispatch = Pickler.dispatch.copy()

    def __init__(self, filename, compress=0, cache_size=10, protocol=None):
        self._filename = filename
        self._filenames = [filename, ]
        self.cache_size = cache_size
        self.compress = compress
        if compress:
            self.file = gzip.GzipFile(filename, 'wb')
        else:
            self.file = open(filename, 'wb')

        # Count the number of npy files that we have created:
        self._npy_counter = 0
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

    def _write_array(self, array, filename):
        if not self.compress:
            self.np.save(filename, array)
            allow_mmap = not array.dtype.hasobject
            compress = False
        else:
            filename += '.z'
            # Closing is required on GzipFile with python 2.6
            with closing(gzip.GzipFile(filename, 'wb',
                                       compresslevel=self.compress)) as fz:
                self.np.save(fz, array)
            compress = True
            allow_mmap = False

        container = NPArrayWrapper(os.path.basename(filename),
                                   type(array),
                                   allow_mmap=allow_mmap,
                                   compress=compress)

        return container, filename

    def save(self, obj):
        """ Subclass the save method, to save ndarray subclasses in npy
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
                    # Pickling doesn't work with memmaped arrays
                    obj = self.np.asarray(obj)
                return Pickler.save(self, obj)

            if not obj.dtype.hasobject:
                try:
                    filename = '%s_%02i.npy' % (self._filename,
                                                self._npy_counter)
                    # This converts the array in a container
                    obj, filename = self._write_array(obj, filename)
                    self._filenames.append(filename)
                    self._npy_counter += 1
                except Exception:
                    # XXX: We should have a logging mechanism
                    print('Failed to save %s to .npy file:\n%s' % (
            self._npy_counter += 1
            try:
                filename = '%s_%02i.npy' % (self._filename,
                                            self._npy_counter)
                # This converts the array in a container
                obj, filename = self._write_array(obj, filename)
                self._filenames.append(filename)
            except:
                self._npy_counter -= 1
                # XXX: We should have a logging mechanism
                print('Failed to save %s to .npy file:\n%s' % (
                    type(obj),
                    traceback.format_exc()))

        return Pickler.save(self, obj)

class NumpyUnpickler(Unpickler):
    """A subclass of the Unpickler to unpickle our numpy pickles.
    """
    dispatch = Unpickler.dispatch.copy()

    def __init__(self, filename, file_handle, mmap_mode=None):
        self._filename = os.path.basename(filename)
        self._dirname = os.path.dirname(filename)
        self.mmap_mode = mmap_mode
        self.file_handle = self._open_pickle(file_handle)
        Unpickler.__init__(self, self.file_handle)
        try:
            import numpy as np
        except ImportError:
            np = None
        self.np = np

    def _open_pickle(self, file_handle):
        return file_handle

    def load_build(self):
        """ This method is called to set the state of a newly created
            object.
            We capture it to replace our place-holder objects,
            NDArrayWrapper, by the array we are interested in. We
            replace them directly in the stack of pickler.
        """
        Unpickler.load_build(self)
        if (isinstance(self.stack[-1], NPArrayWrapper) or
            isinstance(self.stack[-1], NDArrayWrapper)):
            if self.np is None:
                raise ImportError("Trying to unpickle an ndarray, "
                                  "but numpy didn't import correctly")
            np_array_wrapper = self.stack.pop()
            array = np_array_wrapper.read(self)
            self.stack.append(array)

    # Be careful to register our new method.
    if PY3:
        dispatch[pickle.BUILD[0]] = load_build
    else:
        dispatch[pickle.BUILD] = load_build


###############################################################################
# Utility functions

def dump(value, filename, compress=0, cache_size=100, protocol=None):
    """Fast persistence of an arbitrary Python object into one or multiple
    files, with dedicated storage for numpy arrays.

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
    if compress is True:
        # By default, if compress is enabled, we want to be using 3 by
        # default
        compress = 3
    if not isinstance(filename, _basestring):
        # People keep inverting arguments, and the resulting error is
        # incomprehensible
        raise ValueError(
            'Second argument should be a filename, %s (type %s) was given'
            % (filename, type(filename))
        )

    try:
        pickler = NumpyPickler(filename,
                               compress=compress,
                               cache_size=cache_size,
                               protocol=protocol)
        pickler.dump(value)
    finally:
        if 'pickler' in locals() and hasattr(pickler, 'file'):
            pickler.file.flush()
            pickler.file.close()
    return pickler._filenames


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
                                   file_handle=file_handle,
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
