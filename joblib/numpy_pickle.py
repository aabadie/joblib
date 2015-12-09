"""Utilities for fast persistence of big data, with optional compression."""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# Copyright (c) 2009 Gael Varoquaux
# License: BSD Style, 3 clauses.

import pickle
import os
import gzip
import warnings

from contextlib import closing

from .numpy_pickle_utils import PY3
from .numpy_pickle_utils import _ZFILE_PREFIX
from .numpy_pickle_utils import Unpickler, Pickler
from .numpy_pickle_utils import gzip_file_factory
from .numpy_pickle_utils import _read_magic, _check_filetype
from .numpy_pickle_utils import _open_memmap
from .numpy_pickle_compat import load_compatibility, NDArrayWrapper
from ._compat import _basestring

###############################################################################
# Utility objects for persistence.


class NumpyArrayWrapper(object):
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

    def __init__(self, subclass, dtype, allow_mmap=True):
        """Constructor. Store the useful information for later."""
        self.subclass = subclass
        self.allow_mmap = allow_mmap
        self.dtype = dtype

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
        # Now we read the array stored at current offset
        # position in file handle
        if (self.allow_mmap and
                unpickler.mmap_mode is not None and
                not self.dtype.hasobject):
            array, next_offset = _open_memmap(unpickler.filename,
                                              unpickler.file_handle.tell(),
                                              mode=unpickler.mmap_mode)
            unpickler.file_handle.seek(next_offset)
        else:
            array = unpickler.np.lib.format.read_array(unpickler.file_handle)

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
    * persistence of numpy arrays in a single file.

    * optional compression using GzipFile, with a special care on avoid
    temporaries.

    Attributes
    ----------
    fp: file
        File object handle used for serializing the input object.
    protocol: int
        Pickle protocol used. Default is pickle.DEFAULT_PROTOCOL under
        python 3, pickle.HIGHEST_PROTOCOL otherwise.
    """

    dispatch = Pickler.dispatch.copy()

    def __init__(self, fp, protocol=None):
        """Constructor. Store the useful information for later."""
        self.file = fp
        self.compress = isinstance(self.file, gzip.GzipFile)

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

    def _create_array_wrapper(self, array):
        """Create and returns a numpy array wrapper from a numpy array."""
        allow_mmap = not array.dtype.hasobject and not self.compress
        wrapper = NumpyArrayWrapper(type(array),
                                    array.dtype,
                                    allow_mmap=allow_mmap)

        return wrapper

    def save(self, obj):
        """Subclass the Pickler `save` method.

        This is a total abuse of the Pickler class in order to use the numpy
        persistence function `save` instead of the default pickle
        implementation. The numpy array is replaced by a custom wrapper in the
        pickle persistence stack and the serialized array is written right
        after in the file. Warning: this breaks the pickle format and prevents
        the usage of pickletools.dis() function.
        """
        if self.np is not None and type(obj) in (self.np.ndarray,
                                                 self.np.matrix,
                                                 self.np.memmap):
            if type(obj) is self.np.memmap:
                # Pickling doesn't work with memmapped arrays
                obj = self.np.asarray(obj)
                return Pickler.save(self, obj)

            # This converts on the fly the array in a wrapper
            Pickler.save(self, self._create_array_wrapper(obj))

            # Array is stored right after the wrapper
            self.np.save(self.file, obj)
            return

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

    """

    dispatch = Unpickler.dispatch.copy()

    def __init__(self, filename, file_handle, mmap_mode=None):
        """Constructor."""
        # The next line is for backward compatibility with pickle generated
        # with joblib versions less than 0.10.
        self._dirname = os.path.dirname(filename)

        self.mmap_mode = mmap_mode
        self.file_handle = file_handle
        self.filename = filename
        self.compat_mode = False
        Unpickler.__init__(self, self.file_handle)
        try:
            import numpy as np
        except ImportError:
            np = None
        self.np = np

    def load_build(self):
        """Called to set the state of a newly created object.

        We capture it to replace our place-holder objects,
        NDArrayWrapper, by the array we are interested in. We
        replace them directly in the stack of pickler.
        """
        Unpickler.load_build(self)

        # For back backward compatibility
        if isinstance(self.stack[-1], (NDArrayWrapper, NumpyArrayWrapper)):
            if self.np is None:
                raise ImportError("Trying to unpickle an ndarray, "
                                  "but numpy didn't import correctly")
            array_wrapper = self.stack.pop()
            self.compat_mode = isinstance(array_wrapper, NDArrayWrapper)
            self.stack.append(array_wrapper.read(self))

    # Be careful to register our new method.
    if PY3:
        dispatch[pickle.BUILD[0]] = load_build
    else:
        dispatch[pickle.BUILD] = load_build


###############################################################################
# Utility functions

def dump(value, filename, compress=0, protocol=None, cache_size=None):
    """Persist an arbitrary Python object.

    Fast persistence of an arbitrary Python object into one file with
    dedicated storage for numpy arrays.

    Parameters
    -----------
    value: any Python object
        The object to store to disk
    filename: str
        The name of the file in which it is to be stored
    compress: int from 0 to 9 or bool, optional
        Optional compression level for the data. 0 or False is no compression.
        Higher means more compression, but also slower read and
        write times. Using a value of 3 is often a good compromise.
        See the notes for more details.
        If compress is True, the compression level used is 3.
    protocol: positive int
        Pickle protocol, see pickle.dump documentation for more details.
    cache_size: positive int, optional
        Fixes the order of magnitude (in megabytes) of the cache used
        for in-memory compression. Note that this is just an order of
        magnitude estimate and that for big arrays, the code will go
        over this value at dump and at load time.
        This option is deprecated in 0.10.

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
        # By default, if compress is enabled, we want to be using 3 by default
        compress = 3

    if not isinstance(filename, _basestring):
        # People keep inverting arguments, and the resulting error is
        # incomprehensible
        raise ValueError(
            'Second argument should be a filename, %s (type %s) was given'
            % (filename, type(filename))
        )

    if cache_size is not None:
        # Cache size is deprecated starting from version 0.10
        warnings.warn("Cache size is deprecated and will be ignored.",
                      DeprecationWarning, stacklevel=2)

    try:
        if compress > 0:
            fp = gzip_file_factory(filename, 'wb',
                                   compresslevel=compress)
        else:
            fp = open(filename, 'wb')
        pickler = NumpyPickler(fp, protocol=protocol)
        pickler.dump(value)
    finally:
        if 'pickler' in locals() and hasattr(pickler, 'file'):
            fp.flush()
            fp.close()
    return [filename]


def load(filename, mmap_mode=None):
    """Reconstruct a Python object from a file persisted with joblib.dump.

    Parameters
    -----------
    filename: str
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
    with open(filename, 'rb') as file_handle:
        magic = _read_magic(file_handle)

    # Backward compatibility with old compression strategy
    if magic == _ZFILE_PREFIX:
        warnings.warn("The file '%s' has been generated with a joblib version "
                      "less than 0.10. "
                      "Please regenerate this pickle file." % filename,
                      DeprecationWarning, stacklevel=2)
        return load_compatibility(filename)

    with closing(_check_filetype(filename, magic)) as file_handle:
        if isinstance(file_handle, gzip.GzipFile) and mmap_mode is not None:
            warnings.warn('File "%(filename)s" appears to be compressed, '
                          'this is not compatible with mmap_mode '
                          '"%(mmap_mode)s" flag passed' % locals(),
                          DeprecationWarning, stacklevel=2)

        # We are careful to open the file handle early and keep it open to
        # avoid race-conditions on renames.
        # That said, if data are stored in companion files, which can be the
        # case with the old persistence format, moving the directory will
        # create a race when joblib tries to access the companion files.
        unpickler = NumpyUnpickler(filename,
                                   file_handle,
                                   mmap_mode=mmap_mode)
        try:
            obj = unpickler.load()
            if unpickler.compat_mode:
                warnings.warn("The file '%s' has been generated with a joblib "
                              "version less than 0.10. "
                              "Please regenerate this pickle file." % filename,
                              DeprecationWarning, stacklevel=2)
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
