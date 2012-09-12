"""Two-stage deferred indexer for objects with expensive __getitem__ calls."""

import numpy as np

#TODO support advanced integer indexing with non-strictly increasing indices (i.e. out-of-order and duplicates)

#--------------------------------------------------------------------------------------------------
#--- CLASS :  LazyIndexer
#--------------------------------------------------------------------------------------------------

class LazyIndexer(object):
    """Two-stage deferred indexer for objects with expensive __getitem__ calls.

    This class was originally designed to extend and speed up the indexing
    functionality of HDF5 datasets as found in :mod:`h5py`, but works on any
    equivalent object (defined as any object with `shape`, `dtype` and
    `__getitem__` members) where a call to __getitem__ may be very expensive.
    The following discussion focuses on the HDF5 use case as the main example.

    Direct extraction of a subset of an HDF5 dataset via the __getitem__
    interface (i.e. `dataset[index]`) has a few issues::

    1. Data access can be very slow (or impossible) if a very large dataset is
       fully loaded into memory and then indexed again at a later stage
    2. Advanced indexing (via boolean masks or sequences of integer indices) is
       only supported on a single dimension in the current version of h5py (2.0)
    3. Even though advanced indexing has limited support, simple indexing (via
       single integer indices or slices) is frequently much faster.

    This class wraps an :class:`h5py.Dataset` or equivalent object and exposes a
    new __getitem__ interface on it. It efficiently composes two stages of
    indexing: a first stage specified at object instantiation time and a second
    stage that applies on top of the first stage when __getitem__ is called on
    this object. The data are only loaded after the combined index is determined,
    addressing issue 1.

    Furthermore, advanced indexing is allowed on any dimension by decomposing
    the selection as a series of slice selections covering contiguous segments
    of the dimension to alleviate issue 2. Finally, this also allows faster
    data retrieval by extracting a large slice from the HDF5 dataset and then
    performing advanced indexing on the resulting :class:`numpy.ndarray` object
    instead, in response to issue 3.

    The `keep` parameter of the :meth:`__init__` and :meth:`__getitem__` methods
    accepts a generic index or slice specification, i.e. anything that would be
    accepted by the :meth:`__getitem__` method of a :class:`numpy.ndarray` of
    the same shape as the dataset. This could be a single integer index, a
    sequence of integer indices, a slice object (representing the colon operator
    commonly used with __getitem__, e.g. representing `x[1:10:2]` as
    `x[slice(1,10,2)]`), a sequence of booleans as a mask, or a tuple containing
    any number of these (typically one index item per dataset dimension). Any
    missing dimensions will be fully selected, and any extra dimensions will
    be ignored.

    Parameters
    ----------
    dataset : :class:`h5py.Dataset` object or equivalent
        Underlying dataset or array object on which lazy indexing will be done.
        This can be any object with shape, dtype and __getitem__ members.
    keep : tuple of int or slice or sequence of int or sequence of bool, optional
        First-stage index as a valid index or slice specification
        (supports arbitrary slicing or advanced indexing on any dimension)
    transform : function, signature ``data = f(data, keep)``, optional
        Transform that will be applied to data after final indexing
    shape_transform : function, signature ``new_shape = f(old_shape)``, optional
        Function that predicts data array shape after first-stage indexing and
        transformation, given its original shape as input
    dtype : :class:`numpy.dtype` object or equivalent, optional
        Type of output array after transformation, i.e. `self[:].dtype`

    Attributes
    ----------
    shape : tuple of int
        Shape of data array after first-stage indexing and transformation,
        i.e. corresponding to `self[:].shape`
    name : string
        Name of HDF5 dataset (or empty string for unnamed ndarrays, etc.)

    """
    def __init__(self, dataset, keep=slice(None), transform=lambda d, k: d, shape_transform=lambda s: s, dtype=None):
        self.dataset = dataset
        self.transform = transform
        # Ensure that keep is a tuple (then turn it into a list to simplify further processing)
        keep = list(keep) if isinstance(keep, tuple) else [keep]
        # Ensure that keep is same length as data shape (truncate or pad with blanket slices as necessary)
        keep = keep[:len(dataset.shape)] + [slice(None)] * (len(dataset.shape) - len(keep))
        self.lookup = []
        # Ensure that each index in lookup is an array of integer indices, or None
        for dim_keep, dim_len in zip(keep, dataset.shape):
            if isinstance(dim_keep, slice):
                # Turn slice into array of integer indices (or None if it contains all indices)
                dim_keep = dim_keep.indices(dim_len)
                dim_keep = np.arange(*dim_keep) if dim_keep != slice(None).indices(dim_len) else None
            else:
                dim_keep = np.atleast_1d(dim_keep)
                # Turn boolean mask into integer indices (True means keep that index), or None if all is True
                if dim_keep.dtype == np.bool and len(dim_keep) == dim_len:
                    dim_keep = np.nonzero(dim_keep)[0] if not dim_keep.all() else None
            self.lookup.append(dim_keep)
        self.shape = shape_transform(tuple([(len(dim_keep) if dim_keep is not None else dim_len)
                                            for dim_keep, dim_len in zip(self.lookup, dataset.shape)]))
        self.dtype = np.dtype(dtype) if dtype is not None else dataset.dtype
        self.name = getattr(self.dataset, 'name', '')

    def __repr__(self):
        """Short human-friendly string representation of lazy indexer object."""
        return "<katfile.%s '%s': shape %s, type '%s' at 0x%x>" % \
               (self.__class__.__name__, self.name, self.shape, self.dtype, id(self))

    def __len__(self):
        """Length operator."""
        return self.shape[0]

    def __getitem__(self, keep):
        """Extract a selected array from the underlying dataset.

        This applies the given second-stage index on top of the first-stage index
        and retrieves the relevant data from the dataset as an array, optionally
        transforming it afterwards.

        Parameters
        ----------
        keep : tuple of int or slice or sequence of int or sequence of bool
            Second-stage index as a valid index or slice specification
            (supports arbitrary slicing or advanced indexing on any dimension)

        Returns
        -------
        data : array
            Extracted output array

        """
        ndim = len(self.dataset.shape)
        # Ensure that keep is a tuple (then turn it into a list to simplify further processing)
        keep = list(keep) if isinstance(keep, tuple) else [keep]
        # The original keep tuple will be passed to data transform
        original_keep = tuple(keep)
        # Ensure that keep is same length as data dimension (truncate or pad with blanket slices as necessary)
        keep = keep[:ndim] + [slice(None)] * (ndim - len(keep))
        # Map current selection to original data indices based on any existing initial selection, per data dimension
        keep = [(dkeep if dlookup is None else dlookup[dkeep]) for dkeep, dlookup in zip(keep, self.lookup)]
        # Iterate over dimensions of dataset, storing information on selection on each dimension:
        # `selection` is a list with one element per dimension; each element is a list of contiguous segments along
        # the dimension, and each segment is represented by a tuple of 3 elements:
        # (dataset selection, post-selection, output array selection)
        # Similarly, `segment_sizes` is a list of lists of segment lengths (empty lists for scalar-selected dimensions)
        selection, segment_sizes = [], []
        for dim_keep, dim_len in zip(keep, self.dataset.shape):
            if np.isscalar(dim_keep):
                # If selection is a scalar, pass directly to dataset selector and remove dimension from output
                selection.append([(dim_keep, None, None)])
                segment_sizes.append([])
            elif isinstance(dim_keep, slice):
                # If selection is a slice, pass directly to dataset selector without post-selection
                start, stop, stride = dim_keep.indices(dim_len)
                segm_size = len(range(start, stop, stride))
                selection.append([(slice(start, stop, stride), slice(None), slice(0, segm_size, 1))])
                segment_sizes.append([segm_size])
            elif len(dim_keep) == 0:
                # If selection is empty, pass to post-selector, as HDF5 datasets do not support zero-length selection
                selection.append([(slice(0, 1, 1), slice(0, 0, 1), slice(0, 0, 1))])
                segment_sizes.append([0])
            else:
                # Anything else is advanced indexing via bool or integer sequences
                dim_keep = np.atleast_1d(dim_keep)
                # Turn boolean mask into integer indices (True means keep that index)
                if dim_keep.dtype == np.bool and len(dim_keep) == dim_len:
                    dim_keep = np.nonzero(dim_keep)[0]
                elif not np.all(dim_keep == np.unique(dim_keep)):
                    raise TypeError('LazyIndexer cannot handle duplicate or unsorted advanced integer indices')
                # Split indices into multiple contiguous segments (specified by first and one-past-last data indices)
                jumps = np.nonzero(np.diff(dim_keep) > 1)[0]
                first = [dim_keep[0]] + dim_keep[jumps + 1].tolist()
                last = dim_keep[jumps].tolist() + [dim_keep[-1]]
                segments = np.c_[first, np.array(last) + 1]
                if len(dim_keep) > 0.2 * dim_len and len(segments) > 1:
                    # If more than 20% of data are selected in 2 or more separate segments (the Ratcliffian benchmark),
                    # select data at dataset level with a single slice spanning segments and then postselect the ndarray
                    selection.append([(slice(segments[0, 0], segments[-1, 1], 1),
                                       dim_keep - dim_keep[0], slice(0, len(dim_keep), 1))])
                    segment_sizes.append([len(dim_keep)])
                else:
                    # Turn each segment into a separate slice at dataset level without post-selection,
                    # and construct contiguous output slices of the same segment sizes
                    segm_sizes = [end - start for start, end in segments]
                    segm_starts = np.cumsum([0] + segm_sizes)
                    selection.append([(slice(start, end, 1), slice(None), slice(segm_starts[n], segm_starts[n + 1], 1))
                                      for n, (start, end) in enumerate(segments)])
                    segment_sizes.append(segm_sizes)
        # Short-circuit the selection if all dimensions are selected with scalars (resulting in a scalar output)
        if segment_sizes == [[]] * ndim:
            return self.transform(self.dataset[tuple([select[0][0] for select in selection])], original_keep)
        # Use a dense N-dimensional meshgrid to slice the data set into chunks, based on segments along each dimension
        chunk_indices = np.mgrid[[slice(0, len(select), 1) for select in selection]]
        # Pre-allocate output ndarray to have the correct shape and dtype (will be at least 1-dimensional)
        out_data = np.empty([np.sum(segments) for segments in segment_sizes if segments], dtype=self.dataset.dtype)
        # Iterate over chunks, extracting them from dataset and inserting them into the right spot in output array
        for chunk_index in chunk_indices.reshape(ndim, -1).T:
            # Extract chunk from dataset (don't use any advanced indexing here, only scalars and slices)
            dataset_select = tuple([select[segment][0] for select, segment in zip(selection, chunk_index)])
            chunk = self.dataset[dataset_select]
            # Perform post-selection on chunk (can be fancier / advanced indexing because chunk is now an ndarray)
            post_select = [select[segment][1] for select, segment in zip(selection, chunk_index)]
            # If any dimensions were dropped due to scalar indexing, drop them from post_select/out_select tuples too
            post_select = tuple([select for select in post_select if select is not None])
            # Do post-selection one dimension at a time, as ndarray does not allow simultaneous advanced indexing
            # on more than one dimension. This caters for the scenario where more than one dimension satisfies
            # the Ratcliffian benchmark (the only way to get advanced post-selection).
            for dim in range(len(chunk.shape)):
                # Only do post-selection on this dimension if non-trivial (otherwise an unnecessary copy happens)
                if not (isinstance(post_select[dim], slice) and post_select[dim] == slice(None)):
                    # Prepend the appropriate number of colons to the selection to place it at the correct dimension
                    chunk = chunk[[slice(None)] * dim + [post_select[dim]]]
            # Determine appropriate output selection and insert chunk into output array
            out_select = [select[segment][2] for select, segment in zip(selection, chunk_index)]
            out_select = tuple([select for select in out_select if select is not None])
            out_data[out_select] = chunk
        return self.transform(out_data, original_keep)
