################################################################################
# Copyright (c) 2017-2021, National Research Foundation (SARAO)
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy
# of the License at
#
#   https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

"""A (lazy) container for the triplet of visibilities, flags and weights."""

import itertools
import logging

import dask.array as da
import numba
import numpy as np
import toolz
from dask.array.rechunk import intersect_chunks
from dask.highlevelgraph import HighLevelGraph

from .chunkstore import PlaceholderChunk
from .flags import DATA_LOST
from .van_vleck import autocorr_lookup_table

logger = logging.getLogger(__name__)


class VisFlagsWeights:
    """Correlator data in the form of visibilities, flags and weights.

    This container stores the triplet of visibilities, flags and weights
    as dask arrays to provide lazy / deferred access to the data.

    Parameters
    ----------
    vis : :class:`dask.array.Array` of complex64, shape (*T*, *F*, *B*)
        Complex visibility data as a function of time, frequency and baseline
    flags : :class:`dask.array.Array` of uint8, shape (*T*, *F*, *B*)
        Flags as a function of time, frequency and baseline
    weights : :class:`dask.array.Array` of float32, shape (*T*, *F*, *B*)
        Visibility weights as a function of time, frequency and baseline
    unscaled_weights : :class:`dask.array.Array` of float32, shape (*T*, *F*, *B*)
        Weights that are not scaled by autocorrelations, thereby representing
        the number of voltage samples that constitutes each visibility (optional)
    name : string, optional
        Identifier that describes the origin of the data (backend-specific)
    """

    def __init__(self, vis, flags, weights, unscaled_weights=None):
        if not (vis.shape == flags.shape == weights.shape):
            raise ValueError(f'Shapes of vis {vis.shape}, flags {flags.shape} '
                             f'and weights {weights.shape} differ')
        if unscaled_weights is not None and (unscaled_weights.shape != vis.shape):
            raise ValueError(f'Shapes of unscaled weights {unscaled_weights.shape} '
                             f'and vis {vis.shape} differ')
        self.vis = vis
        self.flags = flags
        self.weights = weights
        self.unscaled_weights = unscaled_weights

    @property
    def shape(self):
        return self.vis.shape


def _default_zero(array):
    if isinstance(array, PlaceholderChunk):
        return np.zeros(array.shape, array.dtype)
    else:
        return array


def _apply_data_lost(orig_flags, lost):
    if not lost:
        return orig_flags
    flags = orig_flags
    for chunk, slices in toolz.partition(2, lost):
        if isinstance(chunk, PlaceholderChunk):
            if flags is orig_flags:
                flags = orig_flags.copy()
            flags[slices] |= DATA_LOST
    return flags


def _narrow(array):
    """Reduce an integer array to the narrowest type that can hold it.

    It is specialised for unsigned types. It will not alter the dtype
    if the array contains negative values.

    If the type is not changed, a view is returned rather than a copy.
    """
    if array.dtype.kind not in ['u', 'i']:
        raise ValueError('Array is not integral')
    if not array.size:
        dtype = np.uint8
    else:
        low = np.min(array)
        high = np.max(array)
        if low < 0:
            dtype = array.dtype
        elif high <= 0xFF:
            dtype = np.uint8
        elif high <= 0xFFFF:
            dtype = np.uint16
        elif high <= 0xFFFFFFFF:
            dtype = np.uint32
        else:
            dtype = array.dtype
    return array.astype(dtype, copy=False)


def corrprod_to_autocorr(corrprods):
    """Find the autocorrelation indices of correlation products.

    Parameters
    ----------
    corrprods : sequence of 2-tuples or ndarray
        Input labels of the correlation products

    Returns
    -------
    auto_indices : np.ndarray
        The indices in corrprods that correspond to auto-correlations
    index1, index2 : np.ndarray
        Lists of the same length as corrprods, containing the indices within
        `auto_indices` referring to the first and second corresponding
        autocorrelations.

    Raises
    ------
    KeyError
        If any of the autocorrelations are missing
    """
    auto_indices = []
    auto_lookup = {}
    for i, baseline in enumerate(corrprods):
        if baseline[0] == baseline[1]:
            auto_lookup[baseline[0]] = len(auto_indices)
            auto_indices.append(i)
    index1 = [auto_lookup[a] for (a, b) in corrprods]
    index2 = [auto_lookup[b] for (a, b) in corrprods]
    return _narrow(np.array(auto_indices)), _narrow(np.array(index1)), _narrow(np.array(index2))


def correct_autocorr_quantisation(vis, corrprods, levels=None):
    """Correct autocorrelations for quantisation effects (Van Vleck correction).

    This is a first-order correction that only adjusts the mean autocorrelations,
    which in turn affects the autocorrelation and crosscorrelation weights.
    A complete correction would also adjust the mean crosscorrelations, and
    further improve the weight estimates based on Bayesian statistics.

    Parameters
    ----------
    vis : :class:`dask.array.Array` of complex64, shape (*T*, *F*, *B*)
        Complex visibility data as function of time, frequency, correlation product
    corrprods : sequence of 2-tuples or ndarray, containing str
        Input labels of the correlation products, used to find autocorrelations
    levels : sequence of float, optional
        Quantisation levels of real/imag components of complex digital signal
        entering correlator (defaults to MeerKAT F-engine output levels)

    Returns
    -------
    corrected_vis : :class:`dask.array.Array` of complex64, shape (*T*, *F*, *B*)
        Complex visibility data with autocorrelations corrected for quantisation
    """
    assert len(corrprods) == vis.shape[2]
    # Ensure that we have only a single chunk on the baseline axis.
    if len(vis.chunks[2]) > 1:
        vis = vis.rechunk({2: vis.shape[2]})
    auto_indices, _, _ = corrprod_to_autocorr(corrprods)
    if levels is None:
        # 255-level "8-bit" output of MeerKAT F-engine requantiser
        levels = np.arange(-127., 128.)
    quantised_autocorr_table, true_autocorr_table = autocorr_lookup_table(levels)

    def _correct_autocorr_quant(vis):
        out = vis.copy()
        out[..., auto_indices] = np.interp(vis[..., auto_indices].real,
                                           quantised_autocorr_table, true_autocorr_table)
        return out

    return da.blockwise(_correct_autocorr_quant, 'ijk', vis, 'ijk', dtype=np.complex64,
                        name='van-vleck-autocorr-' + vis.name)


@numba.jit(nopython=True, nogil=True)
def weight_power_scale(vis, weights, auto_indices, index1, index2, out=None, divide=True):
    """Divide (or multiply) weights by autocorrelations (ndarray version).

    The weight associated with visibility (i,j) is divided (or multiplied) by
    the corresponding real visibilities (i,i) and (j,j).

    This function is designed to be usable with :func:`dask.array.blockwise`.

    Parameters
    ----------
    vis : np.ndarray
        Chunk of visibility data, with dimensions time, frequency, baseline
        (or any two dimensions then baseline). It must contain all the
        baselines of a stream, even though only the autocorrelations are used.
    weights : np.ndarray
        Chunk of weight data, with the same shape as `vis`
    auto_indices, index1, index2 : np.ndarray
        Arrays returned by :func:`corrprod_to_autocorr`
    out : np.ndarray, optional
        If specified, the output array, with same shape as `vis` and
        dtype ``np.float32``
    divide : bool, optional
        True if weights will be divided by autocorrelations, otherwise
        they will be multiplied
    """
    auto_scale = np.empty(len(auto_indices), np.float32)
    out = np.empty(vis.shape, np.float32) if out is None else out
    bad_weight = np.float32(2.0**-32)
    for i in range(vis.shape[0]):
        for j in range(vis.shape[1]):
            for k in range(len(auto_indices)):
                autocorr = vis[i, j, auto_indices[k]].real
                auto_scale[k] = np.reciprocal(autocorr) if divide else autocorr
            for k in range(vis.shape[2]):
                p = auto_scale[index1[k]] * auto_scale[index2[k]]
                # If either or both of the autocorrelations has zero power then
                # there is likely something wrong with the system. Set the
                # weight to very close to zero (not actually zero, since that
                # can cause divide-by-zero problems downstream).
                if not np.isfinite(p):
                    p = bad_weight
                out[i, j, k] = p * weights[i, j, k]
    return out


def _scale_weights(vis, weights, corrprods, divide):
    """Divide (or multiply) weights by autocorrelations (dask array version)."""
    assert len(corrprods) == vis.shape[2]
    # Ensure that we have only a single chunk on the baseline axis.
    if len(vis.chunks[2]) > 1:
        vis = vis.rechunk({2: vis.shape[2]})
    if len(weights.chunks[2]) > 1:
        weights = weights.rechunk({2: weights.shape[2]})
    auto_indices, index1, index2 = corrprod_to_autocorr(corrprods)
    return da.blockwise(weight_power_scale, 'ijk', vis, 'ijk', weights, 'ijk',
                        dtype=np.float32, auto_indices=auto_indices,
                        index1=index1, index2=index2, divide=divide)


class ChunkStoreVisFlagsWeights(VisFlagsWeights):
    """Correlator data stored in a chunk store.

    Parameters
    ----------
    store : :class:`ChunkStore` object
        Chunk store
    chunk_info : dict mapping array name to info dict
        Dict specifying prefix, dtype, shape and chunks per array
    corrprods : sequence of 2-tuples of input labels, optional
        Correlation products. If given, compute both (scaled) `weights` and
        `unscaled_weights` by dividing or multiplying by the autocorrelations
        as needed. If `None`, the stored weights become `weights` and
        `unscaled_weights` is `None`, i.e. disabled (useful for testing).
    stored_weights_are_scaled : bool, optional
        True if the weights in the chunk store are already scaled by
        the autocorrelations. This determines how (scaled) `weights`
        and `unscaled_weights` are obtained from the stored weights.
        Should be True if `corrprods` is `None`.
    van_vleck : {'off', 'autocorr'}, optional
        Type of Van Vleck (quantisation) correction to perform
    index : tuple of slice, optional
        Slice expression to apply to each array before combining them. At the
        moment this can only have two elements (no slicing of baselines),
        because ``weights_channel`` only has time and frequency dimensions.

    Attributes
    ----------
    vis_prefix : string
        Prefix of correlator_data / visibility array, viz. its S3 bucket name
    """

    def __init__(self, store, chunk_info, corrprods=None,
                 stored_weights_are_scaled=True, van_vleck='off', index=()):
        self.store = store
        self.vis_prefix = chunk_info['correlator_data']['prefix']
        darray = {}
        for array, info in chunk_info.items():
            array_name = store.join(info['prefix'], array)
            chunk_args = (array_name, info['chunks'], info['dtype'])
            errors = DATA_LOST if array == 'flags' else 'placeholder'
            darray[array] = store.get_dask_array(*chunk_args, index=index, errors=errors)
        flags_orig_name = darray['flags'].name
        flags_raw_name = store.join(chunk_info['flags']['prefix'], 'flags_raw')
        # Combine original flags with data_lost indicating where values were lost from
        # other arrays.
        lost_map = np.empty([len(c) for c in darray['flags'].chunks], dtype="O")
        for index in np.ndindex(lost_map.shape):
            lost_map[index] = []
        for array_name, array in darray.items():
            if array_name == 'flags':
                continue
            # Source keys may appear multiple times in the array, so to save
            # memory we can pre-create the objects for the keys and reuse them
            # (idea borrowed from dask.array.rechunk).
            src_keys = np.empty([len(c) for c in array.chunks], dtype="O")
            for index in np.ndindex(src_keys.shape):
                src_keys[index] = (array.name,) + index
            # array may have fewer dimensions than flags
            # (specifically, for weights_channel).
            chunks = array.chunks
            if array.ndim < darray['flags'].ndim:
                chunks += tuple((x,) for x in darray['flags'].shape[array.ndim:])
            intersections = intersect_chunks(darray['flags'].chunks, chunks)
            for src_key, pieces in zip(src_keys.flat, intersections):
                for piece in pieces:
                    dst_index, slices = zip(*piece)
                    # if src_key is missing, then the parts of dst_index
                    # indicated by slices must be flagged.
                    # TODO: fast path for when slices covers the whole chunk?
                    lost_map[dst_index].extend([src_key, slices])
        dsk = {
            (flags_raw_name,) + key: (
                _apply_data_lost,
                (flags_orig_name,) + key,
                value
            ) for key, value in np.ndenumerate(lost_map)
        }
        dsk = HighLevelGraph.from_collections(
            flags_raw_name, dsk, dependencies=list(darray.values())
        )
        flags = da.Array(dsk, flags_raw_name,
                         chunks=darray['flags'].chunks,
                         shape=darray['flags'].shape,
                         dtype=darray['flags'].dtype)
        darray['flags'] = flags

        # Turn missing blocks in the other arrays into zeros to make them
        # valid dask arrays.
        for array_name, array in darray.items():
            if array_name == 'flags':
                continue
            new_name = 'filled-' + array.name
            indices = itertools.product(*(range(len(c)) for c in array.chunks))
            dsk = {
                (new_name,) + index: (
                    _default_zero,
                    (array.name,) + index
                ) for index, shape in zip(indices, itertools.product(*array.chunks))
            }
            dsk = HighLevelGraph.from_collections(new_name, dsk, dependencies=[array])
            darray[array_name] = da.Array(dsk, new_name,
                                          chunks=array.chunks,
                                          shape=array.shape,
                                          dtype=array.dtype)

        # Optionally correct visibilities for quantisation effects
        vis = darray['correlator_data']
        if van_vleck == 'autocorr':
            vis = correct_autocorr_quantisation(vis, corrprods)
        elif van_vleck != 'off':
            raise ValueError("The van_vleck parameter should be one of ['off', 'autocorr'], "
                             f"got '{van_vleck}' instead")

        # Combine low-resolution weights and high-resolution weights_channel
        stored_weights = darray['weights'] * darray['weights_channel'][..., np.newaxis]
        # Scale weights according to power (or remove scaling if already applied)
        if corrprods is not None:
            if stored_weights_are_scaled:
                weights = stored_weights
                unscaled_weights = _scale_weights(vis, stored_weights, corrprods, divide=False)
            else:
                weights = _scale_weights(vis, stored_weights, corrprods, divide=True)
                unscaled_weights = stored_weights
        else:
            if not stored_weights_are_scaled:
                raise ValueError('Stored weights are unscaled but no corrprods are provided')
            weights = stored_weights
            # Don't bother with unscaled weights (it's optional)
            unscaled_weights = None
        VisFlagsWeights.__init__(self, vis, flags, weights, unscaled_weights)
