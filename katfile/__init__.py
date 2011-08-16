"""Karoo Array Telescope library to interact with HDF5 and MeasurementSet files."""

from katpoint import is_iterable

from .simplevisdata import SimpleVisData, WrongVersion
from .concatvisdata import ConcatVisData
from .hdf5 import H5DataV1, H5DataV2

formats = [H5DataV2, H5DataV1]

#--------------------------------------------------------------------------------------------------
#--- FUNCTION :  open
#--------------------------------------------------------------------------------------------------

def open(filename, ref_ant='', channel_range=None, time_offset=0.0, **kwargs):
    """Open data file(s) with loader of the appropriate version.

    Parameters
    ----------
    filename : string or sequence of strings
        Data file name or list of file names
    ref_ant : string, optional
        Name of reference antenna (default is first antenna in use)
    channel_range : sequence of 2 ints, optional
        Index of first and last frequency channel to load (defaults to all)
    time_offset : float, optional
        Offset to add to all timestamps, in seconds
    kwargs : dict
        Extra parameters are passed on to underlying accessor class

    Returns
    -------
    data : object of subclass of :class:`SimpleVisData`
        Object providing :class:`SimpleVisData` interface to file

    """
    for format in formats:
        try:
            return ConcatVisData(format, filename, ref_ant, channel_range, time_offset, **kwargs) \
                   if is_iterable(filename) else format(filename, ref_ant, channel_range, time_offset, **kwargs)
        except WrongVersion:
            continue
    else:
        raise WrongVersion("Unknown data file format or version")
