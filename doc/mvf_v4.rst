MVF version 4 (MeerKAT)
=======================

The version 4 format is the standard format for MeerKAT visibility data.
Unlike previous versions, the data for an observation does not reside in
a single HDF5 file, as such files would be unmanageably large. Instead,
the data is split into chunks, each in its own file, which are loaded
from disk or the network on demand. For this reason, the term "data set"
is preferred over "file".

Concepts
--------

Streams
    A *stream* is a collection of data and any associated metadata, whether
    multicast, queriable (e.g., sensors) or stored on disk. Every stream in
    a subarray product has a unique name and a type. A stream may
    consist of multiple items of related data e.g., visibilities, flags
    and weights may form a single stream.
Subarray product
    A collection of streams in the MeerKAT Science Data Processor (SD)
    forms a *subarray product*.
Capture block
    A capture block is a contiguous period over which data is captured from
    a specific subarray product. A subarray product can only capture one
    capture block at a time.

    Each visibility belongs to a specific stream and capture block
    within a subarray product.

    Capture block IDs are currently numbers representing the start time
    in the UNIX epoch, but they should be treated as opaque strings.
Chunk store
    A location (such as a local disk or the MeerKAT archive) that stores
    the data from a capture block.

Metadata
--------

The metadata for a data set is stored in a `Redis`_ dump file
(extension ``.rdb``), which is exported by
:doc:`katsdptelstate <katsdptelstate:index>`. Refer to
:doc:`katsdptelstate <katsdptelstate:index>` for details of how
attributes and sensors are encoded in the Redis database.

.. _Redis: http://redis.io/

A single ``.rdb`` file contains metadata for a single subarray but
potentially for multiple streams and capture blocks. The default capture
block and stream to access are stored in ``capture_block_id`` and
``stream_name``.

Keys are stored in one of the following namespaces:

- the global namespace
- :samp:`{stream_name}` (the "stream namespace")
- :samp:`{capture_block_id}` (the "capture-block namespace")
- :samp:`{capture_block_id}.{stream_name}` (the "capture-stream namespace")

Here ``.`` is used to indicate a sub-namespace, but the actual separator
is subject to change and one should always use
:meth:`~.katsdptelstate.telescope_state.TelescopeState.join` to
construct compound namespace names.

Keys may move between these namespaces without notice. Readers should
search for keys from the most specific to the least specific appropriate
namespace (see for example
:meth:`katdal.datasources.view_capture_stream`).

Where keys contain strings, they might contain either raw bytes (which
should be decoded as UTF-8) or Unicode text. Readers should be prepared
to accept either. The goal is to eventually migrate all such fields to
use text.

:doc:`katsdptelstate <katsdptelstate:index>` stores two types of values:
immutable "attributes", and "sensors" which are a list of timestamped
values. In the documentation below, most keys contain attributes, and
sensors are indicated.

Global metadata
^^^^^^^^^^^^^^^

A subset of the sensors in the MeerKAT system are stored in the file, in
the global namespace. Documenting the MeerKAT sensors is beyond the
scope of this documentation.

The following keys are also stored.

``sdp_config`` (dict)
    The JSON object used to configure the SDP subarray product. It is
    not intended to be parsed (the relevant information is intended to
    be available via other means), but it contains a wealth of
    debugging information about the streams and connections between
    them.

``sdp_capture_block_id`` (string) — sensor
    All capture block IDs seen so far. This should not be confused with
    ``capture_block_id``, which indicates the default capture block ID
    that should be consulted when the file is opened without specifying
    a capture block ID.

``sdp_image_tag`` (string)
    The Docker image tag for the Docker images forming the realtime SDP
    capture system. This is the closest thing to a "version number" for
    the implementation.

``sdp_image_overrides`` (dict)
    Alternative Docker image tags for specific services within SDP,
    overriding ``sdp_image_tag``. Overriding individual images is a
    debugging tool and it should *always* be empty for science
    observations.

``config.*`` (dict)
    Command-line options passed to each of the services within SDP.

``sdp_task_details`` (dict)
    Debug information about each of the services launched for the
    subarray product, including the host on which it ran and the Mesos
    TaskInfo structure.


Common stream metadata
^^^^^^^^^^^^^^^^^^^^^^
The list of streams that can be accessed from the archive is available
in ``sdp_archived_streams`` (in the global namespace). Within each
stream, the following keys may be defined (not all make sense for
every stream type).

Only ``stream_type`` and ``src_streams`` are guaranteed to be in the
stream namespace i.e., independent of the capture block. The others may
appear either in the capture-stream namespace or the stream namespace.

``inherit`` (string)
    If present, it indicates another stream from which this stream
    inherits properties. Any property that cannot be found in the
    namespace of the current stream should first be looked up in that
    stream's namespace.

    This is typically used where a single multicast stream is recorded
    in multiple places. Each copy inherits the majority of metadata from
    the original and overrides a few keys.

``stream_type`` (string)
    Valid values are

    ``sdp.vis``
        Uncalibrated visibilities, flags and weights
    ``sdp.flags``
        Similar to ``sdp.vis``, but containing only flags
    ``sdp.cal``
        Calibration solutions. Older files may contain a ``cal`` stream
        which omits the stream information and which does not appear in
        ``sdp_archived_streams``, so that should be considered as a
        fallback.
    ``sdp.continuum_image``
        Continuum image (as a list of CLEAN components) and
        self-calibration solutions. FITS files will be stored in the
        MeerKAT archive but katdal does not currently support accessing
        them.
    ``sdp.spectral_image``
        Spectral-line image. FITS files will be stored in the
        MeerKAT archive but katdal does not currently support accessing
        them.

``src_streams`` (list of string)
    The streams from which the current stream was computed. These are
    not necessarily listed in ``sdp_archived_streams``, particularly if
    they were produced by the MeerKAT Correlator/Beamformer (CBF) rather
    than the SDP.

``n_chans`` (int)
    Number of channels in a channelised product

``n_chans_per_substream`` (int)
    Number of channels in each SPEAD heap. Not relevant when loading
    archived data.

``bandwidth`` (float, Hz)
    Bandwidth of the stream.

``center_freq`` (float, Hz)
    Middle of the central channel. Note that if the number of channels
    is even, this is actually half a channel higher than the middle of
    the band.

``channel_range`` (int, int)
    A half-open range of channels taken from the source stream. The
    length of this range might not equal ``n_chans`` due to channel
    averaging.

Visibility stream metadata
^^^^^^^^^^^^^^^^^^^^^^^^^^

The following are relevant to ``sdp.vis`` and ``sdp.flags`` streams.

``n_bls`` (int)
    Number of baselines. Note that a baseline is a correlation between
    two polarised inputs (a single entry in a Jones matrix).

``bls_ordering`` (2D array)
    An array of pairs of strings. Each pair names two antenna inputs
    that form a baseline. There will be ``n_bls`` rows. Note that this
    can be either a list of 2-element lists or a numpy array.

``sync_time``, ``int_time``, ``first_timestamp`` (float)
    Refer to :ref:`timestamps` below.

``excise`` (bool)
    True if RFI detected in the source stream is excised during
    time and channel averaging. If missing, assume it is true.

``calibrations_applied`` (list of string)
    Names of ``sdp.cal`` streams whose corrections have been applied to
    the data.

``need_weights_power_scale`` (bool)
    Refer to :ref:`weights` below. If missing, assume it is false.

``s3_endpoint_url`` (string), ``chunk_info``
    Refer to :ref:`data` below.

Calibration solutions
^^^^^^^^^^^^^^^^^^^^^

Streams of type ``sdp.cal`` have the following keys.

``antlist`` (list of string)
    List of antenna names. Arrays of calibration solutions use this
    order along the antenna axis.

``pol_ordering`` (list of string)
    List of polarisations (from ``v`` and ``h``). Arrays of calibration
    solutions use this order along the polarisation axis.

``bls_ordering`` (2D array)
    Same meaning as for ``sdp.vis`` streams, but describes the internal
    ordering used within the calibration pipeline and not of much use to
    users.

``param_*``
    Parameters used to configure the calibration.

``product_G`` (2D array) — sensor
    Phase gain solutions, indexed by antenna and polarisation.

``product_K`` (2D array) — sensor
    Delay solutions (in seconds?), indexed by antenna and polarisation. To
    correct data at frequency :math:`\nu`, multiply it by
    :math:`e^{-2\pi i\cdot K\cdot \nu}`.

``product_B_parts`` (int)
    Number of keys across which bandpass solutions are split.

:samp:`product_B{N}` (3D array) — sensor
    Bandpass solutions, indexed by channel, antenna and polarisation.

    For implementation reasons, the bandpass solutions are split across
    multiple keys. *N* is in the range [0, ``product_B_parts``), and
    these pieces should be concatenated along the channel (first) axis
    to reconstruct the full solution. If some pieces are missing (which
    is rare but can occur), they should be assumed to have the same
    shape as the present pieces.

``product_KCROSS_DIODE`` (array) — sensor
    TODO

``product_BCROSS_DIODE`` (array) — sensor
    TODO

``refant`` (string)
    `katpoint`_ description of the selected reference antenna (whose
    name will also appear in ``antlist``). The reference antenna is only
    chosen when first needed in the capture block, so this key may be
    absent if there was no calibration.

:samp:`shared_solve_*N*`, :samp:`last_dump_index*N*`
    These are used for internal communication between the calibration
    processes, and are not intended for external use.

Some common points to note that about the solutions:

- Solutions describe the systematic errors. To correct data, it must be divided
  by the solutions.

- The key will only be present if at least one solution was computed.

- The timestamp associated with each sensor value is the timestamp of
  the middle of the data that was used to compute the solution.

- Solutions may contain NaN values, which indicates that there was
  insufficient information to compute a solution (for example, because
  all the data was flagged).

- Solutions are only valid as long as the system gain controls (TODO:
  name for these) are not altered. It is thus not generally advisable to
  re-use gains from one capture block to correct data from another
  capture block.

Image stream metadata
^^^^^^^^^^^^^^^^^^^^^

The following apply to ``sdp.continuum_image`` and ``sdp.spectral_image``
streams.

``target_list`` (dict)
    This is only applicable for imaging streams. Each key is a
    `katpoint`_ target description and the value is the *normalised target
    name*, which is a string used to form target-specific sub-namespaces of the
    stream and capture-stream namespaces. A normalised target name looks
    similar to the target name but has a limited character set (suitable for
    forming filenames and telstate namespaces) and, where necessary, a sequence
    number appended to ensure uniqueness.

.. _katpoint: https://github.com/ska-sa/katpoint

For each ``sdp.continuum_image`` stream, there is a sub-namespace per target
(named with the normalised target name) with the following keys (keeping
in mind that ``.`` is used to indicate whichever separator is in use by
katsdptelstate for this database):

``target0.clean_components`` (dict)
    Image of the target field as a set of point sources. The ``target0``
    sub-namespace is used to allow for possible alternative ways to run
    the continuum imager in which a single execution would image
    multiple fields, in which case there would be :samp:`target{N}`
    sub-namespaces up to some *N*. This is not currently expected for
    MeerKAT science observations.

    The dictionary has two keys:

    ``description`` (string)
        `katpoint`_ description of the target field (specifically, the
        phase centre).

    ``components`` (list of string)
        `katpoint_` target descriptions for the CLEAN components. The
        names are arbitrary. This describes the **perceived** sky i.e., are
        modulated by the primary beam.


:samp:`m{NNN}.gains` (array)
    TODO: is this the final form for the selfcal solutions?

.. _linking-streams:

.. _timestamps:

Timestamps
^^^^^^^^^^

Timestamps are not stored explicitly. Instead, the first timestamp and
the interval between dumps are stored, from which timestamps can be
synthesised. The ith dump has a central timestamp (in the UNIX epoch) of
:math:`\text{sync_time} + \text{first_timestamp} + i \times
\text{int_time}`. The split of the initial timestamp into two parts is
for technical reasons.

There is also ``first_timestamp_adc``, which is the same as
``first_timestamp`` but in units of the digitiser ADC counts. It is
stored only for internal implementation reasons and should not be relied
upon.

.. _data:

Data
----

Visibilities, flags and weights are subdivided into small *chunks*. The
chunking model is based on `dask`_. Visibilities are treated as a 3D
array, with axes for time, frequency and baseline. The data is divided
into pieces along each axis. Each piece is stored in a separate file
in the archive, in `.npy format`_. The metadata necessary to reconstruct
the array is stored in the telescope state and documented in more detail
later. It is possible that some chunks will be missing, because they
were lost during the capture process. On load, katdal will replace such
chunks with default values and set the ``data_lost`` flag for them.
Weights and flags are similarly treated.

.. _dask: http://docs.dask.org/en/latest/

.. _.npy format: https://docs.scipy.org/doc/numpy-1.14.0/neps/npy-format.html

Chunks are named :samp:`{type}/{AAAAA}_{BBBBB}_{CCCCC}.npy` where *type*
is one of ``correlator_data`` (visibilities), ``flags``, ``weights``;
and *AAAAA*, *BBBBB* and *CCCCC* are the (zero-based) indices of the
first element in the chunk along each axis. Additionally, there are
chunks named :samp:`weights_channel/{AAAAA}_{BBBBB}.npy`, explained
below.

Note that the chunking scheme typically differs between visibilities,
flags and weights, so files with the same base name do not necessarily
refer to the same point in time or frequency.

All the data for one stream is located in a single chunk store. If it is
in the MeerKAT archive, the URL to the base of this chunk store
(implementing the S3 protocol) is stored in ``s3_endpoint_url``.
Capture-stream specific information is stored in ``chunk_info``, a
two-level dictionary. The outer key is the *type* listed above, and the
inner key is one of:

``prefix`` (string)
    A path prefix for the data. In the case of S3, this is the bucket
    name. For local storage, it is a directory name (the parent of the
    :samp:`{type}` directory).
``dtype`` (string)
    Numpy dtype of the data, which is expected to match the dtype
    encoded in the individual chunk files.
``shape`` (tuple)
    Shape of the virtual dask array obtained by joining together all the
    chunks.
``chunks`` (tuple of tuples)
    Sizes of the chunks along each axis, in the format used by dask.


.. _weights:

Weights
^^^^^^^
To save space, the weights are represented in an indirect form that
requires some calculation to reconstruct. The actual weight for a
visibility is the product of three values:

- The value in the ``weights`` chunk.
- A baseline-independent value in the ``weights_channel`` chunk.
- If the stream has a ``need_weights_power_scale`` key in telstate and
  the value is true, the inverse of the product of the autocorrelation
  power for the two inputs in the baseline.

Flags
^^^^^
Each flag is a bitfield. The meaning of the individual bits is
documented in the :mod:`katdal.flags` module. Note that it is possible
that a flag chunk is present but the corresponding visibility or weight
data is missing, in which case it is the reader's responsibility to set
the ``data_lost`` bit.

The MeerKAT Science Data Processor typically uses two levels of
flagging: a conservative first-pass flagger run directly on the
correlator output, and a more accurate flagger that operates on
data that has been averaged and (in some cases) calibrated. The latter
appears in a stream of type ``sdp.flags``, which contains only flags. It
can be linked to the corresponding visibilities and weights by checking
its :ref:`source streams <linking-streams>`. The flags in this stream are a
superset of the flags in the originating stream and are guaranteed to
have the same timestamp and frequency metadata, so can be used in place
of the original flags. However, due to data loss it is possible that
the replacement flags will have slightly more or fewer dumps at the end,
which will need to be handled.
