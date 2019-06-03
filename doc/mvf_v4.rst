MVF version 4 (MeerKAT)
=======================

The version 4 format is the standard format for MeerKAT visibility data.
Unlike previous versions, the data for an observation does not reside in
a single HDF5 file, as such files would be unmanageably large. Instead,
the data is split into chunks, each in its own file, which are loaded
from disk or the network on demand. For this reason, the term "dataset"
is preferred over "file".

The metadata for a dataset is stored in a `redis`_ dump file
(extension .rdb), which is exported by
:doc:`katsdptelstate <katsdptelstate:index>`. Refer to
:doc:`katsdptelstate <katsdptelstate:index>` for details of how
attributes and sensors are encoded in the redis database.

.. _redis: http://redis.io/

Further details are still to be written.
