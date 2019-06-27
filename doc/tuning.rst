Tuning your application
=======================
It is possible to load data at high bandwidth using katdal: rates over
2.5 GB/s have been seen when loading from a local disk. However, it
requires an understanding of the storage layout and choice of an
appropriate access pattern.

This chapter is aimed at loading :doc:`mvf_v4` data, as older versions
typically contain far less data. Some of the advice is generic but some
of the methods described here will not work on older data sets.

Chunking
--------
The most important thing to understand is that the data is split into
chunks, each of which are stored as a file on disk or an object in an S3
store. Retrieving any element of a chunk causes the entire chunk to be
retrieved. Thus, aligning accesses to whole chunks will give the best
performance, as data is not discarded.

As an illustration, consider an application that has an outer loop over
the baselines, and loads data for one baseline at a time. Chunks
typically span all baselines, so each time one baseline is loaded,
katdal will actually load the entire data set. If the application can
be redesigned to fetch data for a small time range for all baselines it
will perform much better.

When using MVFv4, katdal uses `dask`_ to manage the chunking. After
opening a data set, you can determine the chunking for a particular
array by examining its ``dataset`` member:

.. code:: python

   >>> d.vis.dataset
   dask.array<1556179171-sdp, shape=(38, 4096, 40), dtype=complex64, chunksize=(32, 1024, 40)>
   >>> d.vis.dataset.chunks
   ((32, 6), (1024, 1024, 1024, 1024), (40,))

.. _dask: https://docs.dask.org/

For this data set, it will be optimal to load visibilities in 32 × 1024
× 40 element pieces.

Note that the chunking scheme may be different for visibilities, flags
and weights.

Joint loading
-------------
The values returned by katdal are not the raw values stored in the
chunks: there is processing involved, such as application of calibration
solutions and flagging of missing data. Some of this processing is
common between visibilities, flags and weights. It's thus more efficient
to load the visibilities, flags and weights as a single operation rather
than as three separate operations.

This can be achieved using :meth:`.DaskLazyIndexer.get`. For example,
replace

.. code:: python

   vis = d.vis[idx]
   flags = d.flags[idx]
   weights = d.weights[idx]

with

.. code:: python

   vis, flags, weights = DaskLazyIndexer.get([d.vis, d.flags, d.weights], idx)

Parallelism
-----------
Dask uses multiple worker threads. It defaults to one thread per CPU
core, but for I/O-bound tasks this is often not enough to achieve
maximum throughput. Refer to the dask `scheduler`_ documentation for
details of how to configure the number of workers.

.. _scheduler: https://docs.dask.org/en/latest/scheduling.html

More workers only helps if there is enough parallel work to be
performed, which means there need to be at least as many chunks loaded
at a time as there are workers (and preferably many more). It's thus
advisable to load as much data at a time as possible without running out
of memory.

Selection
---------
Using :meth:`DataSet.select` is relatively expensive. For the best
performance, it should only be used occasionally (for example, to filter
out unwanted data at the start), with array access notation or
:meth:`.DaskLazyIndexer.get` used to break up large data sets into
manageable pieces.

Dask also performs better with selections that select contiguous data.
You might be able to get a little more performance by using
:meth:`.DataSet.scans` (which will yield a series of contiguous
selections) rather than using :meth:`~.DataSet.select` with
``scans='track'``.

Network versus local disk
-------------------------
When loading data from the network, latency is typically higher, and so
more workers will be needed to achieve peak throughput. Network access
is also more sensitive to access patterns that are mis-aligned with
chunks, because chunks are not cached in memory by the operation system
and hence must be re-fetched over the network if they are accessed
again.

Benchmarking
------------
To assist with testing out the effects of changing these tuning
parameters, the katdal source code includes a script called
``mvf_read_benchmark.py`` that allows a data set to be loaded in
various ways and reports the average throughput. The command-line
options are somewhat limited so you may need to edit it yourself, for
example, to add a custom selection.
