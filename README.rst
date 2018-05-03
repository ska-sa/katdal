katdal
======

This package serves as a data access library to interact with the chunk stores
and HDF5 files produced by the MeerKAT radio telescope and its predecessors
(KAT-7 and Fringe Finder). It uses memory carefully, allowing data sets to be
inspected and partially loaded into memory. Data sets may be concatenated and
split via a flexible selection mechanism. In addition, it provides a script to
convert these data sets to CASA MeasurementSets.

Quick Tutorial
--------------

Open any data set through a single function to obtain a data set object:

.. code:: python

  import katdal
  d = katdal.open('1234567890.h5')

This automatically determines the version and storage location of the data set.
The versions roughly map to the various instruments::

  - v1 : Fringe Finder (HDF5 file)
  - v2 : KAT-7 (HDF5 file)
  - v3 : MeerKAT (HDF5 file)
  - v4 : MeerKAT (chunk store based on objects in Ceph)

Multiple data sets (even of different versions) may also be concatenated
together (as long as they have the same dump rate):

.. code:: python

  d = katdal.open(['1234567890.h5', '1234567891.h5'])

Inspect the contents of the data set by printing the object:

.. code:: python

  print d

Here is a typical output::

  ===============================================================================
  Name: 1313067732.h5 (version 2.0)
  ===============================================================================
  Observer: someone  Experiment ID: 2118d346-c41a-11e0-b2df-a4badb44fe9f
  Description: 'Track on Hyd A,Vir A, 3C 286 and 3C 273'
  Observed from 2011-08-11 15:02:14.072 SAST to 2011-08-11 15:19:47.810 SAST
  Dump rate: 1.00025 Hz
  Subarrays: 1
  ID  Antennas                            Inputs  Corrprods
   0  ant1,ant2,ant3,ant4,ant5,ant6,ant7  14      112
  Spectral Windows: 1
  ID  CentreFreq(MHz)  Bandwidth(MHz)  Channels  ChannelWidth(kHz)
   0  1822.000         400.000          1024      390.625
  -------------------------------------------------------------------------------
  Data selected according to the following criteria:
  subarray=0
  ants=['ant1', 'ant2', 'ant3', 'ant4', 'ant5', 'ant6', 'ant7']
  spw=0
  -------------------------------------------------------------------------------
  Shape: (1054 dumps, 1024 channels, 112 correlation products) => Size: 967.049 MB
  Antennas: *ant1,ant2,ant3,ant4,ant5,ant6,ant7  Inputs: 14  Autocorr: yes  Crosscorr: yes
  Channels: 1024 (index 0 - 1023, 2021.805 MHz - 1622.195 MHz), each 390.625 kHz wide
  Targets: 4 selected out of 4 in catalogue
  ID  Name    Type      RA(J2000)     DEC(J2000)  Tags  Dumps  ModelFlux(Jy)
   0  Hyd A   radec      9:18:05.28  -12:05:48.9          333      33.63
   1  Vir A   radec     12:30:49.42   12:23:28.0          251     166.50
   2  3C 286  radec     13:31:08.29   30:30:33.0          230      12.97
   3  3C 273  radec     12:29:06.70    2:03:08.6          240      39.96
  Scans: 8 selected out of 8 total       Compscans: 1 selected out of 1 total
  Date        Timerange(UTC)       ScanState  CompScanLabel  Dumps  Target
  11-Aug-2011/13:02:14 - 13:04:26    0:slew     0:             133    0:Hyd A
              13:04:27 - 13:07:46    1:track    0:             200    0:Hyd A
              13:07:47 - 13:08:37    2:slew     0:              51    1:Vir A
              13:08:38 - 13:11:57    3:track    0:             200    1:Vir A
              13:11:58 - 13:12:27    4:slew     0:              30    2:3C 286
              13:12:28 - 13:15:47    5:track    0:             200    2:3C 286
              13:15:48 - 13:16:27    6:slew     0:              40    3:3C 273
              13:16:28 - 13:19:47    7:track    0:             200    3:3C 273

The first segment of the printout displays the static information of the data
set, including observer, dump rate and all the available subarrays and spectral
windows in the data set. The second segment (between the dashed lines) highlights
the active selection criteria. The last segment displays dynamic information
that is influenced by the selection, including the overall visibility array
shape, antennas, channel frequencies, targets and scan info.

The data set is built around the concept of a three-dimensional visibility array
with dimensions of time, frequency and correlation product. This is reflected in
the *shape* of the dataset:

.. code:: python

  d.shape

which returns ``(1054, 1024, 112)``, meaning 1054 dumps by 1024 channels by 112
correlation products.

Let's select a subset of the data set:

.. code:: python

  d.select(scans='track', channels=slice(200,300), ants='ant4')
  print d

This results in the following printout::

  ===============================================================================
  Name: /Users/schwardt/Downloads/1313067732.h5 (version 2.0)
  ===============================================================================
  Observer: siphelele  Experiment ID: 2118d346-c41a-11e0-b2df-a4badb44fe9f
  Description: 'track on Hyd A,Vir A, 3C 286 and 3C 273 for Lud'
  Observed from 2011-08-11 15:02:14.072 SAST to 2011-08-11 15:19:47.810 SAST
  Dump rate: 1.00025 Hz
  Subarrays: 1
  ID  Antennas                            Inputs  Corrprods
   0  ant1,ant2,ant3,ant4,ant5,ant6,ant7  14      112
  Spectral Windows: 1
  ID  CentreFreq(MHz)  Bandwidth(MHz)  Channels  ChannelWidth(kHz)
   0  1822.000         400.000          1024      390.625
  -------------------------------------------------------------------------------
  Data selected according to the following criteria:
  channels=slice(200, 300, None)
  subarray=0
  scans='track'
  ants='ant4'
  spw=0
  -------------------------------------------------------------------------------
  Shape: (800 dumps, 100 channels, 4 correlation products) => Size: 2.560 MB
  Antennas: ant4  Inputs: 2  Autocorr: yes  Crosscorr: no
  Channels: 100 (index 200 - 299, 1943.680 MHz - 1905.008 MHz), each 390.625 kHz wide
  Targets: 4 selected out of 4 in catalogue
  ID  Name    Type      RA(J2000)     DEC(J2000)  Tags  Dumps  ModelFlux(Jy)
   0  Hyd A   radec      9:18:05.28  -12:05:48.9          200      31.83
   1  Vir A   radec     12:30:49.42   12:23:28.0          200     159.06
   2  3C 286  radec     13:31:08.29   30:30:33.0          200      12.61
   3  3C 273  radec     12:29:06.70    2:03:08.6          200      39.32
  Scans: 4 selected out of 8 total       Compscans: 1 selected out of 1 total
  Date        Timerange(UTC)       ScanState  CompScanLabel  Dumps  Target
  11-Aug-2011/13:04:27 - 13:07:46    1:track    0:             200    0:Hyd A
              13:08:38 - 13:11:57    3:track    0:             200    1:Vir A
              13:12:28 - 13:15:47    5:track    0:             200    2:3C 286
              13:16:28 - 13:19:47    7:track    0:             200    3:3C 273

Compared to the first printout, the static information has remained the same
while the dynamic information now reflects the selected subset. There are many
possible selection criteria, as illustrated below:

.. code:: python

  d.select(timerange=('2011-08-11 13:10:00', '2011-08-11 13:15:00'), targets=[1, 2])
  d.select(spw=0, subarray=0)
  d.select(ants='ant1,ant2', pol='H', scans=(0,1,2), freqrange=(1700e6, 1800e6))

See the docstring of ``DataSet.select`` for more detailed information (i.e.
do ``d.select?`` in IPython). Take note that only one subarray and one spectral
window must be selected.

Once a subset of the data has been selected, you can access the data and
timestamps on the data set object:

.. code:: python

  vis = d.vis[:]
  timestamps = d.timestamps[:]

Note the ``[:]`` indexing, as the ``vis`` and ``timestamps`` properties are
special ``LazyIndexer`` objects that only give you the actual data when
you use indexing, in order not to inadvertently load the entire array into memory.

For the example dataset and no selection the ``vis`` array will have a shape of
``(1054, 1024, 112)``. The time dimension is labelled by ``d.timestamps``, the
frequency dimension by ``d.channel_freqs`` and the correlation product dimension
by ``d.corr_products``.

Another key concept in the data set object is that of *sensors*. These are named
time series of arbritrary data that are either loaded from the data set
(*actual* sensors) or calculated on the fly (*virtual* sensors). Both variants
are accessed through the *sensor cache* (available as ``d.sensor``) and cached
there after the first access. The data set object also provides convenient
properties to expose commonly-used sensors, as shown in the plot example below:

.. code:: python

  import matplotlib.pyplot as plt
  plt.plot(d.az, d.el, 'o')
  plt.xlabel('Azimuth (degrees)')
  plt.ylabel('Elevation (degrees)')

Other useful attributes include ``ra``, ``dec``, ``lst``, ``mjd``, ``u``,
``v``, ``w``, ``target_x`` and ``target_y``. These are all one-dimensional
NumPy arrays that dynamically change length depending on the active selection.

As in katdal's predecessor (scape) there is a ``DataSet.scans`` generator
that allows you to step through the scans in the data set. It returns the
scan index, scan state and target object on each iteration, and updates
the active selection on the data set to include only the current scan.
It is also possible to iterate through the compound scans with the
``DataSet.compscans`` generator, which yields the compound scan index, label
and first target on each iteration for convenience. These two iterators may also
be used together to traverse the data set structure:

.. code:: python

  for compscan, label, target in d.compscans():
      plt.figure()
      for scan, state, target in d.scans():
          if state in ('scan', 'track'):
              plt.plot(d.ra, d.dec, 'o')
      plt.xlabel('Right ascension (J2000 degrees)')
      plt.ylabel('Declination (J2000 degrees)')
      plt.title(target.name)

Finally, all the targets (or fields) in the data set are stored in a catalogue
available at ``d.catalogue``, and the original HDF5 file is still accessible via
a back door installed at ``d.file`` in the case of a single-file data set.
