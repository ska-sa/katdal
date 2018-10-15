History
=======

0.11 (2018-10-15)
-----------------
* Python 3 support via python-future (finally!)
* Load L1 flags if available (#164)
* Reduced memory usage (#165) and speedups (#155, #169, #170, #171, #182)
* S3 chunk store now uses requests directly instead of via botocore (#166)
* Let lazy indexer use oindex semantics like in the past (#180)
* Fix concatenated data sets (#161)
* Fix IPython / Jupyter tab completion for sensor cache (#176)

0.10.1 (2018-05-18)
-------------------
* Restore NumPy 1.14 support (all data flagged otherwise)

0.10 (2018-05-17)
-----------------
* Rally around the MeerKAT Visibility Format (MVF)
* First optimised converter from MVF v4 to MS: mvftoms
* Latest v4 fixes (synthetic timestamps, autodetection, NPY files in Ceph)
* Flag and zero missing chunks
* Now requires katsdptelstate (released), dask, h5py 2.3 and Python 2.7
* Restore S3 unit tests and NumPy 1.11 (on Ubuntu 16.04) support

0.9.5 (2018-02-22)
------------------
* New HDF5 v3.9 file format in anticipation of v4 (affects obs_params)
* Fix receiver serial numbers in recent MeerKAT data sets
* Add dask support to ChunkStore
* katdal.open() works on v4 RDB files

0.9 (2018-01-16)
----------------
* New ChunkStore and telstate-based parser for future v4 format
* Use python-casacore (>=2.2.1) to create Measurement Sets instead of blank.ms
* Read new-style noise diode sensor names, serial numbers and L0 stream metadata
* Select multiple polarisations (useful for cross-pol)
* Relax the "expected number of dumps" check to avoid spurious warnings
* Fix NumPy 1.14 warnings

0.8 (2017-08-08)
----------------
* Fix upside-down MeerKAT images
* SensorData rework to load gain solutions and access telstate efficiently
* Improve mapping of sensor events onto dumps, especially for long (8 s) dumps
* Fix NumPy 1.13 warnings and errors
* Support UHF receivers

0.7.1 (2017-01-19)
------------------

* Fix MODEL_DATA / CORRECTED_DATA shapes in h5toms
* Produce calibration solution tables in h5toms and improve error messages
* Autodetect receiver band on older RTS files

0.7 (2016-12-14)
----------------

* Support weights in file and improve vis / weights / flags API
* Support multiple receivers and improve centre frequency extraction
* Speed up h5toms by ordering visibilities by time
* Fix band selection and corr products for latest SDP (cam2telstate)
* Allow explicit MS names in h5toms

0.6 (2016-09-16)
----------------

* Initial release of katdal
