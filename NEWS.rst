History
=======

0.23 (2024-06-28)
-----------------
* New `mvf_download` script (also promote `mvf_copy` and remove junk) (#380)
* Select targets by their tags (#377)
* Rename `np.product` to support numpy >= 2.0 and make unit tests more robust (#372)

0.22 (2023-11-28)
-----------------
* Restore np.bool in Numba averaging function to prevent mvftoms crash (#370)
* Replace underscores with dashes when loading old buckets from RDBs (#370)
* Select multiple targets with same name to avoid dropped scans in MS (#369)
* Support on-the-fly (OTF) scans in mvftoms (#366)

0.21 (2023-05-12)
-----------------
* Fix support for numpy >= 1.24 and move unit tests from nose to pytest (#361)
* Complete rewrite of S3ChunkStore retries for more robust archive downloads (#363)
* Remove IMAGING_WEIGHT column full of zeroes from MS (#356)
* Improve tests with ES256-encoded JWT tokens and more robust MinIO health check (#360)

0.20.1 (2022-04-29)
-------------------
* Fix broken `dataset.vis[n]` due to DaskLazyIndexer / ChunkStore interaction (#355)

0.20 (2022-04-14)
-----------------
* Fix support for dask >= 2022.01.1 in ChunkStore (#351)
* Allow mvftoms to continue with partial MS after an interruption (#348)
* New mvf_copy.py script that can be used to extract autocorrelations only (#349)
* Treat Ceph 403 errors properly in S3ChunkStore (#352)

0.19 (2021-11-23)
-----------------
* Support scans and non-radec targets like planets in mvftoms (#333)
* Expose the raw flags of MVF4 datasets (#335)
* Expose CBF F-engine sensors: applied delays, phases and gains (#338)
* Verify that S3 bucket is not empty to detect datasets archived to tape (#344)
* Populate SIGMA_SPECTRUM and redo SIGMA and WEIGHT in mvftoms (#347)
* Have a sensible DataSet.name and also add a separate DataSet.url (#337)
* Allow deselection of antennas using '~m0XX' (#340)
* Allow nested DaskLazyIndexers (#336)
* Fix mvftoms on macOS and Python 3.8+ (#339)

0.18 (2021-04-20)
-----------------
* Switch to PyJWT 2 and Python 3.6, cleaning up Python 2 relics (#321 - #323)
* Allow preselection of channels and dumps upon katdal.open() to save time and memory (#324)
* Allow user to select fields, scans and antennas in mvftoms (#269)
* Support h5py 3.0 string handling in MVF3 (#331)
* Refactor requirement files to remove recursive dependencies (#329)

0.17 (2021-01-27)
-----------------
* This is the last release that will support Python 3.5
* Pin PyJWT version to 1.x to avoid breaking API changes (#320)
* Van Vleck correction! (autocorrelations only, though) (#316)
* Expose excision, aka raw weights (#308)
* Better unit testing of DataSource and S3ChunkStore in general (#319)
* Support indexed telstate keys (the 1000th cut that killed Python 2) (#304)
* Split out separate utility classes for Minio (#310)
* Fix filtering of sensor events with invalid status (#306)

0.16 (2020-08-28)
-----------------
* This is the last release that will support Python 2 (python2 maintenance branch)
* New 'time_offset' sensor property that adjusts timestamps of any sensor (#307)
* Fix calculation of cbf_dump_period for 'wide' / 'narrowN' instruments (#301)
* Increase katstore search window by 600 seconds to find infrequent updates (#302)
* Refactor SensorData to become a lazy abstract interface without caching (#292)
* Refactor SensorCache to use MutableMapping (#300)
* Fix rx_serial sensor use and file mode warning in MVFv3 files (#298, #299)

0.15 (2020-03-13)
-----------------
* Improve S3 chunk store: check tokens, improve timeouts and retries (#272 - #277)
* Retry truncated reads and 50x errors due to S3 server overload (#274)
* Apply flux calibration if available (#278, #279)
* Improve mvf_rechunk and mvf_read_benchmark scripts (#280, #281, #284)
* Fix selection by target description (#271)
* Mark Python 2 support as deprecated (#282)

0.14 (2019-10-02)
-----------------
* Make L2 product by applying self-calibration corrections (#253 - #256)
* Speed up uvw calculations (#252, #262)
* Produce documentation on readthedocs.org (#244, #245, #247, #250, #261)
* Clean up mvftoms and fix REST_FREQUENCY in SOURCE sub-table (#258)
* Support katstore64 API (#265)
* Improve chunk store: detect short reads, speed up handling of lost data (#259, #260)
* Use katpoint 0.9 and dask 1.2.1 features (#262, #243)

0.13 (2019-05-09)
-----------------
* Load RDB files straight from archive (#233, #241)
* Retrieve raw sensor data from CAM katstore (#234)
* Work around one-CBF-dump offset issue (#238)
* Improved MS output: fixed RECEPTOR_ANGLE (#230), added WEIGHT_SPECTRUM (#231)
* Various optimisations to applycal (#224), weights (#226), S3 reads (#229)
* Use katsdptelstate 0.8 and dask 1.1 features (#228, #233, #240)

0.12 (2019-02-12)
-----------------
* Optionally make L1 product by applying calibration corrections (#194 - #198)
* Let default reference antenna in v4 datasets be "array" antenna (#202, #220)
* Use katsdptelstate v0.7: generic encodings, memory backend (#196, #201, #212)
* Prepare for multi-dump chunks (#213, #214, #216, #217, #219)
* Allow L1 flags to be ignored (#209, #210)
* Deal with deprecated dask features (#204, #215)
* Remove RADOS chunk store (it's all via S3 from here on)

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
