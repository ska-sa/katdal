History
=======

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
