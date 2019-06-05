.. _hdf5_format_v2:

MVF version 2 (KAT-7)
======================

.. sectionauthor:: Simon Ratcliffe <sratcliffe@ska.ac.za>, Ludwig Schwardt <ludwig@ska.ac.za>

Introduction
------------

With the introduction of the KAT-7 correlator, we have taken the opportunity to revisit the correlator data storage format. This document describes this updated format.

Basic Concept
-------------
A single HDF5 corresponds to a single observation (contiguous telescope time segment for a specified subarray).

At highest level split into Data and MetaData.

MetaData contains two distinct types:

 * Configuration is known a priori and is static for the duration of the observation.
 * Sensors contains dynamic information provided in the form of katcp sensors. Typically only full known post observation.

Flags and History are special cases objects that get populated during run time but not from sensors. These are also the only groups that could get updated post augmentation.

Some datasets such as the noise_diode flags are synthesised from sensor information post capture. These base sensors could then be removed if space is a concern.

A major/minor version number is included in the file. The major indicates the overall structural philosophy (this document describes version 2.x). The minor is used
to identify the mandtory members of the MetaData and Markup groups included in the file. This allows addition of members (and modification of existing members) to the required list without wholesale changes to the file structure. The mandatory members are described in the following document: TBA.

If used to store voltage data then both correlator_data and timestamps are omitted as timing is synthesized on the fly.

Nut - number of correlator timeslots in this observation
Nt - number of averaged time timeslots
Nuf - number of correlator frequency channels
Nf - number of averaged frequency channels
Nbl - number of baselines
Np - number of polarisation products
Na - number of antennas in a given subarray
AntennaK - first antenna in a given subarray
AntennaN - last antenna in a given subarray

HDF5 Format
-----------

The structural format is shown below.

Groups are named using CamelCase, datasets are all lower case with underscores.
Attributes are indicated next to a group in {}::

 / {augment_ts}
   {experiment_id}
   {version}
 
 /Data/ {ts_of_first_timeslot}
      /correlator_data - (Nt,Nf,Nbl,2) array of float32 visibilities (real and imag components)
      /timestamps - (Nt) array of float64 timestamps (UT seconds since Unix epoch)
      /voltage_data - (optional) (Na, Nt, Nf) array of 8bit voltage samples
 
 /MetaData/
          /Configuration/
                        /Antennas/ {num_antennas, subarray_id}
                                 /AntennaK..N/ {description, delays, diameter, location, etc...}
                                             / beam_pattern
                                             / h_coupler_noise_diode_model
                                             / h_pin_noise_diode_model
                                             / v_coupler_noide_diode_model
                                             / v_pin_noise_diode_model
                        /Correlator/ {num_channels, center_freq, channel_bw, etc...}
                        /Observation/ {type, pi, contact, sw_build_versions, etc...}
                        /PostProcessing/ {channel_averaging, rfi_threshold, etc...}
                                       /time_averaging - TBD detail of baseline dep time avg
         /Sensors/
                 /Antennas/ {num_antennas, subarray_id}
                          /AntennaK..N/
                                      /... - dataset per antenna and pedestal sensor
                 /DBE/
                        /... - dataset per DBE sensor                
                 /Enviro/
                        /... - dataset per enviro sensor
                 /Other/
                       /... - dataset per other sensor
                 /RFE/
                     /... - dataset per RFE sensor
                 /Source/
                        /phase_center
                        /antenna_target - array of target sensors for each antenna

 /Markup/
        /dropped_data - (optional) describes data dropped by receivers
        /flags - (Nt,Nf,Nbl) post averaged uint8 flags - 1bit per flag, packed
        /flags_description - (Nflags,3) index, name and description for each packed flag type
        /flags_full - (optional) (Nut,Nuf,Nbl) pre-averaged uint8 flags - 1bit per flag, packed
        /labels - (optional) descriptions of intent of each observational phase (e.g. scan, slew, cal, etc..)
        /noise_diode - (Nt,Na) noise diode state during this averaged timeslot
        /noise_diode_full - (optional) (Nut,Na) noise diode state per correlator timeslot
        /weights - (Nt,Nf,Nbl,Nweights) weights for each sample

 /History/
         /augment_log - Log output of augmentation process
         /script_log - Log output of observation script
