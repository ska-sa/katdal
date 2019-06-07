Sign conventions
----------------

Visibilities
============

For a wave with frequency :math:`\omega` and wave number :math:`k`, the
phasor is

.. math:: e^{(\omega t - kz)i}

Visibilities are then :math:`e_1 \overline{e_2}`.

In KAT-7, the opposite sign convention is used in the HDF5 files, but katdal
conjugates the visibilities to match MeerKAT.

Baseline coordinates
====================

The UVW coordinates for the baseline (A, B) are
:math:`(u, v, w)_A - (u, v, w)_B`. Combined with the above, this means
that ideal visibilities (ignoring any effects apart from geometric
delay) are

.. math:: V(u, v, w) = \int \frac{I(l, m)}{n} e^{2\pi i(ul + vm + w(n - 1))}\ dl\ dm

Polarisation
============

KAT-7 and MeerKAT are linear feed systems. On MeerKAT, if one points
one's right thumb in the direction of vertical polarisation and the
right index finger in the direction of horizontal polarisation, then the
right middle finger points from the antenna towards the source.

When exporting to a Measurement Set, katdal maps H to (IEEE) x and V to
y, and introduces a 90° offset to the parallactic angle rotation.

KAT-7 has the opposite convention for polarisation (due to the lack of a
sub-reflector). katdal does **not** make any effort to compensate for
this. Measurement sets exported from KAT-7 data should thus not be used
for polarimetry without further correction.
