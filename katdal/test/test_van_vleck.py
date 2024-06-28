################################################################################
# Copyright (c) 2020-2021, National Research Foundation (SARAO)
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy
# of the License at
#
#   https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

import numpy as np

from katdal.van_vleck import autocorr_lookup_table, norm0_cdf


def test_norm0_cdf():
    scale = 2.0
    x = np.array([0.0, 2.0, 4.0, 6.0])
    # Generated by scipy.stats.norm.cdf(x, scale=2.0)
    expected = np.array([0.5, 0.8413447460685429, 0.9772498680518208, 0.9986501019683699])
    actual = norm0_cdf(x, scale)
    np.testing.assert_allclose(actual, expected, rtol=0., atol=np.finfo(float).eps)
    actual = norm0_cdf(-x, scale)
    np.testing.assert_allclose(actual, 1.0 - expected, rtol=0., atol=np.finfo(float).eps)
    actual = norm0_cdf(x[-1], scale)
    np.testing.assert_allclose(actual, expected[-1], rtol=0., atol=np.finfo(float).eps)


def test_autocorr_correction():
    # 15-level "4-bit" KAT-7 requantiser (contiguous ints for now)
    levels = np.arange(-7., 8.)
    quantised_ac_table, true_ac_table = autocorr_lookup_table(levels)
    N = 100000
    rs = np.random.RandomState(42)
    autocorrs = [0.06, 0.2, 1.0, 10.0, 100.0, 1000.0]
    # Excess above usual sample standard deviation due to loss of information caused by quantisation,
    # generated by Bayesian quantisation correction code (ask the author for details)
    rtol_factors = [2.5, 1.24, 1.16, 1.02, 1.5, 2.9]
    for true_ac, rtol_factor in zip(autocorrs, rtol_factors):
        # Generate complex random voltages with appropriate variance
        scale = np.sqrt(true_ac / 2.)
        x = rs.normal(scale=scale, size=N) + 1j * rs.normal(scale=scale, size=N)
        # Estimate power of the unquantised complex signal as a sanity check
        unquantised_sample_ac = x.dot(x.conj()).real / N
        # The standard deviation of sample variance of N complex normals of variance `var`
        # is var / sqrt(N). Use rtol since stdev is proportional to var and set it to 3 sigma.
        rtol = 3.0 / np.sqrt(N)
        np.testing.assert_allclose(unquantised_sample_ac, true_ac, rtol=rtol)
        # Quantise x to the nearest integer and clip (assumes levels are contiguous ints)
        xq = x.round()
        np.clip(xq.real, levels[0], levels[-1], out=xq.real)
        np.clip(xq.imag, levels[0], levels[-1], out=xq.imag)
        # Estimate power of the quantised signal and correct the effects of quantisation
        quantised_sample_ac = xq.dot(xq.conj()).real / N
        corrected_ac = np.interp(quantised_sample_ac, quantised_ac_table, true_ac_table)
        np.testing.assert_allclose(corrected_ac, true_ac, rtol=rtol_factor * rtol)
        np.testing.assert_allclose(corrected_ac, unquantised_sample_ac, rtol=rtol_factor * rtol)
