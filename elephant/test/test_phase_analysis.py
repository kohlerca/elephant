# -*- coding: utf-8 -*-
"""
Unit tests for the phase analysis module.

:copyright: Copyright 2014-2022 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""
from __future__ import division, print_function

import unittest

import numpy as np
import quantities as pq
import scipy.io
from neo import SpikeTrain, AnalogSignal
from numpy.ma.testutils import assert_allclose

import elephant.phase_analysis
from elephant.datasets import download_datasets


class SpikeTriggeredPhaseTestCase(unittest.TestCase):

    def setUp(self):
        tlen0 = 100 * pq.s
        f0 = 20. * pq.Hz
        fs0 = 1 * pq.ms
        t0 = np.arange(
            0, tlen0.rescale(pq.s).magnitude,
            fs0.rescale(pq.s).magnitude) * pq.s
        self.anasig0 = AnalogSignal(
            np.sin(2 * np.pi * (f0 * t0).simplified.magnitude),
            units=pq.mV, t_start=0 * pq.ms, sampling_period=fs0)
        self.st0 = SpikeTrain(
            np.arange(50, tlen0.rescale(pq.ms).magnitude - 50, 50) * pq.ms,
            t_start=0 * pq.ms, t_stop=tlen0)
        self.st1 = SpikeTrain(
            [100., 100.1, 100.2, 100.3, 100.9, 101.] * pq.ms,
            t_start=0 * pq.ms, t_stop=tlen0)

    def test_perfect_locking_one_spiketrain_one_signal(self):
        phases, amps, times = elephant.phase_analysis.spike_triggered_phase(
            elephant.signal_processing.hilbert(self.anasig0),
            self.st0,
            interpolate=True)

        assert_allclose(phases[0], - np.pi / 2.)
        assert_allclose(amps[0], 1, atol=0.1)
        assert_allclose(times[0].magnitude, self.st0.magnitude)
        self.assertEqual(len(phases[0]), len(self.st0))
        self.assertEqual(len(amps[0]), len(self.st0))
        self.assertEqual(len(times[0]), len(self.st0))

    def test_perfect_locking_many_spiketrains_many_signals(self):
        phases, amps, times = elephant.phase_analysis.spike_triggered_phase(
            [
                elephant.signal_processing.hilbert(self.anasig0),
                elephant.signal_processing.hilbert(self.anasig0)],
            [self.st0, self.st0],
            interpolate=True)

        assert_allclose(phases[0], -np.pi / 2.)
        assert_allclose(amps[0], 1, atol=0.1)
        assert_allclose(times[0].magnitude, self.st0.magnitude)
        self.assertEqual(len(phases[0]), len(self.st0))
        self.assertEqual(len(amps[0]), len(self.st0))
        self.assertEqual(len(times[0]), len(self.st0))

    def test_perfect_locking_one_spiketrains_many_signals(self):
        phases, amps, times = elephant.phase_analysis.spike_triggered_phase(
            [
                elephant.signal_processing.hilbert(self.anasig0),
                elephant.signal_processing.hilbert(self.anasig0)],
            [self.st0],
            interpolate=True)

        assert_allclose(phases[0], -np.pi / 2.)
        assert_allclose(amps[0], 1, atol=0.1)
        assert_allclose(times[0].magnitude, self.st0.magnitude)
        self.assertEqual(len(phases[0]), len(self.st0))
        self.assertEqual(len(amps[0]), len(self.st0))
        self.assertEqual(len(times[0]), len(self.st0))

    def test_perfect_locking_many_spiketrains_one_signal(self):
        phases, amps, times = elephant.phase_analysis.spike_triggered_phase(
            elephant.signal_processing.hilbert(self.anasig0),
            [self.st0, self.st0],
            interpolate=True)

        assert_allclose(phases[0], -np.pi / 2.)
        assert_allclose(amps[0], 1, atol=0.1)
        assert_allclose(times[0].magnitude, self.st0.magnitude)
        self.assertEqual(len(phases[0]), len(self.st0))
        self.assertEqual(len(amps[0]), len(self.st0))
        self.assertEqual(len(times[0]), len(self.st0))

    def test_interpolate(self):
        phases_int, _, _ = elephant.phase_analysis.spike_triggered_phase(
            elephant.signal_processing.hilbert(self.anasig0),
            self.st1,
            interpolate=True)

        self.assertLess(phases_int[0][0], phases_int[0][1])
        self.assertLess(phases_int[0][1], phases_int[0][2])
        self.assertLess(phases_int[0][2], phases_int[0][3])
        self.assertLess(phases_int[0][3], phases_int[0][4])
        self.assertLess(phases_int[0][4], phases_int[0][5])

        phases_noint, _, _ = elephant.phase_analysis.spike_triggered_phase(
            elephant.signal_processing.hilbert(self.anasig0),
            self.st1,
            interpolate=False)

        self.assertEqual(phases_noint[0][0], phases_noint[0][1])
        self.assertEqual(phases_noint[0][1], phases_noint[0][2])
        self.assertEqual(phases_noint[0][2], phases_noint[0][3])
        self.assertEqual(phases_noint[0][3], phases_noint[0][4])
        self.assertNotEqual(phases_noint[0][4], phases_noint[0][5])

        # Verify that when using interpolation and the spike sits on the sample
        # of the Hilbert transform, this is the same result as when not using
        # interpolation with a spike slightly to the right
        self.assertEqual(phases_noint[0][2], phases_int[0][0])
        self.assertEqual(phases_noint[0][4], phases_int[0][0])

    def test_inconsistent_numbers_spiketrains_hilbert(self):
        self.assertRaises(
            ValueError, elephant.phase_analysis.spike_triggered_phase,
            [
                elephant.signal_processing.hilbert(self.anasig0),
                elephant.signal_processing.hilbert(self.anasig0)],
            [self.st0, self.st0, self.st0], False)

        self.assertRaises(
            ValueError, elephant.phase_analysis.spike_triggered_phase,
            [
                elephant.signal_processing.hilbert(self.anasig0),
                elephant.signal_processing.hilbert(self.anasig0)],
            [self.st0, self.st0, self.st0], False)

    def test_spike_earlier_than_hilbert(self):
        # This is a spike clearly outside the bounds
        st = SpikeTrain(
            [-50, 50],
            units='s', t_start=-100 * pq.s, t_stop=100 * pq.s)
        phases_noint, _, _ = elephant.phase_analysis.spike_triggered_phase(
            elephant.signal_processing.hilbert(self.anasig0),
            st,
            interpolate=False)
        self.assertEqual(len(phases_noint[0]), 1)

        # This is a spike right on the border (start of the signal is at 0s,
        # spike sits at t=0s). By definition of intervals in
        # Elephant (left borders inclusive, right borders exclusive), this
        # spike is to be considered.
        st = SpikeTrain(
            [0, 50],
            units='s', t_start=-100 * pq.s, t_stop=100 * pq.s)
        phases_noint, _, _ = elephant.phase_analysis.spike_triggered_phase(
            elephant.signal_processing.hilbert(self.anasig0),
            st,
            interpolate=False)
        self.assertEqual(len(phases_noint[0]), 2)

    def test_spike_later_than_hilbert(self):
        # This is a spike clearly outside the bounds
        st = SpikeTrain(
            [1, 250],
            units='s', t_start=-1 * pq.s, t_stop=300 * pq.s)
        phases_noint, _, _ = elephant.phase_analysis.spike_triggered_phase(
            elephant.signal_processing.hilbert(self.anasig0),
            st,
            interpolate=False)
        self.assertEqual(len(phases_noint[0]), 1)

        # This is a spike right on the border (length of the signal is 100s,
        # spike sits at t=100s). However, by definition of intervals in
        # Elephant (left borders inclusive, right borders exclusive), this
        # spike is not to be considered.
        st = SpikeTrain(
            [1, 100],
            units='s', t_start=-1 * pq.s, t_stop=200 * pq.s)
        phases_noint, _, _ = elephant.phase_analysis.spike_triggered_phase(
            elephant.signal_processing.hilbert(self.anasig0),
            st,
            interpolate=False)
        self.assertEqual(len(phases_noint[0]), 1)

    # This test handles the correct dealing with input signals that have
    # different time units, including a CompoundUnit
    def test_regression_269(self):
        # This is a spike train on a 30KHz sampling, one spike at 1s, one just
        # before the end of the signal
        cu = pq.CompoundUnit("1/30000.*s")
        st = SpikeTrain(
            [30000., (self.anasig0.t_stop - 1 * pq.s).rescale(cu).magnitude],
            units=pq.CompoundUnit("1/30000.*s"),
            t_start=-1 * pq.s, t_stop=300 * pq.s)
        phases_noint, _, _ = elephant.phase_analysis.spike_triggered_phase(
            elephant.signal_processing.hilbert(self.anasig0),
            st,
            interpolate=False)
        self.assertEqual(len(phases_noint[0]), 2)


class WeightedPhaseLagIndexTestCase(unittest.TestCase):
    files_to_download_ground_truth = None
    files_to_download_artificial = None
    files_to_download_real = None

    @classmethod
    def setUpClass(cls):
        np.random.seed(73)

        # The files from G-Node GIN 'elephant-data' repository will be
        # downloaded once into a local temporary directory
        # and then loaded/ read for each test function individually.

        # REAL DATA
        real_data_path = "unittest/phase_analysis/weighted_phase_lag_index/" \
                         "data/wpli_real_data"
        cls.files_to_download_real = (
            ("i140703-001_ch01_slice_TS_ON_to_GO_ON_correct_trials.mat",
             "0e76454c58208cab710e672d04de5168"),
            ("i140703-001_ch02_slice_TS_ON_to_GO_ON_correct_trials.mat",
             "b06059e5222e91eb640caad0aba15b7f"),
            ("i140703-001_cross_spectrum_of_channel_1_and_2_of_slice_"
             "TS_ON_to_GO_ON_corect_trials.mat",
             "2687ef63a4a456971a5dcc621b02e9a9")
        )
        for filename, checksum in cls.files_to_download_real:
            # files will be downloaded to ELEPHANT_TMP_DIR
            cls.tmp_path = download_datasets(
                f"{real_data_path}/{filename}", checksum=checksum)
        # ARTIFICIAL DATA
        artificial_data_path = "unittest/phase_analysis/" \
            "weighted_phase_lag_index/data/wpli_specific_artificial_dataset"
        cls.files_to_download_artificial = (
            ("artificial_LFPs_1.mat", "4b99b15f89c0b9a0eb6fc14e9009436f"),
            ("artificial_LFPs_2.mat", "7144976b5f871fa62f4a831f530deee4"),
        )
        for filename, checksum in cls.files_to_download_artificial:
            # files will be downloaded to ELEPHANT_TMP_DIR
            cls.tmp_path = download_datasets(
                f"{artificial_data_path}/{filename}", checksum=checksum)
        # GROUND TRUTH DATA
        ground_truth_data_path = "unittest/phase_analysis/" \
                        "weighted_phase_lag_index/data/wpli_ground_truth"
        cls.files_to_download_ground_truth = (
            ("ground_truth_WPLI_from_ft_connectivity_wpli_"
             "with_real_LFPs_R2G.csv", "4d9a7b7afab7d107023956077ab11fef"),
            ("ground_truth_WPLI_from_ft_connectivity_wpli_"
             "with_artificial_LFPs.csv", "92988f475333d7badbe06b3f23abe494"),
        )
        for filename, checksum in cls.files_to_download_ground_truth:
            # files will be downloaded into ELEPHANT_TMP_DIR
            cls.tmp_path = download_datasets(
                f"{ground_truth_data_path}/{filename}", checksum=checksum)

    def setUp(self):
        self.tolerance = 1e-15

        # load real/artificial LFP-dataset for ground-truth consistency checks
        # real LFP-dataset
        dataset1_real = scipy.io.loadmat(
            f"{self.tmp_path.parent}/{self.files_to_download_real[0][0]}",
            squeeze_me=True)
        dataset2_real = scipy.io.loadmat(
            f"{self.tmp_path.parent}/{self.files_to_download_real[1][0]}",
            squeeze_me=True)

        # get relevant values
        self.lfps1_real = dataset1_real['lfp_matrix'] * pq.uV
        self.sf1_real = dataset1_real['sf'] * pq.Hz
        self.lfps2_real = dataset2_real['lfp_matrix'] * pq.uV
        self.sf2_real = dataset2_real['sf'] * pq.Hz
        # create AnalogSignals from the real dataset
        self.lfps1_real_AnalogSignal = AnalogSignal(
            signal=self.lfps1_real, sampling_rate=self.sf1_real)
        self.lfps2_real_AnalogSignal = AnalogSignal(
            signal=self.lfps2_real, sampling_rate=self.sf2_real)

        # artificial LFP-dataset
        dataset1_artificial = scipy.io.loadmat(
            f"{self.tmp_path.parent}/"
            f"{self.files_to_download_artificial[0][0]}", squeeze_me=True)
        dataset2_artificial = scipy.io.loadmat(
            f"{self.tmp_path.parent}/"
            f"{self.files_to_download_artificial[1][0]}", squeeze_me=True)
        # get relevant values
        self.lfps1_artificial = dataset1_artificial['lfp_matrix'] * pq.uV
        self.sf1_artificial = dataset1_artificial['sf'] * pq.Hz
        self.lfps2_artificial = dataset2_artificial['lfp_matrix'] * pq.uV
        self.sf2_artificial = dataset2_artificial['sf'] * pq.Hz
        # create AnalogSignals from the artificial dataset
        self.lfps1_artificial_AnalogSignal = AnalogSignal(
            signal=self.lfps1_artificial, sampling_rate=self.sf1_artificial)
        self.lfps2_artificial_AnalogSignal = AnalogSignal(
            signal=self.lfps2_artificial, sampling_rate=self.sf2_artificial)

        # load ground-truth reference calculated by:
        # Matlab package 'FieldTrip': ft_connectivity_wpli()
        self.wpli_ground_truth_ft_connectivity_wpli_real = np.loadtxt(
            f"{self.tmp_path.parent}/"
            f"{self.files_to_download_ground_truth[0][0]}",
            delimiter=',', dtype=np.float64)
        self.wpli_ground_truth_ft_connectivity_artificial = np.loadtxt(
            f"{self.tmp_path.parent}/"
            f"{self.files_to_download_ground_truth[1][0]}",
            delimiter=',', dtype=np.float64)

    def test_WPLI_ground_truth_consistency_real_LFP_dataset(self):
        """
        Test if the WPLI is consistent with the reference implementation
        ft_connectivity_wpli() of the MATLAB-package FieldTrip using
        LFP-dataset cuttings from the multielectrode-grasp  G-Node GIN
        repository, which can be found here:
        https://doi.gin.g-node.org/10.12751/g-node.f83565/
        The cutting was performed with this python-script:
        multielectrode_grasp_i140703-001_cutting_script_TS_ON_to_GO_ON.py
        which is available on https://gin.g-node.org/INM-6/elephant-data
        in folder validation/phase_analysis/weighted_phase_lag_index/scripts,
        where also the MATLAB-script for ground-truth generation is located.
        """
        # Quantity-input
        with self.subTest(msg="Quantity input"):
            freq, wpli = elephant.phase_analysis.weighted_phase_lag_index(
                self.lfps1_real, self.lfps2_real, self.sf1_real)
            np.testing.assert_allclose(
                wpli, self.wpli_ground_truth_ft_connectivity_wpli_real,
                atol=self.tolerance, rtol=self.tolerance, equal_nan=True)
        # np.array-input
        with self.subTest(msg="np.array input"):
            freq, wpli = elephant.phase_analysis.weighted_phase_lag_index(
                self.lfps1_real.magnitude, self.lfps2_real.magnitude,
                self.sf1_real)
            np.testing.assert_allclose(
                wpli, self.wpli_ground_truth_ft_connectivity_wpli_real,
                atol=self.tolerance, rtol=self.tolerance, equal_nan=True)
        # neo.AnalogSignal-input
        with self.subTest(msg="neo.AnalogSignal input"):
            freq, wpli = elephant.phase_analysis.weighted_phase_lag_index(
                self.lfps1_real_AnalogSignal, self.lfps2_real_AnalogSignal)
            np.testing.assert_allclose(
                wpli, self.wpli_ground_truth_ft_connectivity_wpli_real,
                atol=self.tolerance, rtol=self.tolerance, equal_nan=True)

    def test_WPLI_ground_truth_consistency_artificial_LFP_dataset(self):
        """
        Test if the WPLI is consistent with the ground truth generated with
        multi-sine artificial LFP-datasets.
        The generation was performed with this python-script:
        generate_artificial_datasets_for_ground_truth_of_wpli.py
        which is available on https://gin.g-node.org/INM-6/elephant-data
        in folder validation/phase_analysis/weighted_phase_lag_index/scripts,
        where also the MATLAB-script for ground-truth generation is located.
        """
        # Quantity-input
        with self.subTest(msg="Quantity input"):
            freq, wpli = elephant.phase_analysis.weighted_phase_lag_index(
                self.lfps1_artificial, self.lfps2_artificial,
                self.sf1_artificial, absolute_value=False)
            np.testing.assert_allclose(
                wpli, self.wpli_ground_truth_ft_connectivity_artificial,
                atol=1e-14, rtol=1e-12, equal_nan=True)
        # np.array-input
        with self.subTest(msg="np.array input"):
            freq, wpli = elephant.phase_analysis.weighted_phase_lag_index(
                self.lfps1_artificial.magnitude,
                self.lfps2_artificial.magnitude, self.sf1_artificial,
                absolute_value=False)
            np.testing.assert_allclose(
                wpli, self.wpli_ground_truth_ft_connectivity_artificial,
                atol=1e-14, rtol=1e-12, equal_nan=True)
        # neo.AnalogSignal-input
        with self.subTest(msg="neo.AnalogSignal input"):
            freq, wpli = elephant.phase_analysis.weighted_phase_lag_index(
                self.lfps1_artificial_AnalogSignal,
                self.lfps2_artificial_AnalogSignal, absolute_value=False)
            np.testing.assert_allclose(
                wpli, self.wpli_ground_truth_ft_connectivity_artificial,
                atol=1e-14, rtol=1e-12, equal_nan=True)

    def test_WPLI_is_zero(self):
        """
        Test if WPLI is close to zero at frequency f=70Hz for the multi-sine
        artificial LFP dataset. White noise prevents arbitrary approximation.
        """
        # Quantity-input
        with self.subTest(msg="Quantity input"):
            freq, wpli = elephant.phase_analysis.weighted_phase_lag_index(
                self.lfps1_artificial, self.lfps2_artificial,
                self.sf1_artificial, absolute_value=False)
            np.testing.assert_allclose(
                wpli[freq == 70], 0, atol=0.004, rtol=self.tolerance)
        # np.array-input
        with self.subTest(msg="np.array input"):
            freq, wpli = elephant.phase_analysis.weighted_phase_lag_index(
                self.lfps1_artificial.magnitude,
                self.lfps2_artificial.magnitude, self.sf1_artificial,
                absolute_value=False)
            np.testing.assert_allclose(
                wpli[freq == 70], 0, atol=0.004, rtol=self.tolerance)
        # neo.AnalogSignal-input
        with self.subTest(msg="neo.AnalogSignal input"):
            freq, wpli = elephant.phase_analysis.weighted_phase_lag_index(
                self.lfps1_artificial_AnalogSignal,
                self.lfps2_artificial_AnalogSignal, absolute_value=False)
            np.testing.assert_allclose(
                wpli[freq == 70], 0, atol=0.004, rtol=self.tolerance)

    def test_WPLI_is_one(self):
        """
        Test if WPLI is one at frequency f=16Hz and 36Hz for the multi-sine
        artificial LFP dataset.
        """
        # Quantity-input
        with self.subTest(msg="Quantity input"):
            freq, wpli = elephant.phase_analysis.weighted_phase_lag_index(
                self.lfps1_artificial, self.lfps2_artificial,
                self.sf1_artificial, absolute_value=False)
            mask = ((freq == 16) | (freq == 36))
            np.testing.assert_allclose(
                wpli[mask], 1, atol=self.tolerance, rtol=self.tolerance)
        # np.array-input
        with self.subTest(msg="np.array input"):
            freq, wpli = elephant.phase_analysis.weighted_phase_lag_index(
                self.lfps1_artificial.magnitude,
                self.lfps2_artificial.magnitude, self.sf1_artificial,
                absolute_value=False)
            mask = ((freq == 16) | (freq == 36))
            np.testing.assert_allclose(
                wpli[mask], 1, atol=self.tolerance, rtol=self.tolerance)
        # neo.AnalogSignal-input
        with self.subTest(msg="neo.AnalogSignal input"):
            freq, wpli = elephant.phase_analysis.weighted_phase_lag_index(
                self.lfps1_artificial_AnalogSignal,
                self.lfps2_artificial_AnalogSignal, absolute_value=False)
            mask = ((freq == 16) | (freq == 36))
            np.testing.assert_allclose(
                wpli[mask], 1, atol=self.tolerance, rtol=self.tolerance)

    def test_WPLI_is_minus_one(self):
        """
        Test if WPLI is minus one at frequency f=52Hz and 100Hz
        for the multi-sine artificial LFP dataset.
        """
        # Quantity-input
        with self.subTest(msg="Quantity input"):
            freq, wpli = elephant.phase_analysis.weighted_phase_lag_index(
                self.lfps1_artificial, self.lfps2_artificial,
                self.sf1_artificial, absolute_value=False)
            mask = ((freq == 52) | (freq == 100))
            np.testing.assert_allclose(
                wpli[mask], -1, atol=self.tolerance, rtol=self.tolerance)
        # np.array-input
        with self.subTest(msg="np.array input"):
            freq, wpli = elephant.phase_analysis.weighted_phase_lag_index(
                self.lfps1_artificial.magnitude,
                self.lfps2_artificial.magnitude, self.sf1_artificial,
                absolute_value=False)
            np.testing.assert_allclose(
                wpli[mask], -1, atol=self.tolerance, rtol=self.tolerance)
        # neo.AnalogSignal-input
        with self.subTest(msg="neo.AnalogSignal input"):
            freq, wpli = elephant.phase_analysis.weighted_phase_lag_index(
                self.lfps1_artificial_AnalogSignal,
                self.lfps2_artificial_AnalogSignal, absolute_value=False)
            np.testing.assert_allclose(
                wpli[mask], -1, atol=self.tolerance, rtol=self.tolerance)

    def test_WPLI_raises_error_if_signals_have_different_shapes(self):
        """
        Test if WPLI raises a ValueError, when the signals have different
        number of trails or different trial lengths.
        """
        # simple samples of different shapes to assert ErrorRaising
        trials2_length3 = np.array([[0, -1, 1], [0, -1, 1]]) * pq.uV
        trials1_length3 = np.array([[0, -1, 1]]) * pq.uV
        trials1_length4 = np.array([[0, 1, 1 / 2, -1]]) * pq.uV
        sampling_frequency = 250 * pq.Hz
        trials2_length3_analogsignal = AnalogSignal(
            signal=trials2_length3, sampling_rate=sampling_frequency)
        trials1_length3_analogsignal = AnalogSignal(
            signal=trials1_length3, sampling_rate=sampling_frequency)
        trials1_length4_analogsignal = AnalogSignal(
            signal=trials1_length4, sampling_rate=sampling_frequency)

        # different numbers of trails
        with self.subTest(msg="diff. trial numbers & Quantity input"):
            np.testing.assert_raises(
                ValueError, elephant.phase_analysis.weighted_phase_lag_index,
                trials2_length3, trials1_length3, sampling_frequency)
        with self.subTest(msg="diff. trial numbers & np.array input"):
            np.testing.assert_raises(
                ValueError, elephant.phase_analysis.weighted_phase_lag_index,
                trials2_length3.magnitude, trials1_length3.magnitude,
                sampling_frequency)
        with self.subTest(msg="diff. trial numbers & neo.AnalogSignal input"):
            np.testing.assert_raises(
                ValueError, elephant.phase_analysis.weighted_phase_lag_index,
                trials2_length3_analogsignal, trials1_length3_analogsignal)
        # different lengths in a trail pair
        with self.subTest(msg="diff. trial lengths & Quantity input"):
            np.testing.assert_raises(
                ValueError, elephant.phase_analysis.weighted_phase_lag_index,
                trials1_length3, trials1_length4, sampling_frequency)
        with self.subTest(msg="diff. trial lengths & np.array input"):
            np.testing.assert_raises(
                ValueError, elephant.phase_analysis.weighted_phase_lag_index,
                trials1_length3.magnitude, trials1_length4.magnitude,
                sampling_frequency)
        with self.subTest(msg="diff. trial lengths & neo.AnalogSignal input"):
            np.testing.assert_raises(
                ValueError, elephant.phase_analysis.weighted_phase_lag_index,
                trials1_length3_analogsignal, trials1_length4_analogsignal)

    @staticmethod
    def test_WPLI_raises_error_if_AnalogSignals_have_diff_sampling_rate():
        """
        Test if WPLI raises a ValueError, when the AnalogSignals have different
        sampling rates.
        """
        signal_x_250_hz = AnalogSignal(signal=np.random.random([40, 2100]),
                                       units=pq.mV, sampling_rate=0.25*pq.kHz)
        signal_y_1000_hz = AnalogSignal(signal=np.random.random([40, 2100]),
                                        units=pq.mV, sampling_rate=1000*pq.Hz)
        np.testing.assert_raises(
            ValueError, elephant.phase_analysis.weighted_phase_lag_index,
            signal_x_250_hz, signal_y_1000_hz)

    def test_WPLI_raises_error_if_sampling_rate_not_given(self):
        """
        Test if WPLI raises a ValueError, when the sampling rate is not given
        for np.array() or Quanitity input.
        """
        signal_x = np.random.random([40, 2100]) * pq.mV
        signal_y = np.random.random([40, 2100]) * pq.mV
        with self.subTest(msg="Quantity-input"):
            np.testing.assert_raises(
                ValueError, elephant.phase_analysis.weighted_phase_lag_index,
                signal_x, signal_y)
        with self.subTest(msg="np.array-input"):
            np.testing.assert_raises(
                ValueError, elephant.phase_analysis.weighted_phase_lag_index,
                signal_x.magnitude, signal_y.magnitude)


if __name__ == '__main__':
    unittest.main()
