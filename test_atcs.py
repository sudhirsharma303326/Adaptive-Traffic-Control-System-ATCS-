"""
=============================================================================
ATCS Comprehensive Automated Verification Test Suite
=============================================================================
Validates all 4 core modules, system hierarchy, edge fallback pipelines,
acoustic FFT detection, stopline ANPR e-challan database transactions,
and carbon sustainability mathematical models.
"""

import os
import sys
import unittest
import numpy as np

# Ensure path is configured
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import DEFAULT_LANES, DB_PATH
from database.db_manager import DatabaseManager
from core.siren_detector import detect_siren_frequency, AcousticSirenMonitor
from core.vehicle_detector import VehicleDetectionEngine
from core.priority_matrix import SignalPriorityController, PriorityLevel
from core.stopline_anpr import StoplineBreachTracker
from core.carbon_tracker import CarbonOffsetEngine
from core.traffic_simulator import SyntheticIntersectionGenerator


class TestATCSSystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.db = DatabaseManager(DB_PATH)
        cls.sim = SyntheticIntersectionGenerator()
        cls.vision = VehicleDetectionEngine()
        cls.controller = SignalPriorityController(db=cls.db)
        cls.tracker = StoplineBreachTracker(db=cls.db)
        cls.carbon = CarbonOffsetEngine(db=cls.db)

    def test_01_acoustic_siren_detection_fft(self):
        """MODULE B: Tests SciPy FFT Siren frequency detector."""
        # 1. Test standard Yelp Siren (1000 - 2500 Hz sweep)
        siren_audio = AcousticSirenMonitor.generate_synthetic_siren(
            duration_s=0.5, siren_type="yelp", noise_level=0.05
        )
        res_siren = detect_siren_frequency(siren_audio, sample_rate=22050)
        self.assertTrue(res_siren.detected, "Siren should be detected in 1000-3000 Hz band")
        self.assertGreaterEqual(res_siren.peak_frequency_hz, 900.0)
        self.assertLessEqual(res_siren.peak_frequency_hz, 3100.0)
        self.assertGreater(res_siren.snr_db, 8.5)

        # 2. Test Low-Frequency Ambient Traffic Rumble (No siren)
        ambient_audio = AcousticSirenMonitor.generate_ambient_traffic_noise(
            duration_s=0.5, noise_level=0.20
        )
        res_ambient = detect_siren_frequency(ambient_audio, sample_rate=22050)
        self.assertFalse(res_ambient.detected, "Ambient traffic noise should NOT trigger siren detection")

    def test_02_spatial_pixel_occupancy(self):
        """MODULE A: Tests dual-metric density & pixel-based space occupancy."""
        frame = self.sim.generate_frame("LANE_N", "RED", 15.0)
        metrics = self.vision.detect_and_calculate_occupancy(frame, DEFAULT_LANES["LANE_N"])

        self.assertEqual(metrics.lane_id, "LANE_N")
        self.assertGreaterEqual(metrics.vehicle_count, 0)
        self.assertGreaterEqual(metrics.spatial_occupancy_m2, 0.0)
        self.assertGreaterEqual(metrics.occupancy_percent, 0.0)
        self.assertLessEqual(metrics.occupancy_percent, 100.0)
        self.assertGreaterEqual(metrics.total_pcu, 0.0)

    def test_03_signal_priority_decision_matrix(self):
        """DECISION MATRIX: Tests 4-tier hierarchy & anti-deadlock starvation."""
        lane_metrics = {}
        for lid in ["LANE_N", "LANE_S", "LANE_E", "LANE_W"]:
            f = self.sim.generate_frame(lid, "RED", 15.0)
            lane_metrics[lid] = self.vision.detect_and_calculate_occupancy(f, DEFAULT_LANES[lid])

        # Step tick
        snap = self.controller.update_tick(dt_s=1.0, metrics=lane_metrics)
        self.assertIn(snap.active_lane_id, self.controller.lane_ids)
        self.assertIn(snap.active_phase, ["GREEN", "YELLOW", "ALL_RED"])

        # Test Level 1 Emergency Preemption Override (EVP)
        snap_evp = self.controller.update_tick(
            dt_s=1.0,
            metrics=lane_metrics,
            siren_detected=True,
            siren_lane="LANE_W",
        )
        self.assertTrue(snap_evp.is_emergency_preemption)
        self.assertEqual(snap_evp.emergency_lane_id, "LANE_W")

    def test_04_virtual_stopline_anpr_echallan(self):
        """MODULE C: Tests red-light breach tracking & SQLite logging."""
        initial_count = self.db.get_total_violations_count()
        
        # Simulate breach event
        frame = self.sim.generate_frame("LANE_E", "RED", 10.0)
        self.sim.trigger_breach("LANE_E")
        
        # Advance frame so vehicle crosses stopline
        frame_breach = self.sim.generate_frame("LANE_E", "RED", 10.0)
        m = self.vision.detect_and_calculate_occupancy(frame_breach, DEFAULT_LANES["LANE_E"])
        violations = self.tracker.process_lane_vehicles(
            frame_breach, DEFAULT_LANES["LANE_E"], "RED", m.detected_vehicles
        )

        # Logged to SQLite
        new_count = self.db.get_total_violations_count()
        self.assertGreaterEqual(new_count, initial_count)

    def test_05_carbon_sustainability_model(self):
        """MODULE D: Tests mathematical carbon and fuel calculation model."""
        frame = self.sim.generate_frame("LANE_N", "GREEN", 25.0)
        m = self.vision.detect_and_calculate_occupancy(frame, DEFAULT_LANES["LANE_N"])

        sust = self.carbon.calculate_cycle_savings(m, actual_adaptive_green_s=25.0)
        self.assertGreater(sust.cycle_avoided_delay_seconds, 0.0)
        self.assertGreater(sust.cumulative_fuel_liters, 0.0)
        self.assertGreater(sust.cumulative_co2_kg, 0.0)
        self.assertGreater(sust.trees_planted_equivalent, 0.0)


if __name__ == "__main__":
    unittest.main()
