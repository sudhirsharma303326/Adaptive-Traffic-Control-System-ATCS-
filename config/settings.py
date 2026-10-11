"""
=============================================================================
ATCS Configuration & System Settings
=============================================================================
Defines system-wide constants, intersection geometry, ROI polygons,
virtual stopline coordinates, vehicle spatial weights, acoustic siren parameters,
and carbon emission calculation factors.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import os

# Base Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
EVIDENCE_DIR = os.path.join(BASE_DIR, "evidence_vault")
DB_PATH = os.path.join(DATA_DIR, "atcs_traffic.db")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(EVIDENCE_DIR, exist_ok=True)


@dataclass
class IntersectionLaneConfig:
    lane_id: str
    name: str
    direction: str  # Northbound, Southbound, Eastbound, Westbound
    # Virtual stopline: ((x1, y1), (x2, y2)) in normalized or 640x480 coordinate space
    stopline: Tuple[Tuple[int, int], Tuple[int, int]]
    # Lane ROI polygon: list of (x, y) defining the approach zone
    roi_polygon: List[Tuple[int, int]]
    # Real-world physical area covered by ROI in square meters (m^2)
    roi_physical_area_m2: float = 120.0
    # Road calibration: estimated pixel-to-meter conversion factor (pixels per meter)
    pixels_per_meter: float = 12.0


# Standard 4-Way Intersection Default Coordinates (based on 640x480 frame)
DEFAULT_LANES: Dict[str, IntersectionLaneConfig] = {
    "LANE_N": IntersectionLaneConfig(
        lane_id="LANE_N",
        name="North Approach",
        direction="Southbound (from North)",
        stopline=((180, 360), (460, 360)),
        roi_polygon=[(160, 100), (480, 100), (520, 440), (120, 440)],
        roi_physical_area_m2=150.0,
        pixels_per_meter=10.0,
    ),
    "LANE_S": IntersectionLaneConfig(
        lane_id="LANE_S",
        name="South Approach",
        direction="Northbound (from South)",
        stopline=((180, 360), (460, 360)),
        roi_polygon=[(160, 100), (480, 100), (520, 440), (120, 440)],
        roi_physical_area_m2=150.0,
        pixels_per_meter=10.0,
    ),
    "LANE_E": IntersectionLaneConfig(
        lane_id="LANE_E",
        name="East Approach",
        direction="Westbound (from East)",
        stopline=((180, 360), (460, 360)),
        roi_polygon=[(160, 100), (480, 100), (520, 440), (120, 440)],
        roi_physical_area_m2=150.0,
        pixels_per_meter=10.0,
    ),
    "LANE_W": IntersectionLaneConfig(
        lane_id="LANE_W",
        name="West Approach",
        direction="Eastbound (from West)",
        stopline=((180, 360), (460, 360)),
        roi_polygon=[(160, 100), (480, 100), (520, 440), (120, 440)],
        roi_physical_area_m2=150.0,
        pixels_per_meter=10.0,
    ),
}


# Vehicle Space Occupancy Physical Footprints (in m^2)
VEHICLE_PHYSICAL_FOOTPRINT_M2 = {
    "motorcycle": 2.5,   # ~2.0m x 1.2m
    "bicycle": 1.5,      # ~1.8m x 0.8m
    "car": 8.5,          # ~4.5m x 1.9m
    "van": 12.0,         # ~5.2m x 2.2m
    "bus": 28.0,         # ~11.5m x 2.5m
    "truck": 32.0,       # ~12.5m x 2.6m
    "emergency": 14.0,   # Ambulance / Police / Fire
    "default": 8.0,
}

# Passenger Car Unit (PCU) Equivalency Factors
PCU_FACTORS = {
    "motorcycle": 0.5,
    "bicycle": 0.2,
    "car": 1.0,
    "van": 1.5,
    "bus": 3.0,
    "truck": 3.0,
    "emergency": 2.0,
    "default": 1.0,
}


@dataclass
class AdaptiveSignalConfig:
    # Timing Limits in seconds
    min_green_seconds: float = 10.0
    max_green_seconds: float = 65.0
    yellow_seconds: float = 3.5
    all_red_seconds: float = 2.0
    
    # Priority & Thresholds
    high_density_threshold_percent: float = 70.0  # Level 3 trigger (>70% spatial occupancy)
    starvation_max_wait_seconds: float = 110.0    # Anti-starvation deadlock protection
    
    # Emergency Vehicle Preemption (EVP)
    evp_clearance_hold_seconds: float = 12.0      # Minimum hold once emergency detected
    evp_all_conflicting_red: bool = True


@dataclass
class SirenAcousticConfig:
    sample_rate: int = 22050                     # Standard audio sampling rate
    frame_size_samples: int = 4096                # FFT window size
    target_band_hz: Tuple[float, float] = (1000.0, 3000.0)  # Standard siren band
    snr_detection_threshold_db: float = 8.5       # SNR threshold over noise floor
    harmonic_peak_ratio_threshold: float = 2.2    # Peak spectral power vs median power


@dataclass
class CarbonModelConfig:
    # Idle Fuel Consumption Rates (Liters per hour of continuous idle)
    fuel_rate_car_lph: float = 1.15
    fuel_rate_bike_lph: float = 0.25
    fuel_rate_bus_truck_lph: float = 3.20
    
    # Emission conversion factor: kg of CO2 produced per Liter of fuel burned
    co2_kg_per_liter: float = 2.42
    
    # Economic cost of fuel per Liter (e.g. INR or USD baseline)
    fuel_cost_per_liter: float = 1.25  # In USD (approx $1.25/L)
    
    # Fixed-Time Baseline cycle time (seconds) for comparison
    static_fixed_green_seconds: float = 45.0
    
    # Carbon offset tree equivalent: ~21.77 kg of CO2 absorbed per urban tree per year
    co2_kg_per_tree_year: float = 21.77


# Master Settings Instance
SIGNAL_CFG = AdaptiveSignalConfig()
SIREN_CFG = SirenAcousticConfig()
CARBON_CFG = CarbonModelConfig()
