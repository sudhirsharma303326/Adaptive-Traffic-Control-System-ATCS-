"""
=============================================================================
MODULE B & SYSTEM HIERARCHY: Priority Decision Matrix & Adaptive Signal Controller
=============================================================================
Enforces strict 4-Tier Hierarchical Decision Matrix to eliminate deadlock:
- Level 1: Emergency Vehicles (Ambulance / Fire Brigade / Police) -> Instant EVP Preemption
- Level 2: VIP Convoys / High-Priority Transit Corridors
- Level 3: High-Density Lanes (>70% Spatial Occupancy)
- Level 4: Standard Regular Traffic Flow

Features:
- Dynamic Green Light Duration Allocation based on spatial density & PCU
- Smart Green Wave Corridor with Instant Override & Conflicting Lane Lockout
- Anti-Starvation Deadlock Protection (Guaranteed maximum wait limit)
- Smooth Yellow Transition (3.5s) and All-Red Clearance Interval (2.0s)
"""

from enum import IntEnum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import time

from config.settings import SIGNAL_CFG, AdaptiveSignalConfig
from core.vehicle_detector import LaneOccupancyMetrics
from database.db_manager import DatabaseManager


class PriorityLevel(IntEnum):
    LEVEL_1_EMERGENCY = 1
    LEVEL_2_VIP = 2
    LEVEL_3_HIGH_DENSITY = 3
    LEVEL_4_STANDARD = 4


@dataclass
class LaneSignalState:
    lane_id: str
    signal_color: str  # "RED", "YELLOW", "GREEN"
    time_remaining_s: float
    current_phase_duration_s: float
    wait_time_s: float
    priority_level: PriorityLevel
    is_evp_active: bool = False


@dataclass
class IntersectionStateSnapshot:
    timestamp: float
    active_lane_id: str
    active_phase: str  # "GREEN", "YELLOW", "ALL_RED"
    lane_states: Dict[str, LaneSignalState]
    is_emergency_preemption: bool
    emergency_lane_id: Optional[str]
    preemption_source: str  # "VISUAL_YOLO", "ACOUSTIC_FFT", "MANUAL", "NONE"


class SignalPriorityController:
    """
    Master 4-Tier Decision Matrix and Adaptive Signal Phase Manager.
    Arbitrates right-of-way across 4-way intersection lanes without deadlocks.
    """

    def __init__(
        self,
        lane_ids: List[str] = ["LANE_N", "LANE_S", "LANE_E", "LANE_W"],
        config: Optional[AdaptiveSignalConfig] = None,
        db_manager: Optional[DatabaseManager] = None,
        db: Optional[DatabaseManager] = None,
    ):
        self.lane_ids = lane_ids
        self.cfg = config or SIGNAL_CFG
        self.db = db or db_manager or DatabaseManager()

        # Signal Phase State
        self.current_lane_idx = 0
        self.active_phase = "GREEN"  # "GREEN", "YELLOW", "ALL_RED"
        self.phase_time_elapsed = 0.0
        self.current_phase_allocated_s = self.cfg.min_green_seconds

        # Lane tracking states
        self.lane_wait_times: Dict[str, float] = {lid: 0.0 for lid in self.lane_ids}
        self.lane_priorities: Dict[str, PriorityLevel] = {
            lid: PriorityLevel.LEVEL_4_STANDARD for lid in self.lane_ids
        }
        self.lane_signal_colors: Dict[str, str] = {
            lid: ("GREEN" if lid == self.lane_ids[0] else "RED") for lid in self.lane_ids
        }

        # Emergency & Override State
        self.is_emergency_active = False
        self.emergency_lane_id: Optional[str] = None
        self.preemption_source = "NONE"
        self.evp_hold_timer = 0.0

        # VIP Override Flag
        self.vip_lane_id: Optional[str] = None

    @property
    def current_active_lane(self) -> str:
        return self.lane_ids[self.current_lane_idx]

    def update_tick(
        self,
        dt_seconds: Optional[float] = None,
        lane_metrics: Optional[Dict[str, LaneOccupancyMetrics]] = None,
        acoustic_siren_detected: bool = False,
        acoustic_siren_lane: Optional[str] = None,
        manual_override_lane: Optional[str] = None,
        dt_s: Optional[float] = None,
        metrics: Optional[Dict[str, LaneOccupancyMetrics]] = None,
        siren_detected: bool = False,
        siren_lane: Optional[str] = None,
    ) -> IntersectionStateSnapshot:
        dt = dt_seconds if dt_seconds is not None else (dt_s if dt_s is not None else 0.5)
        m_dict = lane_metrics if lane_metrics is not None else (metrics if metrics is not None else {})
        acoustic_siren = acoustic_siren_detected or siren_detected
        acoustic_target_lane = acoustic_siren_lane or siren_lane

        self.phase_time_elapsed += dt

        # 1. Update waiting times for non-green lanes
        for lid in self.lane_ids:
            if self.lane_signal_colors[lid] != "GREEN":
                self.lane_wait_times[lid] += dt
            else:
                self.lane_wait_times[lid] = 0.0

        # 2. Evaluate Priority Hierarchy for all lanes
        self._evaluate_priority_levels(
            m_dict, acoustic_siren, acoustic_target_lane, manual_override_lane
        )

        # 3. Handle Level 1 Emergency Preemption (Instant Smart Green Wave)
        if self.is_emergency_active and self.emergency_lane_id:
            return self._handle_emergency_preemption(dt, m_dict)

        # 4. Standard Adaptive Cycle State Machine
        if self.active_phase == "GREEN":
            # Check if current green time expired
            if self.phase_time_elapsed >= self.current_phase_allocated_s:
                # Transition to YELLOW
                self.active_phase = "YELLOW"
                self.phase_time_elapsed = 0.0
                self.lane_signal_colors[self.current_active_lane] = "YELLOW"

        elif self.active_phase == "YELLOW":
            if self.phase_time_elapsed >= self.cfg.yellow_seconds:
                # Transition to ALL_RED
                self.active_phase = "ALL_RED"
                self.phase_time_elapsed = 0.0
                for lid in self.lane_ids:
                    self.lane_signal_colors[lid] = "RED"

        elif self.active_phase == "ALL_RED":
            if self.phase_time_elapsed >= self.cfg.all_red_seconds:
                # Select next best lane using Decision Matrix & compute adaptive green
                next_lane_id = self._select_next_lane(m_dict)
                self.current_lane_idx = self.lane_ids.index(next_lane_id)
                self.active_phase = "GREEN"
                self.phase_time_elapsed = 0.0

                # Compute dynamic green duration based on live occupancy
                self.current_phase_allocated_s = self._calculate_adaptive_green_duration(
                    next_lane_id, m_dict
                )

                for lid in self.lane_ids:
                    self.lane_signal_colors[lid] = "GREEN" if lid == next_lane_id else "RED"

        return self._build_state_snapshot()

    def _evaluate_priority_levels(
        self,
        lane_metrics: Dict[str, LaneOccupancyMetrics],
        acoustic_siren: bool,
        acoustic_lane: Optional[str],
        manual_override: Optional[str],
    ):
        """Classifies each lane into Level 1, Level 2, Level 3, or Level 4."""
        emergency_candidate = None
        source = "NONE"

        # Check visual emergency detection from YOLOv8
        for lid, m in lane_metrics.items():
            if m.has_emergency_vehicle:
                emergency_candidate = lid
                source = "VISUAL_YOLO"
                break

        # Check acoustic siren fail-safe
        if acoustic_siren and not emergency_candidate:
            emergency_candidate = acoustic_lane or self.current_active_lane
            source = "ACOUSTIC_FFT"

        if manual_override and not emergency_candidate:
            emergency_candidate = manual_override
            source = "MANUAL_EVP"

        if emergency_candidate:
            self.is_emergency_active = True
            self.emergency_lane_id = emergency_candidate
            self.preemption_source = source
            self.evp_hold_timer = self.cfg.evp_clearance_hold_seconds
        elif self.is_emergency_active:
            # Decrement hold timer once emergency vehicle exits
            self.evp_hold_timer -= 0.1
            if self.evp_hold_timer <= 0:
                self.is_emergency_active = False
                self.emergency_lane_id = None
                self.preemption_source = "NONE"

        # Assign per-lane priorities
        for lid in self.lane_ids:
            if lid == self.emergency_lane_id and self.is_emergency_active:
                self.lane_priorities[lid] = PriorityLevel.LEVEL_1_EMERGENCY
            elif lid == self.vip_lane_id:
                self.lane_priorities[lid] = PriorityLevel.LEVEL_2_VIP
            elif (
                lid in lane_metrics
                and lane_metrics[lid].occupancy_percent >= self.cfg.high_density_threshold_percent
            ):
                self.lane_priorities[lid] = PriorityLevel.LEVEL_3_HIGH_DENSITY
            else:
                self.lane_priorities[lid] = PriorityLevel.LEVEL_4_STANDARD

    def _handle_emergency_preemption(
        self, dt_seconds: float, lane_metrics: Dict[str, LaneOccupancyMetrics]
    ) -> IntersectionStateSnapshot:
        """
        Instant Smart Green Wave Corridor:
        Forces immediate green to emergency lane and instantly locks conflicting lanes to RED.
        """
        target_lane = self.emergency_lane_id

        if self.current_active_lane == target_lane and self.lane_signal_colors[target_lane] == "GREEN":
            # Already green; extend green phase to hold clearance
            self.current_phase_allocated_s = max(self.current_phase_allocated_s, 20.0)
            for lid in self.lane_ids:
                self.lane_signal_colors[lid] = "GREEN" if lid == target_lane else "RED"
        else:
            # Fast transition to clear conflicting active traffic
            if self.active_phase == "GREEN":
                self.active_phase = "YELLOW"
                self.phase_time_elapsed = 0.0
                self.lane_signal_colors[self.current_active_lane] = "YELLOW"
            elif self.active_phase == "YELLOW" and self.phase_time_elapsed >= 1.5:  # Fast yellow
                self.active_phase = "GREEN"
                self.phase_time_elapsed = 0.0
                self.current_lane_idx = self.lane_ids.index(target_lane)
                self.current_phase_allocated_s = self.cfg.evp_clearance_hold_seconds
                for lid in self.lane_ids:
                    self.lane_signal_colors[lid] = "GREEN" if lid == target_lane else "RED"

        return self._build_state_snapshot()

    def _select_next_lane(self, lane_metrics: Dict[str, LaneOccupancyMetrics]) -> str:
        """
        Anti-Deadlock Arbitrator:
        Selects next lane using Level 1-4 priority + anti-starvation threshold.
        """
        # 1. Check Anti-Starvation (highest non-emergency rule)
        for lid in self.lane_ids:
            if self.lane_wait_times[lid] >= self.cfg.starvation_max_wait_seconds:
                return lid

        # 2. Check Level 2 VIP
        if self.vip_lane_id and self.vip_lane_id != self.current_active_lane:
            return self.vip_lane_id

        # 3. Check Level 3 High-Density Lanes (> 70% Occupancy)
        high_density_lanes = [
            lid
            for lid in self.lane_ids
            if lid != self.current_active_lane
            and lid in lane_metrics
            and lane_metrics[lid].occupancy_percent >= self.cfg.high_density_threshold_percent
        ]
        if high_density_lanes:
            # Pick highest spatial occupancy among Level 3
            return max(high_density_lanes, key=lambda l: lane_metrics[l].spatial_occupancy_m2)

        # 4. Level 4 Standard Traffic: Dynamic proportional pressure selection
        candidates = [lid for lid in self.lane_ids if lid != self.current_active_lane]
        if not candidates:
            candidates = self.lane_ids

        # Score candidate based on pressure + wait time boost
        def lane_score(lid: str) -> float:
            m = lane_metrics.get(lid)
            density = m.occupancy_percent if m else 10.0
            pcu = m.total_pcu if m else 1.0
            wait_penalty = (self.lane_wait_times[lid] / 30.0) * 15.0
            return (0.6 * density) + (0.4 * pcu * 5.0) + wait_penalty

        return max(candidates, key=lane_score)

    def _calculate_adaptive_green_duration(
        self, selected_lane: str, lane_metrics: Dict[str, LaneOccupancyMetrics]
    ) -> float:
        """
        Calculates dynamic green light duration proportional to spatial occupancy
        and PCU traffic pressure across the 4-way intersection.
        """
        if selected_lane not in lane_metrics:
            return self.cfg.min_green_seconds

        target = lane_metrics[selected_lane]
        
        # True Queue Pressure calculation
        # Normalized between 0.0 and 1.0
        norm_density = min(1.0, target.occupancy_percent / 100.0)
        norm_pcu = min(1.0, target.total_pcu / 25.0)
        pressure = (0.65 * norm_density) + (0.35 * norm_pcu)

        # Linear allocation between min_green and max_green
        dyn_green = self.cfg.min_green_seconds + pressure * (
            self.cfg.max_green_seconds - self.cfg.min_green_seconds
        )

        return round(float(dyn_green), 1)

    def _build_state_snapshot(self) -> IntersectionStateSnapshot:
        """Assembles a point-in-time snapshot of signal heads and timers."""
        lane_states = {}
        for lid in self.lane_ids:
            color = self.lane_signal_colors[lid]
            if color == "GREEN":
                time_left = max(0.0, self.current_phase_allocated_s - self.phase_time_elapsed)
                alloc = self.current_phase_allocated_s
            elif color == "YELLOW":
                time_left = max(0.0, self.cfg.yellow_seconds - self.phase_time_elapsed)
                alloc = self.cfg.yellow_seconds
            else:
                time_left = 0.0
                alloc = 0.0

            lane_states[lid] = LaneSignalState(
                lane_id=lid,
                signal_color=color,
                time_remaining_s=round(time_left, 1),
                current_phase_duration_s=round(alloc, 1),
                wait_time_s=round(self.lane_wait_times[lid], 1),
                priority_level=self.lane_priorities[lid],
                is_evp_active=(self.is_emergency_active and lid == self.emergency_lane_id),
            )

        return IntersectionStateSnapshot(
            timestamp=time.time(),
            active_lane_id=self.current_active_lane,
            active_phase=self.active_phase,
            lane_states=lane_states,
            is_emergency_preemption=self.is_emergency_active,
            emergency_lane_id=self.emergency_lane_id,
            preemption_source=self.preemption_source,
        )
