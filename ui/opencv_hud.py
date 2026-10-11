"""
=============================================================================
ATCS OpenCV Real-Time 4-Lane HUD Display
=============================================================================
Renders a 2x2 multi-camera intersection grid with live HUD overlays:
- 4 CCTV Lane Feeds (North, South, East, West)
- Dynamic Traffic Signal Status (Green, Yellow, Red timers)
- Level 1-4 Priority Matrix tags
- Spatial Occupancy meters (m² & %)
- Smart Green Wave Emergency Override Alerts
- Real-Time Sustainability Counter (Fuel, CO2, Delay)
- Interactive Keyboard Triggers (Ambulance, Acoustic Siren, Red Breach)
"""

import cv2
import numpy as np
import time
from typing import Dict, Any, Optional

from core.priority_matrix import IntersectionStateSnapshot
from core.vehicle_detector import LaneOccupancyMetrics
from core.carbon_tracker import SustainabilityMetrics


class OpenCVIntersectionHUD:
    def __init__(self, target_width: int = 1280, target_height: int = 800):
        self.target_width = target_width
        self.target_height = target_height
        self.pane_w = target_width // 2
        self.pane_h = (target_height - 130) // 2  # Leave space for top & bottom bars
        self.flash_counter = 0

    def render_dashboard(
        self,
        lane_frames: Dict[str, np.ndarray],
        state_snapshot: IntersectionStateSnapshot,
        occupancy_metrics: Dict[str, LaneOccupancyMetrics],
        sustainability: SustainabilityMetrics,
        fps: float,
        latest_violation_msg: Optional[str] = None,
    ) -> np.ndarray:
        """Composes 4 lane camera streams + telemetry into a single unified HUD frame."""
        self.flash_counter = (self.flash_counter + 1) % 60
        canvas = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)

        # 1. TOP HEADER BANNER (Height: 70px)
        self._render_header(canvas, state_snapshot, fps)

        # 2. 2x2 CAMERA QUAD GRID (Height: pane_h * 2)
        # Order: Top-Left: LANE_N, Top-Right: LANE_S, Bottom-Left: LANE_E, Bottom-Right: LANE_W
        quad_layout = [
            ("LANE_N", 0, 70),
            ("LANE_S", self.pane_w, 70),
            ("LANE_E", 0, 70 + self.pane_h),
            ("LANE_W", self.pane_w, 70 + self.pane_h),
        ]

        for lane_id, px, py in quad_layout:
            frame = lane_frames.get(lane_id)
            if frame is not None:
                # Resize lane feed to pane dimension
                resized = cv2.resize(frame, (self.pane_w, self.pane_h))
                
                # Annotate lane specific telemetry on the feed
                m = occupancy_metrics.get(lane_id)
                s = state_snapshot.lane_states.get(lane_id)
                self._annotate_lane_feed(resized, lane_id, m, s)
                
                # Copy into canvas
                canvas[py : py + self.pane_h, px : px + self.pane_w] = resized

                # Border around pane
                border_color = (0, 255, 0) if (s and s.signal_color == "GREEN") else (60, 60, 60)
                cv2.rectangle(
                    canvas, (px, py), (px + self.pane_w, py + self.pane_h), border_color, 2
                )

        # 3. EMERGENCY OVERRIDE BANNER (If Active)
        if state_snapshot.is_emergency_preemption:
            self._render_emergency_banner(canvas, state_snapshot)

        # 4. BOTTOM TELEMETRY & SUSTAINABILITY BAR (Height: 60px)
        self._render_bottom_bar(canvas, sustainability, latest_violation_msg)

        return canvas

    def _render_header(
        self, canvas: np.ndarray, state: IntersectionStateSnapshot, fps: float
    ):
        """Renders top command center bar."""
        cv2.rectangle(canvas, (0, 0), (self.target_width, 70), (18, 20, 24), -1)
        cv2.line(canvas, (0, 70), (self.target_width, 70), (0, 180, 240), 2)

        # Title
        cv2.putText(
            canvas,
            "ATCS // AI-POWERED ADAPTIVE TRAFFIC CONTROL SYSTEM",
            (20, 28),
            cv2.FONT_HERSHEY_DUPLEX,
            0.68,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "EDGE-AI YOLOv8 SENSING * ACOUSTIC FFT * VIRTUAL ANPR * REAL-TIME CARBON OFFSET",
            (20, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 200, 255),
            1,
            cv2.LINE_AA,
        )

        # Active Lane & Phase Badge
        phase_color = (0, 255, 0) if state.active_phase == "GREEN" else (0, 220, 255) if state.active_phase == "YELLOW" else (0, 0, 255)
        badge_text = f"ACTIVE: {state.active_lane_id} [{state.active_phase}]"
        cv2.rectangle(canvas, (self.target_width - 340, 14), (self.target_width - 110, 56), (30, 35, 40), -1)
        cv2.rectangle(canvas, (self.target_width - 340, 14), (self.target_width - 110, 56), phase_color, 2)
        cv2.putText(
            canvas,
            badge_text,
            (self.target_width - 330, 42),
            cv2.FONT_HERSHEY_DUPLEX,
            0.55,
            phase_color,
            2,
            cv2.LINE_AA,
        )

        # FPS meter
        cv2.putText(
            canvas,
            f"FPS: {fps:.1f}",
            (self.target_width - 90, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

    def _annotate_lane_feed(
        self,
        pane: np.ndarray,
        lane_id: str,
        m: Optional[LaneOccupancyMetrics],
        s: Optional[Any],
    ):
        """Draws telemetry metrics HUD overlay directly on each lane frame."""
        h, w = pane.shape[:2]
        
        # Telemetry Card (Bottom-left of pane)
        card_w, card_h = 240, 100
        cx, cy = 10, h - card_h - 10
        overlay = pane.copy()
        cv2.rectangle(overlay, (cx, cy), (cx + card_w, cy + card_h), (10, 12, 15), -1)
        cv2.addWeighted(overlay, 0.75, pane, 0.25, 0, pane)
        cv2.rectangle(pane, (cx, cy), (cx + card_w, cy + card_h), (70, 80, 90), 1)

        if m is not None:
            veh_count = m.vehicle_count
            occ_m2 = m.spatial_occupancy_m2
            occ_pct = m.occupancy_percent
            pcu = m.total_pcu
        else:
            veh_count, occ_m2, occ_pct, pcu = 0, 0.0, 0.0, 0.0

        p_level = f"Level {s.priority_level.value}" if s else "Level 4"
        
        # Lines
        cv2.putText(pane, f"VEHICLES: {veh_count}  |  PCU: {pcu:.1f}", (cx + 10, cy + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1)
        cv2.putText(pane, f"SPATIAL OCC: {occ_m2:.1f} m2 ({occ_pct:.0f}%)", (cx + 10, cy + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1)
        
        # Mini Occupancy Progress Bar
        bar_x, bar_y, bar_w, bar_h = cx + 10, cy + 55, card_w - 20, 8
        cv2.rectangle(pane, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 45, 50), -1)
        fill_w = int(bar_w * min(1.0, occ_pct / 100.0))
        bar_color = (0, 255, 0) if occ_pct < 50 else (0, 220, 255) if occ_pct < 75 else (0, 0, 255)
        cv2.rectangle(pane, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)

        cv2.putText(pane, f"PRIORITY: {p_level}", (cx + 10, cy + 85), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 220, 100), 1)

    def _render_emergency_banner(self, canvas: np.ndarray, state: IntersectionStateSnapshot):
        """Renders flashing smart green wave corridor alert."""
        if self.flash_counter < 30:
            banner_y = 70
            cv2.rectangle(
                canvas,
                (0, banner_y),
                (self.target_width, banner_y + 45),
                (0, 30, 200),
                -1,
            )
            msg = f"*** SMART GREEN WAVE CORRIDOR ACTIVE *** [TARGET: {state.emergency_lane_id} | TRIGGER: {state.preemption_source} | CONFLICTING LANES LOCKED RED]"
            cv2.putText(
                canvas,
                msg,
                (40, banner_y + 30),
                cv2.FONT_HERSHEY_DUPLEX,
                0.62,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    def _render_bottom_bar(
        self,
        canvas: np.ndarray,
        sust: SustainabilityMetrics,
        violation_msg: Optional[str],
    ):
        """Renders bottom live sustainability offset bar and controls info."""
        y = self.target_height - 60
        cv2.rectangle(canvas, (0, y), (self.target_width, self.target_height), (15, 17, 20), -1)
        cv2.line(canvas, (0, y), (self.target_width, y), (0, 180, 240), 1)

        # Sustainability Metrics Display
        metrics_text = (
            f"CO2 PREVENTED: {sust.cumulative_co2_kg:.2f} kg  |  "
            f"FUEL SAVED: {sust.cumulative_fuel_liters:.2f} L  |  "
            f"DELAY AVOIDED: {sust.cumulative_delay_hours:.2f} hrs  |  "
            f"TREES EQUIV: {sust.trees_planted_equivalent:.1f}"
        )
        cv2.putText(
            canvas,
            metrics_text,
            (20, y + 26),
            cv2.FONT_HERSHEY_DUPLEX,
            0.52,
            (0, 255, 150),
            1,
            cv2.LINE_AA,
        )

        # Keyboard Controls legend
        controls = "KEYS: [A] Spawn Ambulance | [S] Acoustic Siren FFT | [R] Red Breach Test | [Q] Exit"
        cv2.putText(
            canvas,
            controls,
            (20, y + 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (160, 170, 180),
            1,
            cv2.LINE_AA,
        )

        # Violation pop-up text if available
        if violation_msg:
            cv2.putText(
                canvas,
                f"E-CHALLAN: {violation_msg}",
                (self.target_width - 480, y + 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (0, 100, 255),
                1,
                cv2.LINE_AA,
            )
