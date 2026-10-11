"""
=============================================================================
Traffic Simulation & Multi-Camera Video Stream Generator
=============================================================================
Generates photorealistic, animated 4-way intersection CCTV video feeds:
- 4 Approach Lanes: North, South, East, West.
- Diverse vehicle entities: Cars, Motorcycles, Buses, Heavy Trucks, Ambulances.
- Physics-based motion: Vehicles approach, queue when RED, accelerate on GREEN,
  and occasionally commit a Red-Light Stopline breach for E-Challan testing.
- Emergency vehicle spawning: Injectable ambulance/police with flashing lightbars.
- Real-time video frame generator compatible with OpenCV and Streamlit.
"""

import numpy as np
import cv2
import random
import time
from typing import Dict, List, Tuple, Optional
from config.settings import DEFAULT_LANES, IntersectionLaneConfig


class SimulatedVehicleEntity:
    def __init__(
        self,
        vehicle_id: int,
        vehicle_type: str,
        lane_id: str,
        y_pos: float = 40.0,
        speed: float = 3.5,
        will_breach_red: bool = False,
    ):
        self.vehicle_id = vehicle_id
        self.vehicle_type = vehicle_type
        self.lane_id = lane_id
        self.y_pos = y_pos
        self.speed = speed
        self.will_breach_red = will_breach_red
        
        # Dimensions and colors
        if vehicle_type == "motorcycle":
            self.width = 24
            self.height = 38
            self.color = (255, 180, 50)  # Cyan/Blue
        elif vehicle_type == "car":
            self.width = 46
            self.height = 70
            self.color = (80, 150, 240)  # Orange/Gold
        elif vehicle_type == "bus":
            self.width = 62
            self.height = 140
            self.color = (50, 200, 50)   # Green
        elif vehicle_type == "truck":
            self.width = 64
            self.height = 150
            self.color = (180, 100, 50)  # Dark Blue
        elif vehicle_type == "emergency":
            self.width = 52
            self.height = 85
            self.color = (245, 245, 245) # White Ambulance with red stripes
        else:
            self.width = 46
            self.height = 70
            self.color = (120, 120, 120)

        # X position lateral jitter within lane approach (centered around 320)
        self.x_pos = 320 + random.randint(-40, 40)
        self.strobe_state = 0


class SyntheticIntersectionGenerator:
    """
    Simulates real-time CCTV streams for 4 intersection lanes.
    """

    def __init__(self, lanes: Optional[Dict[str, IntersectionLaneConfig]] = None):
        self.lanes = lanes or DEFAULT_LANES
        self.weather = "clear"
        self.vehicles: Dict[str, List[SimulatedVehicleEntity]] = {lid: [] for lid in self.lanes}
        self.vehicle_id_seq = 100
        self.frame_width = 640
        self.frame_height = 480
        self.last_spawn_time = {lid: time.time() for lid in self.lanes}

        # Initialize with baseline traffic
        for lid in self.lanes:
            self._seed_lane_vehicles(lid, count=random.randint(2, 5))

    def set_weather(self, weather: str):
        """Changes the camera simulation weather without resetting traffic state."""
        allowed_weather = {"clear", "rain", "fog", "night"}
        self.weather = weather if weather in allowed_weather else "clear"

    def _seed_lane_vehicles(self, lane_id: str, count: int = 3):
        y_cursor = 70.0
        for _ in range(count):
            vtype = random.choices(["car", "motorcycle", "bus", "truck"], weights=[0.55, 0.25, 0.10, 0.10])[0]
            self.vehicle_id_seq += 1
            v = SimulatedVehicleEntity(
                vehicle_id=self.vehicle_id_seq,
                vehicle_type=vtype,
                lane_id=lane_id,
                y_pos=y_cursor,
                speed=random.uniform(2.5, 4.0),
            )
            self.vehicles[lane_id].append(v)
            y_cursor += v.height + random.randint(25, 45)

    def spawn_emergency_vehicle(self, lane_id: str):
        """Manually or event-driven injection of an emergency vehicle."""
        self.vehicle_id_seq += 1
        ev = SimulatedVehicleEntity(
            vehicle_id=self.vehicle_id_seq,
            vehicle_type="emergency",
            lane_id=lane_id,
            y_pos=30.0,
            speed=6.0,  # Fast moving
            will_breach_red=False,
        )
        self.vehicles[lane_id].insert(0, ev)
        return ev

    def trigger_red_breach(self, lane_id: str):
        """Forces the next approaching vehicle in lane to run a red light."""
        if self.vehicles[lane_id]:
            self.vehicles[lane_id][0].will_breach_red = True
            self.vehicles[lane_id][0].speed = 5.0

    trigger_breach = trigger_red_breach
    spawn_emergency = spawn_emergency_vehicle

    def generate_lane_frame(
        self,
        lane_id: str,
        signal_state: str,
        time_remaining_s: float,
        weather: Optional[str] = None,
    ) -> np.ndarray:
        """
        Renders an individual CCTV approach feed for a lane with road markings,
        moving traffic, brake lights, and visual stoplines.
        """
        lane_cfg = self.lanes[lane_id]
        frame = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        
        # 1. Asphalt Road Texture (Dark Slate Gray)
        frame[:] = (38, 40, 44)

        # 2. Road Borders & Curbs
        cv2.rectangle(frame, (0, 0), (120, self.frame_height), (55, 60, 65), -1)
        cv2.rectangle(frame, (520, 0), (self.frame_width, self.frame_height), (55, 60, 65), -1)
        
        # Curb lines
        cv2.line(frame, (120, 0), (120, self.frame_height), (120, 130, 140), 2)
        cv2.line(frame, (520, 0), (520, self.frame_height), (120, 130, 140), 2)

        # 3. Lane Approach ROI (Subtle semi-transparent highlight)
        roi_pts = np.array(lane_cfg.roi_polygon, dtype=np.int32)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [roi_pts], (45, 55, 50))
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        cv2.polylines(frame, [roi_pts], True, (70, 90, 80), 1, cv2.LINE_AA)

        # 4. Center Dashed White Lines
        for y in range(0, self.frame_height, 40):
            cv2.line(frame, (320, y), (320, y + 20), (220, 220, 220), 2)

        # 5. Virtual Stopline (Physical stopline marker on asphalt)
        p1, p2 = lane_cfg.stopline
        stopline_color = (0, 0, 255) if signal_state == "RED" else (0, 220, 0)
        cv2.line(frame, p1, p2, (240, 240, 240), 4)  # Road paint
        # Virtual sensor beam
        cv2.line(frame, p1, p2, stopline_color, 2, cv2.LINE_AA)

        # Pedestrian Crosswalk (Zebra Stripes past stopline)
        for zx in range(140, 500, 30):
            cv2.rectangle(frame, (zx, int(p1[1] + 12)), (zx + 18, int(p1[1] + 45)), (200, 200, 200), -1)

        # 6. Update Vehicle Positions and Render
        stopline_y = p1[1]
        active_vehicles = self.vehicles[lane_id]

        # Vehicle Spawning Logic
        now = time.time()
        if len(active_vehicles) < 6 and (now - self.last_spawn_time[lane_id] > random.uniform(2.5, 4.5)):
            vtype = random.choices(["car", "motorcycle", "bus", "truck"], weights=[0.55, 0.25, 0.10, 0.10])[0]
            self.vehicle_id_seq += 1
            nv = SimulatedVehicleEntity(
                vehicle_id=self.vehicle_id_seq,
                vehicle_type=vtype,
                lane_id=lane_id,
                y_pos=20.0,
                speed=random.uniform(2.5, 4.0),
            )
            # Ensure no direct collision with spawn zone
            if not active_vehicles or active_vehicles[0].y_pos > 80.0:
                active_vehicles.insert(0, nv)
                self.last_spawn_time[lane_id] = now

        # Update & draw vehicles
        i = 0
        while i < len(active_vehicles):
            v = active_vehicles[i]
            lead_vehicle = active_vehicles[i - 1] if i > 0 else None

            # Determine Target Speed
            should_stop = False
            dist_to_stopline = stopline_y - (v.y_pos + v.height)

            if signal_state in ["RED", "YELLOW"] and not v.will_breach_red:
                if 0 < dist_to_stopline < 60:
                    should_stop = True

            # Collision avoidance with lead vehicle
            if lead_vehicle is not None:
                gap = lead_vehicle.y_pos - (v.y_pos + v.height)
                if gap < 20:
                    should_stop = True

            # Adjust speed
            if should_stop:
                v.speed = max(0.0, v.speed - 0.35)
            else:
                target_speed = 3.5 if v.vehicle_type != "emergency" else 5.5
                v.speed = min(target_speed, v.speed + 0.15)

            v.y_pos += v.speed

            # Draw vehicle body
            vx1 = int(v.x_pos - v.width // 2)
            vy1 = int(v.y_pos)
            vx2 = int(v.x_pos + v.width // 2)
            vy2 = int(v.y_pos + v.height)

            # Shadow
            cv2.rectangle(frame, (vx1 + 4, vy1 + 4), (vx2 + 4, vy2 + 4), (20, 20, 20), -1)
            # Main chassis
            cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), v.color, -1)
            cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), (40, 40, 40), 1)

            # Windshield / Roof Detail
            wh = int(v.height * 0.28)
            cv2.rectangle(frame, (vx1 + 4, vy1 + int(v.height * 0.22)), (vx2 - 4, vy1 + int(v.height * 0.22) + wh), (30, 35, 40), -1)

            # Emergency Flashing Strobe Bar
            if v.vehicle_type == "emergency":
                v.strobe_state = (v.strobe_state + 1) % 4
                s_color1 = (0, 0, 255) if v.strobe_state < 2 else (255, 0, 0)
                s_color2 = (255, 0, 0) if v.strobe_state < 2 else (0, 0, 255)
                bar_y = vy1 + 10
                cv2.rectangle(frame, (vx1 + 6, bar_y), (vx1 + 20, bar_y + 8), s_color1, -1)
                cv2.rectangle(frame, (vx2 - 20, bar_y), (vx2 - 6, bar_y + 8), s_color2, -1)

            # Headlights / Tail Brake Lights
            if should_stop:
                # Bright red brake lights at top/rear
                cv2.circle(frame, (vx1 + 6, vy1 + 4), 3, (0, 0, 255), -1)
                cv2.circle(frame, (vx2 - 6, vy1 + 4), 3, (0, 0, 255), -1)

            # Remove vehicles that drive off screen
            if v.y_pos > self.frame_height + 50:
                active_vehicles.pop(i)
            else:
                i += 1

        # 7. Apply camera weather before HUD overlays stay crisp and readable.
        self._apply_weather_effects(frame, weather or self.weather)

        # 8. Traffic Signal Head HUD (Upper Right Corner)
        self._render_signal_head(frame, signal_state, time_remaining_s)

        # 9. Lane Identification Overlay
        cv2.putText(
            frame,
            f"{lane_cfg.name.upper()} [{lane_id}]",
            (15, 30),
            cv2.FONT_HERSHEY_DUPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        return frame

    # Seamless alias for generate_lane_frame
    generate_frame = generate_lane_frame

    def _apply_weather_effects(self, frame: np.ndarray, weather: str):
        """Adds lightweight weather artifacts while retaining camera visibility."""
        if weather == "night":
            dark_frame = cv2.convertScaleAbs(frame, alpha=0.62, beta=4)
            blue_tint = np.full_like(frame, (28, 40, 72))
            cv2.addWeighted(dark_frame, 0.78, blue_tint, 0.22, 0, frame)
        elif weather == "fog":
            fog_layer = np.full_like(frame, (190, 198, 205))
            cv2.addWeighted(fog_layer, 0.22, frame, 0.78, 0, frame)
            frame[:] = cv2.detailEnhance(frame, sigma_s=10, sigma_r=0.15)
        elif weather == "rain":
            rain_layer = frame.copy()
            for x in range(25, self.frame_width, 38):
                cv2.line(rain_layer, (x, 20), (x - 18, 88), (185, 205, 225), 1, cv2.LINE_AA)
            cv2.addWeighted(rain_layer, 0.28, frame, 0.72, 0, frame)

    def _render_signal_head(self, frame: np.ndarray, signal_state: str, time_remaining_s: float):
        """Draws realistic LED traffic signal light head with active glow."""
        x, y = 570, 20
        w, h = 50, 110
        cv2.rectangle(frame, (x, y), (x + w, y + h), (20, 20, 20), -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 2)

        # Signal bulbs
        rad = 12
        cx = x + w // 2
        # RED
        red_color = (0, 0, 255) if signal_state == "RED" else (0, 0, 60)
        cv2.circle(frame, (cx, y + 22), rad, red_color, -1)
        # YELLOW
        yel_color = (0, 220, 255) if signal_state == "YELLOW" else (0, 60, 60)
        cv2.circle(frame, (cx, y + 55), rad, yel_color, -1)
        # GREEN
        grn_color = (0, 255, 0) if signal_state == "GREEN" else (0, 60, 0)
        cv2.circle(frame, (cx, y + 88), rad, grn_color, -1)

        # Countdown Timer Box
        cv2.rectangle(frame, (x - 45, y + 35), (x - 5, y + 75), (10, 10, 10), -1)
        cv2.rectangle(frame, (x - 45, y + 35), (x - 5, y + 75), (80, 80, 80), 1)
        t_str = f"{int(time_remaining_s):02d}"
        t_color = (0, 255, 0) if signal_state == "GREEN" else (0, 200, 255) if signal_state == "YELLOW" else (0, 0, 255)
        cv2.putText(frame, t_str, (x - 40, y + 65), cv2.FONT_HERSHEY_DUPLEX, 0.7, t_color, 2)
