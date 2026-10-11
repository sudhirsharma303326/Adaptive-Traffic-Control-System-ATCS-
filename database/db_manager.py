"""
=============================================================================
ATCS Database Manager (SQLite)
=============================================================================
Provides production-ready, thread-safe persistence for:
1. Automated E-Challan records (Red-Light breaches & ANPR)
2. Lane Traffic Telemetry (counts, occupancy %, signal states)
3. Carbon & Sustainability metrics (avoided delay, fuel saved, CO2 prevented)
"""

import sqlite3
import threading
from typing import List, Dict, Any, Optional
from datetime import datetime
import os
import json
from config.settings import DB_PATH


class DatabaseManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, db_path: str = DB_PATH):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(DatabaseManager, cls).__new__(cls)
                cls._instance.db_path = db_path
                cls._instance._init_schema()
            return cls._instance

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self):
        """Initializes tables, indexes, and constraints if they do not exist."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # 1. E-Challan Violations Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS echallans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    lane_id TEXT NOT NULL,
                    vehicle_type TEXT NOT NULL,
                    license_plate TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    image_path TEXT NOT NULL,
                    violation_type TEXT NOT NULL DEFAULT 'RED_LIGHT_JUMP',
                    fine_amount REAL NOT NULL DEFAULT 1000.0,
                    status TEXT NOT NULL DEFAULT 'PENDING'
                );
            """)

            # 2. Traffic Flow Telemetry Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS traffic_telemetry (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    lane_id TEXT NOT NULL,
                    vehicle_count INTEGER NOT NULL,
                    spatial_occupancy_m2 REAL NOT NULL,
                    occupancy_percent REAL NOT NULL,
                    pcu_count REAL NOT NULL,
                    signal_state TEXT NOT NULL,
                    allocated_green_s REAL NOT NULL,
                    priority_level INTEGER NOT NULL
                );
            """)

            # 3. Sustainability and Carbon Offset Log
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sustainability_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    cycle_id INTEGER,
                    avoided_delay_seconds REAL NOT NULL,
                    fuel_saved_liters REAL NOT NULL,
                    co2_prevented_kg REAL NOT NULL,
                    monetary_savings_usd REAL NOT NULL
                );
            """)

            # Create Indexes for fast querying
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_echallans_plate ON echallans(license_plate);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_echallans_time ON echallans(timestamp);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_telemetry_time ON traffic_telemetry(timestamp);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sustainability_time ON sustainability_log(timestamp);")
            
            conn.commit()

    def log_echallan(
        self,
        lane_id: str,
        vehicle_type: str,
        license_plate: str,
        confidence: float,
        image_path: str,
        violation_type: str = "RED_LIGHT_JUMP",
        fine_amount: float = 1000.0,
        status: str = "PENDING",
    ) -> int:
        """Inserts an e-challan record and returns the new violation ID."""
        ts = datetime.utcnow().isoformat()
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO echallans (
                        timestamp, lane_id, vehicle_type, license_plate,
                        confidence, image_path, violation_type, fine_amount, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ts,
                        lane_id,
                        vehicle_type,
                        license_plate,
                        confidence,
                        image_path,
                        violation_type,
                        fine_amount,
                        status,
                    ),
                )
                conn.commit()
                return cursor.lastrowid

    def log_telemetry(
        self,
        lane_id: str,
        vehicle_count: int,
        spatial_occupancy_m2: float,
        occupancy_percent: float,
        pcu_count: float,
        signal_state: str,
        allocated_green_s: float,
        priority_level: int,
    ):
        """Logs real-time lane traffic state."""
        ts = datetime.utcnow().isoformat()
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO traffic_telemetry (
                        timestamp, lane_id, vehicle_count, spatial_occupancy_m2,
                        occupancy_percent, pcu_count, signal_state, allocated_green_s, priority_level
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ts,
                        lane_id,
                        vehicle_count,
                        spatial_occupancy_m2,
                        occupancy_percent,
                        pcu_count,
                        signal_state,
                        allocated_green_s,
                        priority_level,
                    ),
                )
                conn.commit()

    def log_sustainability(
        self,
        cycle_id: int,
        avoided_delay_seconds: float,
        fuel_saved_liters: float,
        co2_prevented_kg: float,
        monetary_savings_usd: float,
    ):
        """Logs sustainability metrics for an adaptive signal cycle."""
        ts = datetime.utcnow().isoformat()
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO sustainability_log (
                        timestamp, cycle_id, avoided_delay_seconds,
                        fuel_saved_liters, co2_prevented_kg, monetary_savings_usd
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ts,
                        cycle_id,
                        avoided_delay_seconds,
                        fuel_saved_liters,
                        co2_prevented_kg,
                        monetary_savings_usd,
                    ),
                )
                conn.commit()

    def get_recent_echallans(self, limit: int = 25) -> List[Dict[str, Any]]:
        """Returns the most recent e-challans."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM echallans ORDER BY id DESC LIMIT ?", (limit,)
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_cumulative_sustainability(self) -> Dict[str, float]:
        """Returns total cumulative carbon and fuel metrics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    COALESCE(SUM(avoided_delay_seconds), 0.0) as total_delay_s,
                    COALESCE(SUM(fuel_saved_liters), 0.0) as total_fuel_l,
                    COALESCE(SUM(co2_prevented_kg), 0.0) as total_co2_kg,
                    COALESCE(SUM(monetary_savings_usd), 0.0) as total_usd,
                    COUNT(id) as logged_cycles
                FROM sustainability_log;
            """)
            row = cursor.fetchone()
            return dict(row) if row else {
                "total_delay_s": 0.0,
                "total_fuel_l": 0.0,
                "total_co2_kg": 0.0,
                "total_usd": 0.0,
                "logged_cycles": 0,
            }

    def get_total_violations_count(self) -> int:
        """Returns the total number of recorded red light violations."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM echallans;")
            return cursor.fetchone()[0]
