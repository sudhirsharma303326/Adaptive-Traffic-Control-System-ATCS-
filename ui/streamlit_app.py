"""
=============================================================================
ATCS Streamlit Operations & Analytics Command Center
=============================================================================
Interactive Web Control Dashboard providing:
1. Live 4-Way Traffic Signal Status & Timers
2. Real-Time Spatial Occupancy Gauges & Vehicle Breakdowns
3. Acoustic Siren FFT Spectrum Visualizer (1000 Hz - 3000 Hz band)
4. Emergency Vehicle Preemption (EVP) Manual & Multi-Modal Override Controls
5. Automated E-Challan Law Enforcement Vault & Violation Inspector
6. Carbon Offset, Fuel Conservation, and Urban Tree Equivalency Meters
"""

import streamlit as st
import numpy as np
import pandas as pd
import time
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import DEFAULT_LANES, DB_PATH, EVIDENCE_DIR
from database.db_manager import DatabaseManager
from core.siren_detector import detect_siren_frequency, AcousticSirenMonitor
from core.priority_matrix import SignalPriorityController, PriorityLevel
from core.vehicle_detector import VehicleDetectionEngine, LaneOccupancyMetrics
from core.carbon_tracker import CarbonOffsetEngine
from core.traffic_simulator import SyntheticIntersectionGenerator


st.set_page_config(
    page_title="ATCS // AI Adaptive Traffic Control System",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom High-End Cyber Dark Theme CSS
st.markdown(
    """
    <style>
    .main {
        background: #f3f3f1;
    }
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
        max-width: 1440px !important;
    }
    .page-shell {
        display: flex;
        flex-direction: column;
        gap: 1.3rem;
        margin-top: 0.5rem;
    }
    .hero-shell {
        display: grid;
        grid-template-columns: 1.05fr 1.15fr;
        align-items: center;
        gap: 2rem;
        padding: 1.2rem 0.2rem 0.4rem;
    }
    .hero-copy {
        padding: 1.2rem 0.2rem 0;
    }
    .eyebrow {
        display: inline-block;
        background: #e7f4d7;
        border: 1px solid rgba(20, 60, 30, 0.08);
        color: #1b2a1b;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        border-radius: 999px;
        padding: 0.42rem 0.8rem;
        margin-bottom: 1.05rem;
    }
    .hero-copy h1 {
        margin: 0;
        font-size: clamp(3rem, 5vw, 5.2rem);
        line-height: 0.98;
        letter-spacing: -0.06em;
        color: #101010;
        font-weight: 800;
    }
    .hero-copy .highlight {
        display: block;
        color: #111111;
    }
    .hero-copy .subtitle {
        margin-top: 1.1rem;
        color: rgba(17, 17, 17, 0.72);
        font-size: 1rem;
        max-width: 620px;
        line-height: 1.8;
    }
    .cta-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.7rem;
        margin-top: 2rem;
    }
    .cta-chip {
        background: #111111;
        color: #f5f5f5;
        border-radius: 999px;
        padding: 0.72rem 1rem;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        box-shadow: 0 12px 24px rgba(17,17,17,0.10);
    }
    .cta-chip.secondary {
        background: #d8f565;
        color: #101010;
        border: 1px solid rgba(16,16,16,0.08);
    }
    .hero-panel {
        background: linear-gradient(145deg, rgba(225, 233, 228, 0.96), rgba(184, 201, 191, 0.9));
        border: 1px solid rgba(20, 29, 24, 0.06);
        border-radius: 24px;
        padding: 0.8rem;
        box-shadow: 0 28px 60px rgba(15, 20, 18, 0.12);
        min-height: 430px;
    }
    .panel-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.25rem 0.5rem 0.8rem;
    }
    .window-dots {
        display: flex;
        gap: 0.45rem;
    }
    .window-dots span {
        width: 9px;
        height: 9px;
        border-radius: 50%;
        background: rgba(18,18,18,0.6);
        display: inline-block;
    }
    .mini-pill {
        background: rgba(12, 27, 20, 0.75);
        color: #edf7f0;
        border-radius: 999px;
        padding: 0.3rem 0.65rem;
        font-size: 0.64rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .sim-window {
        background: linear-gradient(180deg, rgba(16, 31, 24, 0.98), rgba(8, 18, 14, 1));
        border-radius: 18px;
        padding: 1rem;
        min-height: 350px;
        position: relative;
        overflow: hidden;
    }
    .sim-window::before {
        content: "";
        position: absolute;
        inset: 8% 8%;
        border-radius: 18px;
        background: linear-gradient(90deg, rgba(21, 30, 24, 0.8) 0%, rgba(16, 31, 24, 0.5) 100%);
        transform: scale(1.04);
    }
    .sim-grid {
        position: relative;
        z-index: 1;
        height: 100%;
        min-height: 320px;
        display: grid;
        grid-template-columns: repeat(2, minmax(0,1fr));
        gap: 0.8rem;
    }
    .lane-card {
        border: 1px solid rgba(218, 250, 125, 0.18);
        background: rgba(255,255,255,0.02);
        border-radius: 12px;
        position: relative;
        overflow: hidden;
    }
    .lane-card::before {
        content: "";
        position: absolute;
        inset: auto 50% 0 50%;
        transform: translateX(-50%);
        width: 80%;
        height: 2px;
        background: rgba(212, 255, 128, 0.75);
    }
    .lane-card::after {
        content: "";
        position: absolute;
        left: 10%;
        right: 10%;
        top: 14%;
        height: 2px;
        background: rgba(255,255,255,0.1);
    }
    .lane-card .signal {
        position: absolute;
        right: 12px;
        top: 12px;
        width: 12px;
        height: 12px;
        background: #d8f565;
        border-radius: 50%;
        box-shadow: 0 0 16px rgba(216, 245, 101, 0.7);
    }
    .lane-name {
        position: absolute;
        left: 12px;
        bottom: 10px;
        color: rgba(232, 242, 233, 0.72);
        font-size: 0.62rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }
    .feature-strip {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.8rem;
        margin-top: 0.2rem;
    }
    .feature-card {
        background: linear-gradient(135deg, #d8f565, #c3ef5e);
        border-radius: 12px;
        min-height: 124px;
        padding: 1rem 1.1rem;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        border: 1px solid rgba(8, 13, 10, 0.08);
        box-shadow: 0 12px 24px rgba(17,17,17,0.03);
    }
    .feature-card:nth-child(2n) {
        background: linear-gradient(135deg, #e6f7d8, #d1f0b4);
    }
    .feature-card:nth-child(3n) {
        background: linear-gradient(135deg, #dfe7d8, #c6d7c2);
    }
    .feature-card:nth-child(4n) {
        background: linear-gradient(135deg, #edf3d3, #d9eb9a);
    }
    .feature-card .feature-id {
        font-size: 0.78rem;
        letter-spacing: 0.08em;
        font-weight: 800;
        color: rgba(17,17,17,0.72);
        text-transform: uppercase;
    }
    .feature-card .feature-title {
        font-size: 1.1rem;
        font-weight: 800;
        color: #111111;
        margin: 0.2rem 0;
    }
    .feature-card .feature-copy {
        font-size: 0.82rem;
        color: rgba(17,17,17,0.74);
        line-height: 1.4;
    }
    .metric-card {
        background: linear-gradient(135deg, #111827 0%, #1f2937 100%);
        border: 1px solid #374151;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4);
    }
    .metric-val {
        font-size: 2.2rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-green {
        background-color: #059669;
        color: #ecfdf5;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
    }
    .badge-red {
        background-color: #dc2626;
        color: #fef2f2;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
    }
    .badge-yellow {
        background-color: #d97706;
        color: #fffbeb;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
    }
    .evp-alert {
        background: linear-gradient(90deg, #dc2626 0%, #991b1b 100%);
        color: white;
        padding: 14px 20px;
        border-radius: 10px;
        font-weight: bold;
        font-size: 1.1rem;
        margin-bottom: 20px;
        animation: pulse 2s infinite;
    }
    .priority-vehicles {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 12px;
        margin: 8px 0 22px;
    }
    .priority-vehicle {
        position: relative;
        overflow: hidden;
        min-height: 112px;
        padding: 14px 16px;
        border: 1px solid #334155;
        border-radius: 12px;
        background: linear-gradient(145deg, #111827 0%, #172033 100%);
    }
    .priority-vehicle.active {
        border-color: #34d399;
        box-shadow: 0 0 0 1px rgba(52, 211, 153, 0.22), 0 0 24px rgba(52, 211, 153, 0.14);
    }
    .priority-vehicle.queued {
        border-color: #fbbf24;
        box-shadow: 0 0 18px rgba(251, 191, 36, 0.10);
    }
    .vehicle-icon {
        width: 48px;
        height: 34px;
        float: right;
        color: #64748b;
    }
    .active .vehicle-icon { color: #34d399; animation: vehicle-pulse 1.25s ease-in-out infinite; }
    .queued .vehicle-icon { color: #fbbf24; }
    .vehicle-kind { color: #cbd5e1; font-size: 0.72rem; letter-spacing: 0.12em; font-weight: 700; }
    .vehicle-state { margin-top: 14px; color: #94a3b8; font-size: 0.82rem; }
    .active .vehicle-state { color: #6ee7b7; font-weight: 700; }
    .queued .vehicle-state { color: #fcd34d; font-weight: 700; }
    @keyframes vehicle-pulse { 50% { transform: translateX(4px); filter: drop-shadow(0 0 7px currentColor); } }
    .stApp {
        background-color: #f3f6f2;
        color: #14231f;
    }
    .main {
        background-color: transparent;
        background-image:
            linear-gradient(rgba(21, 44, 35, 0.035) 1px, transparent 1px),
            linear-gradient(90deg, rgba(21, 44, 35, 0.035) 1px, transparent 1px);
        background-size: 48px 48px;
    }
    .block-container {
        max-width: 1480px !important;
        padding-top: 1.1rem !important;
    }
    [data-testid="stSidebar"] {
        background: #eaf0e9;
        border-right: 1px solid rgba(20, 43, 34, 0.1);
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #58665e;
    }
    .sidebar-brand, .brandbar, .brandmark, .brand-status {
        display: flex;
        align-items: center;
    }
    .sidebar-brand {
        gap: 0.7rem;
        margin: 0.25rem 0 0.5rem;
    }
    .sidebar-brand-icon, .brandmark-icon {
        display: grid;
        place-items: center;
        flex: 0 0 auto;
        width: 38px;
        height: 38px;
        border-radius: 50%;
        background: #10241f;
        color: #b7f34a;
        font-size: 1.65rem;
        line-height: 1;
        box-shadow: inset 0 0 0 4px #eaf0e9, 0 0 0 1px #10241f;
    }
    .sidebar-brand strong, .brandmark strong {
        display: block;
        color: #10241f;
        font-size: 1.14rem;
        letter-spacing: 0.12em;
        line-height: 1.1;
    }
    .sidebar-brand small, .brandmark small {
        display: block;
        margin-top: 0.2rem;
        color: #68766d;
        font-size: 0.58rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
    }
    .brandbar {
        justify-content: space-between;
        padding: 0.45rem 0.2rem 0.8rem;
        border-bottom: 1px solid rgba(20, 43, 34, 0.1);
    }
    .brandmark { gap: 0.75rem; }
    .brandmark-icon {
        width: 42px;
        height: 42px;
        box-shadow: inset 0 0 0 4px #f3f6f2, 0 0 0 1px #10241f;
    }
    .brandmark strong { font-size: 1.3rem; }
    .brand-status {
        gap: 0.5rem;
        padding: 0.5rem 0.8rem;
        border: 1px solid rgba(20, 43, 34, 0.12);
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.72);
        color: #344a3e;
        font-size: 0.67rem;
        font-weight: 700;
        letter-spacing: 0.09em;
    }
    .brand-status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #76bd21;
        box-shadow: 0 0 0 4px rgba(118, 189, 33, 0.14);
    }
    .page-shell { margin-top: 0.35rem; }
    .hero-shell {
        min-height: 440px;
        padding: 1.8rem 0.2rem 1.2rem;
        background:
            radial-gradient(circle at 0% 65%, rgba(181, 243, 74, 0.28), transparent 19rem),
            radial-gradient(circle at 100% 35%, rgba(83, 207, 229, 0.2), transparent 20rem);
    }
    .hero-copy h1 {
        max-width: 700px;
        color: #10241f;
        font-size: clamp(3.2rem, 5.5vw, 5.5rem);
        letter-spacing: -0.065em;
    }
    .hero-copy .highlight { color: #70b51d; }
    .hero-copy .subtitle {
        max-width: 560px;
        color: #65736b;
        font-size: 1.04rem;
    }
    .eyebrow {
        background: rgba(255, 255, 255, 0.7);
        border-color: rgba(20, 43, 34, 0.12);
        color: #243c30;
    }
    .cta-chip {
        background: #10241f;
        color: #f6f8f5;
        border-radius: 999px;
        letter-spacing: 0.015em;
        text-transform: none;
    }
    .cta-chip.secondary {
        background: #b8f04e;
        color: #14231f;
    }
    .hero-panel {
        border-radius: 26px;
        box-shadow: 0 24px 60px rgba(16, 36, 31, 0.16);
    }
    .sim-window { min-height: 350px; }
    .feature-strip { gap: 0.85rem; }
    .feature-card, .feature-card:nth-child(2n), .feature-card:nth-child(3n), .feature-card:nth-child(4n) {
        min-height: 138px;
        border: 1px solid rgba(20, 43, 34, 0.1);
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.78);
        box-shadow: 0 8px 24px rgba(16, 36, 31, 0.045);
    }
    .feature-card .feature-id { color: #77a833; }
    .feature-card .feature-title { color: #14231f; }
    .feature-card .feature-copy { color: #68766d; }
    .metric-card {
        min-height: 142px;
        border: 1px solid rgba(20, 43, 34, 0.1);
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.88);
        box-shadow: 0 10px 28px rgba(16, 36, 31, 0.06);
    }
    .metric-label {
        color: #718078;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.1em;
    }
    .metric-val { color: #14231f; }
    .metric-card div[style*="color: #d1d5db"] { color: #718078 !important; }
    .metric-card div[style*="color: #9ca3af"] { color: #718078 !important; }
    .priority-vehicle {
        border: 1px solid rgba(20, 43, 34, 0.1);
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.86);
        box-shadow: 0 8px 24px rgba(16, 36, 31, 0.045);
    }
    .priority-vehicle.active {
        border-color: #91c84e;
        background: #f5faed;
        box-shadow: 0 0 0 1px rgba(118, 189, 33, 0.12);
    }
    .priority-vehicle.queued { background: #fffaf0; }
    .vehicle-kind { color: #68766d; }
    .vehicle-state { color: #58665e; }
    .evp-alert {
        border: 1px solid rgba(185, 48, 48, 0.2);
        border-radius: 14px;
        background: #fff0ed;
        color: #9e3028;
        box-shadow: 0 8px 24px rgba(126, 38, 30, 0.07);
    }
    .badge-green { background: #e7f5d5; color: #356b16; }
    .badge-yellow { background: #fff2d6; color: #8a5a00; }
    .badge-red { background: #fde8e5; color: #a33c31; }
    .lane-feed-card {
        min-height: 177px;
        margin-bottom: 0.65rem;
        padding: 1rem;
        border: 1px solid rgba(20, 43, 34, 0.1);
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.9);
        box-shadow: 0 8px 24px rgba(16, 36, 31, 0.045);
    }
    .lane-feed-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 0.5rem;
        color: #14231f;
        font-size: 0.9rem;
    }
    .lane-feed-header .badge-green,
    .lane-feed-header .badge-yellow,
    .lane-feed-header .badge-red {
        padding: 0.32rem 0.55rem;
        border-radius: 999px;
        font-size: 0.65rem;
        letter-spacing: 0.035em;
        white-space: nowrap;
    }
    .lane-feed-priority {
        margin: 0.42rem 0 0.75rem;
        color: #77847b;
        font-size: 0.72rem;
    }
    .lane-feed-stat {
        display: flex;
        justify-content: space-between;
        gap: 0.5rem;
        padding: 0.34rem 0;
        border-top: 1px solid rgba(20, 43, 34, 0.07);
        color: #68766d;
        font-size: 0.72rem;
    }
    .lane-feed-stat b { color: #243c30; white-space: nowrap; }
    .lane-feed-stat small { color: #7b8980; font-size: 0.66rem; font-weight: 500; }
    .stButton > button {
        min-height: 2.6rem;
        border: 1px solid rgba(20, 43, 34, 0.14);
        border-radius: 999px;
        background: #10241f;
        color: #f7faf5;
        font-weight: 650;
        transition: background 160ms ease, transform 160ms ease;
    }
    .stButton > button:hover {
        border-color: #10241f;
        background: #263d32;
        color: white;
        transform: translateY(-1px);
    }
    .stButton > button:focus:not(:active) {
        border-color: #6ea927;
        color: #10241f;
        box-shadow: 0 0 0 0.15rem rgba(118, 189, 33, 0.2);
    }
    [data-testid="stTabs"] button[role="tab"] {
        color: #64736a;
        font-weight: 650;
    }
    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        color: #294c20;
        border-bottom-color: #80bd2c;
    }
    h1, h2, h3 { color: #14231f; letter-spacing: -0.035em; }
    [data-testid="stImage"] img {
        border: 1px solid rgba(20, 43, 34, 0.1);
        border-radius: 14px;
    }
    @media (max-width: 760px) {
        .priority-vehicles { grid-template-columns: 1fr; }
        .hero-shell { grid-template-columns: 1fr; padding-top: 1.2rem; }
        .feature-strip { grid-template-columns: 1fr 1fr; }
        .hero-panel { min-height: 0; }
    }
    @media (max-width: 520px) {
        .feature-strip { grid-template-columns: 1fr; }
        .cta-row { flex-direction: row; }
        .brand-status { font-size: 0; padding: 0.6rem; }
        .brand-status-dot { width: 9px; height: 9px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_system_instances():
    """Initializes and persists system singletons across Streamlit re-runs."""
    db = DatabaseManager(DB_PATH)
    controller = SignalPriorityController(db_manager=db)
    carbon_engine = CarbonOffsetEngine(db_manager=db)
    sim_gen = SyntheticIntersectionGenerator()
    siren_monitor = AcousticSirenMonitor()
    vision_engine = VehicleDetectionEngine()
    return db, controller, carbon_engine, sim_gen, siren_monitor, vision_engine


db, controller, carbon_engine, sim_gen, siren_monitor, vision_engine = get_system_instances()

# --- SIDEBAR CONTROLS ---
st.sidebar.markdown(
    """
    <div class="sidebar-brand">
      <span class="sidebar-brand-icon">+</span>
      <div><strong>ATCS</strong><small>by Innov Squad</small></div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.caption("AI-powered adaptive traffic management")

st.sidebar.subheader("🚨 Priority Override Actions")
if st.sidebar.button("🚑 Spawn Ambulance", width="stretch"):
    sim_gen.spawn_emergency_vehicle("LANE_N")
    st.session_state["manual_override_lane"] = "LANE_N"
    st.sidebar.success("Ambulance dispatched to North Approach!")

if st.sidebar.button("🔊 Acoustic Siren", width="stretch"):
    st.session_state["trigger_acoustic_siren"] = True
    st.sidebar.warning("Acoustic 1.8 kHz siren pulse simulated!")

if st.sidebar.button("🚘 Activate VIP Corridor", width="stretch"):
    controller.vip_lane_id = "LANE_E"
    st.sidebar.success("VIP priority corridor queued on East Approach!")

if controller.vip_lane_id and st.sidebar.button("✕ Clear VIP Corridor", width="stretch"):
    controller.vip_lane_id = None
    st.sidebar.info("VIP priority corridor cleared.")

if st.sidebar.button("⛔ Test Red-Light Breach", width="stretch"):
    sim_gen.trigger_red_breach("LANE_E")
    st.sidebar.info("Vehicle queued for red violation in East Approach!")

st.sidebar.divider()
st.sidebar.subheader("⚙️ Intersection Parameters")
weather_mode = st.sidebar.selectbox(
    "Camera Weather",
    ["clear", "rain", "fog", "night"],
    format_func=lambda value: value.title(),
)
sim_gen.set_weather(weather_mode)
min_green = st.sidebar.slider("Min Green Time (s)", 5, 20, 10)
max_green = st.sidebar.slider("Max Green Time (s)", 30, 90, 65)
starvation_limit = st.sidebar.slider("Anti-Starvation Threshold (s)", 60, 180, 110)

controller.cfg.min_green_seconds = min_green
controller.cfg.max_green_seconds = max_green
controller.cfg.starvation_max_wait_seconds = starvation_limit

st.sidebar.divider()
st.sidebar.caption("System Status: OPERATIONAL (Edge Accelerated)")

# --- MAIN DASHBOARD HEADER ---
st.markdown(
    """
    <div class="brandbar">
      <div class="brandmark">
        <span class="brandmark-icon">+</span>
        <div><strong>ATCS</strong><small>by Innov Squad</small></div>
      </div>
      <div class="brand-status"><span class="brand-status-dot"></span> LIVE SYSTEM · JUNCTION 04</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="page-shell">
      <div class="hero-shell">
        <div class="hero-copy">
          <div class="eyebrow">Adaptive signal intelligence</div>
          <h1>Signals that <span class="highlight">think ahead</span></h1>
          <div class="subtitle">
            AI-powered traffic orchestration for safer intersections, lower delay, and greener urban mobility.
          </div>
          <div class="cta-row">
            <div class="cta-chip secondary">Live control</div>
            <div class="cta-chip">Autonomous phase logic</div>
          </div>
        </div>
        <div class="hero-panel">
          <div class="panel-top">
            <div class="window-dots"><span></span><span></span><span></span></div>
            <div class="mini-pill">signal board</div>
          </div>
          <div class="sim-window">
            <div class="sim-grid">
              <div class="lane-card"><span class="lane-name">North lane</span><div class="signal"></div></div>
              <div class="lane-card"><span class="lane-name">East lane</span><div class="signal"></div></div>
              <div class="lane-card"><span class="lane-name">West lane</span><div class="signal"></div></div>
              <div class="lane-card"><span class="lane-name">South lane</span><div class="signal"></div></div>
            </div>
          </div>
        </div>
      </div>
      <div class="feature-strip">
        <div class="feature-card">
          <div class="feature-id">01</div>
          <div class="feature-title">Dynamic timing</div>
          <div class="feature-copy">Adaptive green allocation based on live real-time occupancies.</div>
        </div>
        <div class="feature-card">
          <div class="feature-id">02</div>
          <div class="feature-title">Emergency preemption</div>
          <div class="feature-copy">Priority corridors for emergency and VIP routes with instant override logic.</div>
        </div>
        <div class="feature-card">
          <div class="feature-id">03</div>
          <div class="feature-title">Enforcement</div>
          <div class="feature-copy">Virtual stopline detection with ANPR-backed evidence and challan logging.</div>
        </div>
        <div class="feature-card">
          <div class="feature-id">04</div>
          <div class="feature-title">Carbon offset</div>
          <div class="feature-copy">Live fuel and emissions savings tracked against the static baseline.</div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption("Consolidated Architecture // YOLOv8 Computer Vision • FFT Acoustic Siren Sensor • ANPR E-Challan • Live Carbon Offset")

# Check for acoustic siren trigger
acoustic_detected = False
if st.session_state.get("trigger_acoustic_siren", False):
    synthetic_audio = AcousticSirenMonitor.generate_synthetic_siren(duration_s=0.5, siren_type="yelp")
    res = siren_monitor.process_audio_chunk(synthetic_audio)
    acoustic_detected = res.detected
    st.session_state["trigger_acoustic_siren"] = False

# Step the simulation forward
lane_metrics = {}
lane_frames = {}

for lid in ["LANE_N", "LANE_S", "LANE_E", "LANE_W"]:
    sig_state = controller.lane_signal_colors.get(lid, "RED")
    frame = sim_gen.generate_lane_frame(
        lid, sig_state, controller.current_phase_allocated_s, weather=weather_mode
    )
    metrics = vision_engine.detect_and_calculate_occupancy(frame, DEFAULT_LANES[lid])
    lane_metrics[lid] = metrics
    lane_frames[lid] = frame

# Update priority matrix controller
snapshot = controller.update_tick(
    dt_seconds=1.0,
    lane_metrics=lane_metrics,
    acoustic_siren_detected=acoustic_detected,
    acoustic_siren_lane="LANE_N" if acoustic_detected else None,
    manual_override_lane=st.session_state.pop("manual_override_lane", None),
)

# Update Carbon Offset
active_metrics = lane_metrics.get(snapshot.active_lane_id, lane_metrics["LANE_N"])
sust_metrics = carbon_engine.calculate_cycle_savings(
    active_metrics, snapshot.lane_states[snapshot.active_lane_id].current_phase_duration_s
)

# Render Emergency Green Wave Alert if Active
if snapshot.is_emergency_preemption:
    st.markdown(
        f"""
        <div class="evp-alert">
            ⚡ EMERGENCY VEHICLE PREEMPTION (EVP) ACTIVE: Immediate Green Corridor Granted to {snapshot.emergency_lane_id} 
            [Source: {snapshot.preemption_source}] — Conflicting Lanes Locked RED.
        </div>
        """,
        unsafe_allow_html=True,
    )

# --- PRIORITY VEHICLE STATUS ---
ambulance_active = snapshot.is_emergency_preemption
vip_active = (
    controller.vip_lane_id is not None
    and snapshot.active_lane_id == controller.vip_lane_id
    and snapshot.active_phase == "GREEN"
)
vip_queued = controller.vip_lane_id is not None and not vip_active
standard_active = (
    not ambulance_active
    and not vip_active
    and snapshot.active_phase == "GREEN"
)

def vehicle_icon(kind: str) -> str:
    """Returns a compact, consistent vehicle mark for the live status cards."""
    if kind == "ambulance":
        return '<svg class="vehicle-icon" viewBox="0 0 64 40" aria-hidden="true"><path fill="currentColor" d="M6 25h4l5-13h25l8 7h8c3 0 5 2 5 5v7H6v-6Zm10-9-3 8h15v-8H16Zm19 0v8h14l-8-8H35Z"/><circle cx="18" cy="34" r="5" fill="currentColor"/><circle cx="49" cy="34" r="5" fill="currentColor"/><path stroke="#111827" stroke-width="2.5" d="M27 19h9m-4.5-4.5v9"/></svg>'
    if kind == "vip":
        return '<svg class="vehicle-icon" viewBox="0 0 64 40" aria-hidden="true"><path fill="currentColor" d="M6 25h4l5-13h25l8 7h8c3 0 5 2 5 5v7H6v-6Zm10-9-3 8h15v-8H16Zm19 0v8h14l-8-8H35Z"/><circle cx="18" cy="34" r="5" fill="currentColor"/><circle cx="49" cy="34" r="5" fill="currentColor"/><path fill="#111827" d="M25 19h17v5H25z"/></svg>'
    return '<svg class="vehicle-icon" viewBox="0 0 64 40" aria-hidden="true"><path fill="currentColor" d="M6 25h5l5-13h24l8 7h8c3 0 5 2 5 5v7H6v-6Zm12-9-3 8h15v-8H18Zm19 0v8h13l-8-8H37Z"/><circle cx="18" cy="34" r="5" fill="currentColor"/><circle cx="49" cy="34" r="5" fill="currentColor"/></svg>'

def vehicle_card(kind: str, label: str, state: str, detail: str, status_class: str) -> str:
    return f'<div class="priority-vehicle {status_class}">{vehicle_icon(kind)}<div class="vehicle-kind">{label}</div><div class="vehicle-state">{state}</div><div style="font-size:.74rem;color:#64748b;margin-top:4px">{detail}</div></div>'

ambulance_state = "ACTIVE" if ambulance_active else "STANDBY"
ambulance_detail = f"Green corridor: {snapshot.emergency_lane_id}" if ambulance_active else "Awaiting emergency cue"
vip_state = "ACTIVE" if vip_active else "QUEUED" if vip_queued else "STANDBY"
vip_detail = f"Priority lane: {controller.vip_lane_id}" if controller.vip_lane_id else "No VIP route selected"
standard_state = "ACTIVE" if standard_active else "STANDBY"
standard_detail = f"Signal: {snapshot.active_lane_id} / {snapshot.active_phase}"
st.markdown(
    '<div class="priority-vehicles">'
    + vehicle_card("car", "STANDARD CAR", standard_state, standard_detail, "active" if standard_active else "")
    + vehicle_card("ambulance", "AMBULANCE", ambulance_state, ambulance_detail, "active" if ambulance_active else "")
    + vehicle_card("vip", "VIP CONVOY", vip_state, vip_detail, "active" if vip_active else "queued" if vip_queued else "")
    + "</div>",
    unsafe_allow_html=True,
)

# --- TOP STATS METRICS ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Active Signal Phase</div>
            <div class="metric-val" style="color: {'#10b981' if snapshot.active_phase == 'GREEN' else '#f59e0b'};">
                {snapshot.active_lane_id}
            </div>
            <div style="margin-top: 6px; font-weight: 500; color: #d1d5db;">
                Phase: {snapshot.active_phase} | Remaining: {snapshot.lane_states[snapshot.active_lane_id].time_remaining_s:.0f}s
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Urban CO₂ Prevented</div>
            <div class="metric-val" style="color: #34d399;">
                {sust_metrics.cumulative_co2_kg:.2f} <span style="font-size: 1.1rem; color: #9ca3af;">kg</span>
            </div>
            <div style="margin-top: 6px; font-weight: 500; color: #d1d5db;">
                🌳 {sust_metrics.trees_planted_equivalent:.1f} Urban Tree Years Equivalent
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Idle Fuel Conserved</div>
            <div class="metric-val" style="color: #60a5fa;">
                {sust_metrics.cumulative_fuel_liters:.2f} <span style="font-size: 1.1rem; color: #9ca3af;">L</span>
            </div>
            <div style="margin-top: 6px; font-weight: 500; color: #d1d5db;">
                💵 ${sust_metrics.cumulative_money_usd:.2f} Commuter Savings
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:
    total_violations = db.get_total_violations_count()
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">E-Challan Violations</div>
            <div class="metric-val" style="color: #f87171;">
                {total_violations}
            </div>
            <div style="margin-top: 6px; font-weight: 500; color: #d1d5db;">
                Virtual Stopline Breaches Logged
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# --- 4-LANE INTERSECTION MONITORING CARDS ---
st.subheader("4-Way Intersection Real-Time Approach Feeds")
c1, c2, c3, c4 = st.columns(4)

lane_cols = {"LANE_N": c1, "LANE_S": c2, "LANE_E": c3, "LANE_W": c4}

for lid, col in lane_cols.items():
    m = lane_metrics[lid]
    s = snapshot.lane_states[lid]
    color_class = "badge-green" if s.signal_color == "GREEN" else "badge-yellow" if s.signal_color == "YELLOW" else "badge-red"
    
    with col:
        st.markdown(
            f"""
            <div class="lane-feed-card">
                <div class="lane-feed-header">
                    <strong>{DEFAULT_LANES[lid].name}</strong>
                    <span class="{color_class}">{s.signal_color} · {s.time_remaining_s:.0f}s</span>
                </div>
                <div class="lane-feed-priority">Priority level {s.priority_level.value}</div>
                <div class="lane-feed-stat"><span>Vehicles</span><b>{m.vehicle_count} <small>· {m.total_pcu:.1f} PCU</small></b></div>
                <div class="lane-feed-stat"><span>Space occupied</span><b>{m.spatial_occupancy_m2:.1f} m² <small>· {m.occupancy_percent:.0f}%</small></b></div>
                <div class="lane-feed-stat"><span>Wait time</span><b>{s.wait_time_s:.0f}s</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        # Render image
        st.image(lane_frames[lid], channels="BGR", width="stretch")

st.divider()

# --- TABS: E-CHALLANS, ACOUSTIC FFT SPECTRUM, SUSTAINABILITY ---
tab_challan, tab_fft, tab_system = st.tabs([
    "📋 Automated E-Challan Registry",
    "🎙️ Acoustic Siren FFT Spectrogram",
    "📈 Carbon & Traffic Flow Analytics",
])

with tab_challan:
    st.subheader("Automated Law Enforcement & Virtual Stopline Vault")
    recent_challans = db.get_recent_echallans(limit=15)
    if recent_challans:
        df_challan = pd.DataFrame(recent_challans)
        st.dataframe(
            df_challan[[
                "id", "timestamp", "lane_id", "vehicle_type", "license_plate", "fine_amount", "violation_type", "status"
            ]],
            width="stretch",
        )
    else:
        st.info("No red-light violations logged yet. Trigger a breach using the sidebar button to simulate!")

with tab_fft:
    st.subheader("Micro-Array Acoustic Siren Spectral Band Analyzer (1000 Hz – 3000 Hz)")
    st.write(
        "Processes acoustic signals via Fast Fourier Transform (FFT) to confirm emergency sirens "
        "even when cameras are occluded by heavy fog or rain."
    )
    # Generate FFT sample for visualization
    test_siren = AcousticSirenMonitor.generate_synthetic_siren(duration_s=0.5, siren_type="yelp")
    fft_res = detect_siren_frequency(test_siren)

    c_fft1, c_fft2 = st.columns([1, 2])
    with c_fft1:
        st.metric("Siren Detected", "YES" if fft_res.detected else "NO")
        st.metric("Peak Frequency", f"{fft_res.peak_frequency_hz:.1f} Hz")
        st.metric("Signal-to-Noise Ratio (SNR)", f"{fft_res.snr_db:.2f} dB")
        st.metric("Pattern Classification", fft_res.pattern_type)
        st.metric("Detection Confidence", f"{fft_res.confidence * 100:.1f}%")

    with c_fft2:
        # Plot spectrum
        freqs = np.fft.rfftfreq(len(test_siren), d=1.0 / 22050)
        mag = np.abs(np.fft.rfft(test_siren))
        df_spec = pd.DataFrame({"Frequency (Hz)": freqs[:1200], "Magnitude": mag[:1200]})
        st.line_chart(df_spec.set_index("Frequency (Hz)"))

with tab_system:
    st.subheader("Mathematical Sustainability Model & Carbon Accounting")
    st.write(
        "Comparing dynamic adaptive green light cycles against conventional fixed-time baseline (45s static cycle)."
    )
    col_s1, col_s2, col_s3 = st.columns(3)
    col_s1.metric("Cumulative Delay Prevented", f"{sust_metrics.cumulative_delay_hours:.2f} hrs")
    col_s2.metric("Cumulative Fuel Saved", f"{sust_metrics.cumulative_fuel_liters:.2f} Liters")
    col_s3.metric("Cumulative CO2 Prevented", f"{sust_metrics.cumulative_co2_kg:.2f} kg")
