"""
=============================================================================
ATCS Interactive Quick-Launcher
=============================================================================
Quick launch utility for the AI-Powered Adaptive Traffic Control System:
- Option 1: Live OpenCV 4-Way Multi-Camera HUD Window (Visual display)
- Option 2: Streamlit Web Operations Command Center
- Option 3: High-Speed Terminal Telemetry Mode (Headless / Server)
- Option 4: Automated Verification Test Suite
"""

import sys
import os
import subprocess

PYTHON_EXE = sys.executable

def main():
    print("=" * 70)
    print("       AI-POWERED ADAPTIVE TRAFFIC CONTROL SYSTEM (ATCS)")
    print("   Edge-AI CV • FFT Acoustic Sensing • ANPR E-Challan • Carbon")
    print("=" * 70)
    print("\nSelect execution mode:")
    print("  [1] Live 4-Lane OpenCV HUD Display Window (Interactive GUI)")
    print("  [2] Streamlit Web Operations & Analytics Command Center")
    print("  [3] Live Console Telemetry Stream (Headless Edge Mode)")
    print("  [4] Run Full Automated System Diagnostics & Verification Suite")
    print("  [5] Exit")

    choice = input("\nEnter choice [1-5] (default: 1): ").strip() or "1"

    if choice == "1":
        print("\nStarting ATCS 4-Lane OpenCV HUD Window...")
        print("Note: In the HUD window, press 'A' for Ambulance, 'S' for Siren, 'R' for Red Breach, 'Q' to Quit.\n")
        subprocess.run([PYTHON_EXE, "atcs_system.py", "--mode", "hud"])
    elif choice == "2":
        print("\nLaunching Streamlit Operations Command Center...")
        subprocess.run([PYTHON_EXE, "-m", "streamlit", "run", "ui/streamlit_app.py"])
    elif choice == "3":
        print("\nStarting ATCS in Live Console Telemetry Mode...")
        subprocess.run([PYTHON_EXE, "atcs_system.py", "--mode", "console", "--duration", "60"])
    elif choice == "4":
        print("\nRunning ATCS System Diagnostics Test Suite...")
        subprocess.run([PYTHON_EXE, "-m", "unittest", "test_atcs.py", "-v"])
    else:
        print("\nExiting ATCS.")

if __name__ == "__main__":
    main()
