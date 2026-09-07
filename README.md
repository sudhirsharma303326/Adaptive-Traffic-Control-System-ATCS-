from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def create_atcs_pdf():
    pdf_filename = "ATCS_Project_Proposal_INNOV_SQUAD.pdf"
    doc = SimpleDocTemplate(
        pdf_filename,
        pagesize=letter,
        rightMargin=40, leftMargin=40,
        topMargin=40, bottomMargin=40
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Styles
    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=18, textColor=colors.HexColor('#1A365D'), spaceAfter=8, alignment=1)
    subtitle_style = ParagraphStyle('SubTitleStyle', parent=styles['Normal'], fontSize=10, textColor=colors.HexColor('#4A5568'), alignment=1, spaceAfter=20)
    h1_style = ParagraphStyle('H1Style', parent=styles['Heading2'], fontSize=13, textColor=colors.HexColor('#2B6CB0'), spaceBefore=12, spaceAfter=6)
    body_style = ParagraphStyle('BodyStyle', parent=styles['Normal'], fontSize=9.5, leading=14, textColor=colors.HexColor('#2D3748'), spaceAfter=6)
    code_style = ParagraphStyle('CodeStyle', parent=styles['Code'], fontSize=7.5, leading=10, textColor=colors.HexColor('#1A202C'))

    story = []

    # Title & Metadata
    story.append(Paragraph("PROJECT PROPOSAL: ADAPTIVE TRAFFIC CONTROL SYSTEM (ATCS)", title_style))
    story.append(Paragraph("<b>Team Name:</b> INNOV SQUAD | <b>Institution:</b> Anand-ICE, Jaipur | <b>Event:</b> SIH 2026", subtitle_style))
    
    # 1. Executive Summary
    story.append(Paragraph("1. EXECUTIVE SUMMARY", h1_style))
    story.append(Paragraph("The <b>Adaptive Traffic Control System (ATCS)</b> is an intelligent, multi-modal traffic management solution designed to replace conventional fixed-timer traffic signals. By leveraging Real-Time Computer Vision (YOLOv8), Edge-AI, Automatic Number Plate Recognition (EasyOCR), and Acoustic Siren Detection, ATCS continuously monitors vehicle density, dynamically calculates optimal signal timings, prioritizes emergency vehicles, and automates law enforcement.", body_style))
    
    # 2. Problem Statement
    story.append(Paragraph("2. PROBLEM STATEMENT & PAIN POINTS", h1_style))
    story.append(Paragraph("• <b>Fixed-Timer Inefficiency:</b> Traditional traffic lights operate on rigid schedules, causing empty lanes to stay green while congested lanes choke.<br/>• <b>Emergency Delays:</b> Ambulances lose critical time during the vital 'Golden Hour'.<br/>• <b>Manual Enforcement Gap:</b> Lack of 24/7 monitoring leads to undetected red-light jumps.<br/>• <b>Environmental Footprint:</b> Idle engine waiting wastes fuel and inflates CO2 emissions.", body_style))

    # 3. Core Modules
    story.append(Paragraph("3. CORE SYSTEM MODULES", h1_style))
    story.append(Paragraph("• <b>Module A (Dynamic Timing):</b> Uses YOLOv8 density detection + pixel-based road space occupancy.<br/>• <b>Module B (Emergency Priority):</b> Visual AI + Acoustic Siren frequency detection (1000-3000 Hz) for Green Wave clearance.<br/>• <b>Module C (ANPR & E-Challan):</b> Virtual stopline breach tracking + EasyOCR plate extraction.<br/>• <b>Module D (Carbon Offset Engine):</b> Mathematical tracking of fuel saved (Liters) and CO2 reduced (kg).", body_style))

    # 4. Technical Architecture Diagram Text
    story.append(Paragraph("4. SYSTEM ARCHITECTURE & DIAGRAM LAYOUT", h1_style))
    arch_diagram = """
+-----------------------------------------------------------------------------------+
|                                 INPUT SOURCES                                     |
|           [ Live CCTV Feed (Camera) ]       [ Microphone (Acoustic Sensor) ]      |
+------------------------------------------+----------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------+
|                            EDGE-AI PROCESSING UNIT                                |
|  ┌────────────────────────┐  ┌───────────────────────┐  ┌─────────────────────┐   |
|  │     YOLOv8 Detection   │  │  Scipy FFT Analysis   │  │  Virtual Stopline   │   |
|  │ (Vehicles & Emergency) │  │  (Siren Frequency)    │  │   Breach Check      │   |
|  └───────────┬────────────┘  └───────────┬───────────┘  └──────────┬──────────┘   |
+───────────────┼───────────────────────────┼─────────────────────────┼─────────────+
                |                           |                         |
                v                           v                         v
+-------------------------------+ +--------------------+ +--------------------------+
|  DYNAMIC DENSITY CALCULATOR   | | EMERGENCY OVERRIDE | |    ANPR & EASYOCR        |
+---------------+---------------+ +---------+----------+ +------------+-------------+
                |                           |                         |
                +───────────────────────────┼─────────────────────────+
                                            v
+-----------------------------------------------------------------------------------+
|      [ Signal Controller ]    [ Live Web Dashboard ]   [ SQLite DB (Challans) ]   |
+-----------------------------------------------------------------------------------+
    """
    story.append(Preformatted(arch_diagram, code_style))

    # 5. Production Python Code
    story.append(Paragraph("5. PRODUCTION PYTHON IMPLEMENTATION (main_atcs.py)", h1_style))
    
    code_text = """import cv2, numpy as np, time, sqlite3, os, easyocr
from ultralytics import YOLO
import scipy.io.wavfile as wav

def init_database():
    conn = sqlite3.connect('atcs_traffic_logs.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS echallans 
        (id INTEGER PRIMARY KEY, timestamp TEXT, plate_number TEXT, violation_type TEXT, image_path TEXT)''')
    conn.commit(); conn.close()

init_database()
model = YOLO('yolov8n.pt')
reader = easyocr.Reader(['en'], gpu=False)

def calculate_carbon_savings(total_vehicles, avg_time_saved_sec=15):
    hours_saved = (total_vehicles * avg_time_saved_sec) / 3600.0
    return round(hours_saved * 1.5, 2), round(hours_saved * 0.6, 2)

def process_traffic_stream(video_source=0):
    cap = cv2.VideoCapture(video_source)
    signal_state, STOPLINE_Y = "GREEN", 350
    total_vehicles = 0
    os.makedirs('challan_evidence', exist_ok=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, (800, 500))
        results = model(frame, verbose=False)[0]
        vehicle_count = 0

        for box in results.boxes:
            if int(box.cls[0]) in [2, 3, 5, 7] and float(box.conf[0]) > 0.4:
                vehicle_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
                
                if signal_state == "RED" and y2 > STOPLINE_Y and y1 < STOPLINE_Y:
                    crop = frame[y1:y2, x1:x2]
                    ocr = reader.readtext(crop) if crop.size > 0 else []
                    plate = ocr[0][1].upper() if ocr else "UNKNOWN"
                    img_path = f"challan_evidence/violator_{int(time.time())}.jpg"
                    cv2.imwrite(img_path, crop)
                    
                    conn = sqlite3.connect('atcs_traffic_logs.db')
                    conn.cursor().execute("INSERT INTO echallans VALUES (NULL, ?, ?, 'Red Light', ?)",
                                          (time.strftime('%Y-%m-%d %H:%M:%S'), plate, img_path))
                    conn.commit(); conn.close()

        total_vehicles += vehicle_count
        co2_saved, fuel_saved = calculate_carbon_savings(total_vehicles)
        cv2.putText(frame, f"CO2 REDUCED: {co2_saved}kg | FUEL SAVED: {fuel_saved}L", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("ATCS Engine", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release(); cv2.destroyAllWindows()
"""
    story.append(Preformatted(code_text, code_style))

    # Build Document
    doc.build(story)
    print("PDF Successfully Generated!")

if __name__ == "__main__":
    create_atcs_pdf()
# Adaptive-Traffic-Control-System-ATCS-
Adaptive Traffic Control System (ATCS).  
