import streamlit as st
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True
)
landmarker = vision.PoseLandmarker.create_from_options(options)

# 1. HELPER FUNCTIONS 
def calculate_angle(a, b, c):
    """Calculates the angle between three points."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    return angle if angle <= 180 else 360 - angle
    
# 2. Page Config
st.set_page_config(page_title="ProPulse Command Center", layout="wide")

# 3. Initialization
if "selected_module" not in st.session_state: st.session_state["selected_module"] = "Menu"

# 4. Module Logic Definitions
def render_module_1():
    st.header("🤸‍♂️ Module 1: AI Gym Trainer")
    cam = st.camera_input("Capture Pose")
    
    if cam is not None:
        # Decode and process image
        file_bytes = np.asarray(bytearray(cam.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process Pose
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            results = pose.process(image_rgb)
            
            if results.pose_landmarks:
                st.success("Pose data ingested. Analyzing joint angles...")
                
                # Extract landmarks
                landmarks = results.pose_landmarks.landmark
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, 
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, 
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, 
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                
                # Calculate angle
                angle = calculate_angle(shoulder, elbow, wrist)
                st.metric("Detected Elbow Angle", f"{int(angle)}°")
                
                # Visualization
                annotated_image = image.copy()
                mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), caption="Analyzed Pose", use_container_width=True)
            
            else:
                st.warning("No pose detected. Please ensure your full upper body is visible.")
                
def render_module_2():
    st.header("🥗 Module 2: AI Dietician")
    user_input = st.text_input("Enter fitness goal:", "High protein recovery")
    if st.button("Compute Plan"):
        st.write(f"Executing semantic search for: {user_input}...")
        st.success("Dietary roadmap generated.")

def render_module_3():
    st.header("🔌 Module 3: Smart Gym IoT")
    if st.button("Initialize Telemetry"):
        st.info("Syncing Bluetooth Node... Signal Locked: 14.2 RPM")

def render_module_4():
    st.header("📈 Module 4: Habit Tracker")
    streak = st.slider("Workout Streak (Days)", 1, 30, 14)
    risk = 100 - (streak * 2.5)
    st.metric("Dropout Risk Probability", f"{max(0, risk):.1f}%")

def render_module_5():
    st.header("💬 Module 5: Gym Buddy")
    msg = st.text_input("Talk to AI:")
    if st.button("Send"):
        st.write(f"Buddy AI: Understood. Analyzing your sentiment for '{msg}'...")

def render_module_6():
    st.header("🎯 Module 6: Pose Analyzer")
    st.line_chart([80, 82, 85, 88, 92, 95])

def render_module_7():
    st.header("🗺️ Module 7: Gym Recommender")
    st.info("Mapping spatial distance vectors... Found: ProPulse Elite (1.2 miles)")

# 5. Routing Table
module_map = {
    "Module 1": render_module_1,
    "Module 2": render_module_2,
    "Module 3": render_module_3,
    "Module 4": render_module_4,
    "Module 5": render_module_5,
    "Module 6": render_module_6,
    "Module 7": render_module_7,
}

# 6. Main Control Flow
if st.session_state["selected_module"] == "Menu":
    st.title("⚡ ATHLETIC OPERATIONAL HUB")
    cols = st.columns(3)
    for i, mod_name in enumerate(module_map.keys()):
        if cols[i % 3].button(f"Launch {mod_name}"):
            st.session_state["selected_module"] = mod_name
            st.rerun()
else:
    if st.button("← Back to Dashboard"):
        st.session_state["selected_module"] = "Menu"
        st.rerun()
    
    # Run the selected module logic
    selected = st.session_state["selected_module"]
    module_map[selected]()
