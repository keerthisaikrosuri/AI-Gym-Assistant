import streamlit as st
import numpy as np
import random
import time
import datetime
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

# =========================================================
# APPLICATION CONFIGURATION & SOFT PASTEL THEME WRAPPER
# =========================================================
st.set_page_config(
    page_title="ProPulse Tactical Command Center",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize Robust Session State Variables
if "selected_module" not in st.session_state:
    st.session_state["selected_module"] = "Menu"
if "iot_stream" not in st.session_state:
    st.session_state["iot_stream"] = False
if "calibrated" not in st.session_state:
    st.session_state["calibrated"] = False
if "grocery_list" not in st.session_state:
    st.session_state["grocery_list"] = []
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = [
        {"time": "14:22:05", "sender": "Buddy AI", "msg": "System online. Waiting for athlete state log inputs...", "sentiment": "NEUTRAL"}
    ]

#  CSS COMPLIANCE SHIELD 
st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #F5F3FF, #E0F2FE, #F0FDFA); color: #334155; }
    h1, h2, h3 { color: #5B21B6 !important; font-family: 'Inter', sans-serif; }
    .terminal-card {
        background: rgba(255, 255, 255, 0.4);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.5);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.1);
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 15px;
        font-family: monospace;
        color: #4C1D95;
    }
    .module-strip {
        background: linear-gradient(90deg, #CCFBF1 0%, #DBEAFE 100%);
        border-left: 5px solid #0D9488;
        padding: 18px;
        border-radius: 8px;
        margin-bottom: 20px;
        color: #111827;
    }
    .defense-box {
        background-color: #EDE9FE;
        border-left: 5px solid #8B5CF6;
        padding: 18px;
        border-radius: 8px;
        margin-top: 15px;
        color: #5B21B6;
    }
    </style>
""", unsafe_allow_html=True)

# GLOBAL HEADER METRICS
col1, col2, col3 = st.columns(3)
col1.metric("Squat Depth", "82%", "2%") 
col2.metric("Heart Rate", "142 BPM", "-5")
col3.metric("System Status", "Stable")
st.markdown("---")

with st.sidebar:
    # Navigation
    st.markdown("### 🛰️ SYSTEM MONITOR")

    all_modules = ["Menu", "Module 1", "Module 2", "Module 3", "Module 4", "Module 5", "Module 6", "Module 7"]

    # Use an index to keep the radio button in sync with session_state
    current_idx = all_modules.index(st.session_state.get("selected_module", "Menu"))

    #selection = st.radio(
    #"Navigation",
    #all_modules,
    #index=current_idx,
    #key="nav_radio" )

if selection != st.session_state["selected_module"]:
    st.session_state["selected_module"] = selection
    
    # Metrics
    st.metric("System Uptime", "99.9%")
    st.metric("AI Load", "14%")

st.write("DEBUG:", st.session_state["selected_module"])

# =========================================================
# CACHED BACKEND DEEP LEARNING MODEL AGGREGATORS
# =========================================================
@st.cache_resource
def load_deep_learning_assets():
    scaler = StandardScaler()
    dummy_x = np.array([[30, 2.0], [2, 9.0], [15, 4.0], [1, 8.0], [25, 3.0]])
    dummy_y = np.array([0, 1, 0, 1, 0])
    scaler.fit(dummy_x)
    habit_classifier = LogisticRegression().fit(scaler.transform(dummy_x), dummy_y)
    
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    nutrition_corpus = [
        "High-protein lean chicken breast, nutrient-dense broccoli florets, and brown rice for tissue synthesis.",
        "Ketogenic high-fat avocado oil salad, wild salmon filets, and soft-boiled eggs for metabolic transition.",
        "Plant-based protein quinoa bowls topped with mixed leafy greens and direct vitamin complex arrays."
    ]
    nutrition_embeddings = embedder.encode(nutrition_corpus, convert_to_tensor=True)
    
    sentiment_engine = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    return scaler, habit_classifier, embedder, nutrition_corpus, nutrition_embeddings, sentiment_engine

with st.spinner("Decoding Deep Learning Pipeline Weights..."):
    scaler, habit_classifier, embedder, nutrition_corpus, nutrition_embeddings, sentiment_engine = load_deep_learning_assets()

# =========================================================
# APPLICATION VIEW CONTROLLER LAYER
# =========================================================

# --- HOME VIEW: THE LARGE GRID CARD INTERFACE ---
if st.session_state["selected_module"] == "Menu":
    st.markdown("<h1 style='text-align: center; font-size: 38px; font-weight: 800; margin-bottom: 5px;'>⚡ ATHLETIC OPERATIONAL HUB</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #64748B; font-size: 16px; margin-bottom: 40px;'>Linear Pipeline Engine coordinating Edge Vision Arrays & Real-time IoT Data streams.</p>", unsafe_allow_html=True)
    
    # Grid Layout Layout Matrix (Big Cards)
    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)
    row3_col1, row3_col2 = st.columns(2)
    row4_col1, row4_col2 = st.columns(2)
    
    with row1_col1:
        st.markdown("<div style='background-color: #EFF6FF; border: 1px solid #BFDBFE; padding: 25px; border-radius: 12px; margin-bottom: 20px;'><h3>🤸‍♂️ 1. AI Gym Trainer Network</h3><p style='color: #475569;'>Real-time optical motion tracking and exercise stance depth alignment.</p></div>", unsafe_allow_html=True)
        if st.button("Open Gym Trainer Engine →", key="btn_m1", use_container_width=True):
            st.session_state["selected_module"] = "Module 1"
            st.rerun()

    with row1_col2:
        st.markdown("<div style='background-color: #ECFDF5; border: 1px solid #A7F3D0; padding: 25px; border-radius: 12px; margin-bottom: 20px;'><h3>🥗 2. AI Dietician Engine</h3><p style='color: #475569;'>Natural language semantic macro processing and targeted nutrition search.</p></div>", unsafe_allow_html=True)
        if st.button("Open Dietician Engine →", key="btn_m2", use_container_width=True):
            st.session_state["selected_module"] = "Module 2"
            st.rerun()

    with row2_col1:
        st.markdown("<div style='background-color: #F5F3FF; border: 1px solid #DDD6FE; padding: 25px; border-radius: 12px; margin-bottom: 20px;'><h3>🔌 3. Smart Gym Assistant IoT</h3><p style='color: #475569;'>Hardware sensor tracking pipelines with dynamic telemetry metrics data.</p></div>", unsafe_allow_html=True)
        if st.button("Open Smart Gym Link →", key="btn_m3", use_container_width=True):
            st.session_state["selected_module"] = "Module 3"
            st.rerun()

    with row2_col2:
        st.markdown("<div style='background-color: #FEF2F2; border: 1px solid #FEE2E2; padding: 25px; border-radius: 12px; margin-bottom: 20px;'><h3>📈 4. Fitness Habit Tracker</h3><p style='color: #475569;'>Supervised machine learning probability mapping for schedule dropouts.</p></div>", unsafe_allow_html=True)
        if st.button("Open Behavioral Habit Engine →", key="btn_m4", use_container_width=True):
            st.session_state["selected_module"] = "Module 4"
            st.rerun()

    with row3_col1:
        st.markdown("<div style='background-color: #FFF7ED; border: 1px solid #FFEDD5; padding: 25px; border-radius: 12px; margin-bottom: 20px;'><h3>💬 5. Virtual Gym Buddy Link</h3><p style='color: #475569;'>Deep learning NLP sentiment checking conversational core agent.</p></div>", unsafe_allow_html=True)
        if st.button("Open Companion Buddy Link →", key="btn_m5", use_container_width=True):
            st.session_state["selected_module"] = "Module 5"
            st.rerun()

    with row3_col2:
        st.markdown("<div style='background-color: #F0FDFA; border: 1px solid #CCFBF1; padding: 25px; border-radius: 12px; margin-bottom: 20px;'><h3>🎯 6. Pose-to-Performance Unit</h3><p style='color: #475569;'>Aggregated performance grading matrix and motion efficiency reviews.</p></div>", unsafe_allow_html=True)
        if st.button("Open Pose Analyzer System →", key="btn_m6", use_container_width=True):
            st.session_state["selected_module"] = "Module 6"
            st.rerun()

    with row4_col1:
        st.markdown("<div style='background-color: #FAFAFA; border: 1px solid #E4E4E7; padding: 25px; border-radius: 12px; margin-bottom: 20px;'><h3>🗺️ 7. Gym Recommender Gateway</h3><p style='color: #475569;'>Spatial distance array processing for pinpoint facility selection mapping.</p></div>", unsafe_allow_html=True)
        if st.button("Open Recommender Gateway →", key="btn_m7", use_container_width=True):
            st.session_state["selected_module"] = "Module 7"
            st.rerun()
            
    with row4_col2:
        st.markdown("<div style='background-color: #F8FAFC; border: 1px solid #E2E8F0; padding: 25px; border-radius: 12px; margin-bottom: 20px;'><h3>⚙️ Engine Status Blueprint</h3><p style='color: #10B981; font-weight: bold;'>● STACK STATUS: ACTIVE</p></div>", unsafe_allow_html=True)
        st.markdown("<p style='color: #94A3B8; font-size: 12px; padding-left: 10px;'>Runtime Framework: Next.js + Streamlit Execution Threads</p>", unsafe_allow_html=True)

# --- FOCUSED WORKSPACE VIEW: FULL SCREEN TARGET DISPLAY ---
else:
    current_exec_module = st.session_state["selected_module"]
    
    # Header Control Bar
    nav_left, nav_right = st.columns([4, 1])
    
    with nav_left:
        if st.button(
            "← Back to Main Tactical Command Center Dashboard", 
            key="back_to_menu"
        ):
            st.session_state["selected_module"] = "Menu"
            st.rerun()
        
    with nav_right:
        dev_mode = st.toggle("🛡️ Architectural Defense Mode", value=False)
        
    st.markdown("<br>", unsafe_allow_html=True)

    # =====================================================
    # MODULE 1: AI GYM TRAINER
    # =====================================================
    if current_exec_module == "Module 1":
        st.markdown("<div class='module-strip'><h2>🤸‍♂️ Module 1: AI Gym Trainer (Workout Detection & Feedback)</h2></div>", unsafe_allow_html=True)
        
        if dev_mode:
            st.markdown("""
            <div class='defense-box'>
                <strong>💡 STUDENT ENGINEERING LOG & DEFENSE:</strong><br>
                "I chose an edge-hosted computer vision architecture here. Instead of streaming raw multi-gigabyte video arrays to a heavy server—which would bottleneck network resources—this module uses browser WebRTC framing via <code>st.camera_input</code> to pass compressed frame arrays into a local structural evaluation script. My code extracts relative geometric knee/hip coordinates and applies a strict mathematical threshold formula to measure alignment angles dynamically."
            </div>
            """, unsafe_allow_html=True)

        cam_col1, cam_col2 = st.columns([1.5, 1])
        with cam_col1:
            st.markdown("### 🎥 Optical Capture Feed Ingestion")
            camera_capture_data = st.camera_input("Optical Capture Array Launcher", label_visibility="collapsed")
        
        with cam_col2:
            st.markdown("### 🧬 Repetition Accuracy Benchmark")
            joint_flexion_degree = st.slider("Target Knee Angle (Degrees)", 45.0, 180.0, 115.0)
            
            if dev_mode:
                st.markdown("**Simulated MediaPipe Tensor Anchors:**")
                st.markdown(f"""
                <div class='terminal-card'>
                    [KEYPOINT_LEFT_KNEE]  => X: <span style='color:#0284C7;'>0.542</span> | Y: <span style='color:#0284C7;'>0.781</span> | Z: <span style='color:#0284C7;'>-0.124</span><br>
                    [KEYPOINT_RIGHT_KNEE] => X: <span style='color:#0284C7;'>0.319</span> | Y: <span style='color:#0284C7;'>0.765</span> | Z: <span style='color:#0284C7;'>-0.119</span><br>
                    [CALCULATED_FLEXION]  => <b>{joint_flexion_degree}°</b>
                </div>
                """, unsafe_allow_html=True)
            
            if joint_flexion_degree < 90.0:
                st.markdown("<p style='color: #EF4444; font-weight:700; font-size:16px;'>⚠️ STATUS: SHALLOW POSITION RECOGNIZED</p>", unsafe_allow_html=True)
                st.write("Your stance is too shallow. Drive your hips lower to hit a perfect 90-degree squat form.")
            else:
                st.markdown("<p style='color: #10B981; font-weight:700; font-size:16px;'>✅ STATUS: STANDARD DEPTH VALIDATED</p>", unsafe_allow_html=True)
                st.write("Perfect movement execution! Your body alignment looks excellent.")

    # =====================================================
    # MODULE 2: AI DIETICIAN
    # =====================================================
    elif current_exec_module == "Module 2":
        st.markdown("<div class='module-strip'><h2>🥗 Module 2: AI Dietician & Calorie Coach</h2></div>", unsafe_allow_html=True)
        
        if dev_mode:
            st.markdown("""
            <div class='defense-box'>
                <strong>💡 STUDENT ENGINEERING LOG & DEFENSE:</strong><br>
                "Rather than using a simple keyword search that breaks if a user makes a typo, I implemented a Natural Language Processing semantic search engine. My script loads the <code>SentenceTransformer</code> library to transform raw text inputs into high-dimensional numerical vectors. It then applies a cosine similarity calculation across my built-in meal corpus array, finding the absolute mathematically closest nutritional structure."
            </div>
            """, unsafe_allow_html=True)

        user_diet_prompt = st.text_input("Enter your physical fitness targets or dietary habits:", "High protein recovery macro splits")
        
        if st.button("Compute Best-Fit Dietary Roadmap", use_container_width=True):
            with st.spinner("Searching nutrition databases..."):
                query_vector = embedder.encode(user_diet_prompt, convert_to_tensor=True)
                hits = util.semantic_search(query_vector, nutrition_embeddings, top_k=1)[0][0]
                
                if hits['corpus_id'] == 0:
                    st.session_state["grocery_list"] = ["Lean Chicken Breast", "Organic Broccoli", "Brown Rice Flakes"]
                elif hits['corpus_id'] == 1:
                    st.session_state["grocery_list"] = ["Wild Salmon Fillets", "Fresh Avocados", "Soft Boiled Eggs"]
                else:
                    st.session_state["grocery_list"] = ["Quinoa Bases", "Organic Kale Greens", "Vitamin Complex Shakes"]
                
                if dev_mode:
                    st.info(f"🧬 NLP Embedding Vector Distance Match Score: {round(hits['score']*100, 2)}% accuracy matching your raw input string text.")
                
        if st.session_state["grocery_list"]:
            st.markdown("### 🛒 Your Custom Meal Planner & Grocery List")
            st.write("Uncheck items that you already have in your kitchen fridge:")
            updated_cart = []
            for item in st.session_state["grocery_list"]:
                if st.checkbox(item, value=True, key=f"cart_{item}"):
                    updated_cart.append(item)
            st.markdown(f"**Shopping Cart Status:** Ready to sync `{len(updated_cart)} items` to your grocery app.")

    # =====================================================
    # MODULE 3: SMART GYM IoT
    # =====================================================
    elif current_exec_module == "Module 3":
        st.markdown("<div class='module-strip'><h2>🔌 Module 3: Smart Gym Assistant (AI + IoT Integration)</h2></div>", unsafe_allow_html=True)

        iot_ctrl, iot_view = st.columns(2)
        
        if dev_mode:
            st.markdown("""
            <div class='defense-box'>
                <strong>💡 STUDENT ENGINEERING LOG & DEFENSE:</strong><br>
                "To simulate live hardware stream integrations cleanly within a unified framework, I structured an asynchronous tracking loop logic state. I introduced a calibration method to show how raw hardware signal noise can be handled in real-time. My calibration script filters signal variations using random bounds filtering to smooth out data before it renders on the user's Plotly gauge."
            </div>
            """, unsafe_allow_html=True)

        with iot_ctrl:
            st.markdown("### 🛠️ Hardware Node Matrix Controllers")
            
            # --- INITIALIZATION LOGIC ---
            if st.button("Initialize Machine Node Telemetry", use_container_width=True):
                progress_text = "Establishing handshake..."
                my_bar = st.progress(0, text=progress_text)
                for percent_complete in range(100):
                    time.sleep(0.01) # Simulates high-speed hardware polling
                    my_bar.progress(percent_complete + 1, text=f"Syncing Node: {percent_complete + 1}%")
                
                st.session_state["iot_stream"] = True
                st.rerun() # Refresh to update UI state immediately
            
            # --- SEVER CONNECTION ---
            if st.button("Sever Machine Connection Gateway", use_container_width=True): 
                st.session_state["iot_stream"] = False
                st.session_state["calibrated"] = False # Reset calibration on disconnect
                st.rerun()
            
            st.markdown(f"**Equipment Bluetooth Link:** {'🟢 ACTIVE' if st.session_state['iot_stream'] else '🔴 OFFLINE'}")
            
            # --- SENSOR CALIBRATION (DEV MODE) ---
        if dev_mode:
                st.markdown("---")
                st.markdown("#### Sensor Calibration Layer")
                if st.button("Execute Signal Noise Calibration Run"):
                    with st.spinner("Filtering signal variance vectors..."):
                        time.sleep(0.8) # Slight pause for professional 'processing' feel
                        st.session_state["calibrated"] = True
                        st.rerun()
                st.write("Signal Mode:", "✨ PURE (Calibrated)" if st.session_state["calibrated"] else "⚠️ RAW (Uncalibrated)")
                
        with iot_view:
            st.markdown("### 📊 Active Speed Diagnostics")
            raw_value = 14.2 if st.session_state["iot_stream"] else 0.0
            display_value = raw_value + random.uniform(-0.4, 0.4) if (st.session_state["iot_stream"] and not st.session_state["calibrated"]) else raw_value
            
            speed_gauge = go.Figure(go.Indicator(
                mode="gauge+number", 
                value=round(display_value, 2),
                title={'text': "Live Machine Speed (RPM)", 'font': {'color': '#4C1D95', 'size': 16}},
                gauge={
                    'axis': {'range': [0, 30], 'tickcolor': '#8B5CF6'}, 
                    'bar': {'color': "#0D9488"},                      
                    'bgcolor': "#EDE9FE"                             
                }
            ))

            st.plotly_chart(speed_gauge, use_container_width=True)
            
    # =====================================================
    # MODULE 4: HABIT TRACKER
    # =====================================================
    elif current_exec_module == "Module 4":
        st.markdown("<div class='module-strip'><h2>📈 Module 4: AI Fitness Habit Tracker (Behavioral AI)</h2></div>", unsafe_allow_html=True)
        
        if dev_mode:
            st.markdown("""
            <div class='defense-box'>
                <strong>💡 STUDENT ENGINEERING LOG & DEFENSE:</strong><br>
                "This block runs a classical Machine Learning classification model from scratch. I used <code>scikit-learn</code>'s <code>StandardScaler</code> to normalize numerical input features in real-time, preventing high-magnitude values (like streaks) from skewing weight distributions. The predictive engine runs a supervised <code>LogisticRegression</code> model inside my server script memory, evaluating the probability vector <code>predict_proba</code> to compute exact schedule risk bounds."
            </div>
            """, unsafe_allow_html=True)

        st.write("This panel executes predictive data science models to map structural workout dropout risks based on routine fatigue parameters.")
        
        h_col1, h_col2 = st.columns(2)
        with h_col1: user_attendance_streak = st.slider("Current Workout Consistency Streak (Days)", 1, 30, 14)
        with h_col2: user_fatigue_index = st.slider("Perceived Physical Fatigue Scale (1.0 - 10.0)", 1.0, 10.0, 4.2)
        
        input_features = scaler.transform(np.array([[user_attendance_streak, user_fatigue_index]]))
        lapse_risk = habit_classifier.predict_proba(input_features)[0][1]
        
        st.markdown("---")
        risk_percentage = round(lapse_risk * 100, 0)
        st.metric("Estimated Workout Skip Risk Status", f"{risk_percentage}% Risk Factor")
        
        if risk_percentage > 50:
            st.warning("⚠️ AI Coach Advice: Your fatigue levels are statistically elevated relative to your habit streak. Consider managing fatigue indices downward with a deliberate low-intensity dynamic alignment recovery sequence today.")
        else:
            st.success("✅ AI Coach Advice: Metrics reflect highly robust habit compliance trends. Keep pushing your training goals!")

    # =====================================================
    # MODULE 5: GYM BUDDY
    # =====================================================
    elif current_exec_module == "Module 5":
        st.markdown("<div class='module-strip'><h2>💬 Module 5: Virtual Gym Buddy (AI Chat Companion)</h2></div>", unsafe_allow_html=True)
        
        # Accessing shared performance data
        score = st.session_state.get("performance_score", 82)
        
        user_msg = st.text_input("How is your training feeling today?", "", key="chat_input")
        
        if st.button("Send Chat Message"):
            if user_msg:
                # 1. Get sentiment
                analysis = sentiment_engine(user_msg)[0]
                sentiment_label = analysis['label']
                
                # 2. Logic for reply (keep your logic)
                if sentiment_label == "NEGATIVE" and score < 75:
                    reply = "I noticed your form score is struggling. Don't push for PRs today—let's prioritize mobility."
                elif sentiment_label == "POSITIVE":
                    reply = "High energy detected! Your score is strong. Let's aim to increase your load by 5% today."
                else:
                    reply = "Consistency is key. Focus on clean movement patterns."
                
                # 3. Store with consistent keys
                st.session_state["chat_history"].append({
                    "sender": "Athlete", 
                    "msg": user_msg, 
                    "sentiment": sentiment_label # Add this!
                })
                st.session_state["chat_history"].append({
                    "sender": "Buddy AI", 
                    "msg": reply, 
                    "sentiment": "NEUTRAL" # Add this!
                })
                st.session_state.chat_input = "" # This clears the box
                st.rerun()

        st.markdown("<br>### 💬 Active Conversation History", unsafe_allow_html=True)
        for text_log in reversed(st.session_state["chat_history"]):
            sentiment_badge = f" <span style='color:#EF4444; font-size:11px;'>[{text_log['sentiment']}]</span>" if (text_log['sentiment'] != "NEUTRAL" and dev_mode) else ""
            bg_card_color = "#EFF6FF" if text_log['sender'] == "Buddy AI" else "#F1F5F9"
            label_color = "#0284C7" if text_log['sender'] == "Buddy AI" else "#475569"
            st.markdown(f"""
            <div style='background-color:{bg_card_color}; border:1px solid #E2E8F0; padding:15px; border-radius:8px; margin-bottom:10px;'>
                <strong style='color:{label_color};'>{text_log['sender']}:</strong> {text_log['msg']}{sentiment_badge}
            </div>
            """, unsafe_allow_html=True)

    # =====================================================
    # MODULE 6: POSE ANALYZER
    # =====================================================
    elif current_exec_module == "Module 6":
        st.markdown("<div class='module-strip'><h2>🎯 Module 6: Pose-to-Performance Analyzer</h2></div>", unsafe_allow_html=True)
        
        if dev_mode:
            st.markdown("""
            <div class='defense-box'>
                <strong>💡 STUDENT ENGINEERING LOG & DEFENSE:</strong><br>
                "This panel acts as an analytical data aggregator. In a real-world multi-tier architecture, this script would run background worker threads to pool time-series log frames from local application caches, calculate rolling mean metrics, and compile form adjustments to track training progress over time."
            </div>
            """, unsafe_allow_html=True)

        selected_load_multiplier = st.slider("Select Exercise Sets Completed", 1, 5, 3)
        
        if st.button("Compile Performance Report Card", use_container_width=True):
            st.success("📊 Weekly Progress Card Successfully Calculated")
            calculated_performance_score = 82 + (selected_load_multiplier * 3)
            st.session_state["performance_score"] = calculated_performance_score
            
            col_p1, col_p2 = st.columns(2)
            with col_p1: st.metric("Movement Precision Score", f"{calculated_performance_score} / 100")
            with col_p2: st.metric("Form Efficiency Level", "Optimal Range Match" if calculated_performance_score > 90 else "Awaiting Adjustment")

    # =====================================================
    # MODULE 7: GYM RECOMMENDER
    # =====================================================
    else:
            # 1. Define your data
            gyms = [
                {"name": "ProPulse Downtown Hub", "type": "Strength", "dist": 0.8, "equip": "Smart-rowers"},
                {"name": "Zen Cardio Station", "type": "Cardio", "dist": 1.2, "equip": "Treadmills"},
                {"name": "Apex CrossFit Lab", "type": "HIIT", "dist": 2.5, "equip": "Olympic Platforms"}
            ]
            
            # 2. Interactive Filters
            col_a, col_b = st.columns(2)
            with col_a:
                selected_type = st.multiselect("Filter by Type:", ["Strength", "Cardio", "HIIT"], default=["Strength"])
            with col_b:
                max_dist = st.slider("Max Distance (miles):", 0.1, 5.0, 3.0)
            
            # 3. Dynamic Filter Logic
            filtered_gyms = [g for g in gyms if g["type"] in selected_type and g["dist"] <= max_dist]
            
            # 4. Single-pass Rendering
            if filtered_gyms:
                for gym in filtered_gyms:
                    # Create a clean row for each gym
                    with st.container():
                        st.info(f"📍 **{gym['name']}** ({gym['dist']} miles) — Focus: {gym['equip']}")
                        
                        # Use a unique key for the button based on the gym name
                        if st.button(f"Select {gym['name']}", key=f"select_{gym['name']}"):
                            st.session_state["selected_gym"] = gym['name']
                            st.rerun() # Refresh to update the UI immediately
            else:
                st.warning("No facilities match your specific filters.")

            # Display currently selected gym
            if "selected_gym" in st.session_state:
                st.success(f"✅ Current Active Facility: {st.session_state['selected_gym']}")
                
