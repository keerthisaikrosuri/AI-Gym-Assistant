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
# 1. SOFT PASTEL THEME + CONFIGURATION
# =========================================================
st.set_page_config(page_title="ProPulse Tactical Hub", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #FDFCF0; color: #5D5B6A; }
    .big-card {
        background-color: #E8F1F2; border: 2px solid #D1E8E4; padding: 20px;
        border-radius: 20px; text-align: center; transition: transform 0.2s;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05); color: #4A5568; margin-bottom: 10px;
        cursor: pointer;
    }
    .big-card:hover { transform: scale(1.03); background-color: #D1E8E4; }
    .module-strip { background: #E8F1F2; border-left: 4px solid #06B6D4; padding: 15px; border-radius: 0 8px 8px 0; margin-bottom: 12px; }
    .terminal-card { background-color: #F3F4F6; border: 1px solid #E5E7EB; padding: 18px; border-radius: 8px; font-family: monospace; }
    .defense-box { background-color: #FDFCF0; border-left: 5px solid #F59E0B; padding: 15px; border-radius: 4px; margin: 15px 0; border: 1px solid #E5E7EB;}
    </style>
""", unsafe_allow_html=True)

# Session State Initialization
if "active_module" not in st.session_state: st.session_state["active_module"] = "Menu"
if "chat_history" not in st.session_state: st.session_state["chat_history"] = []
if "grocery_list" not in st.session_state: st.session_state["grocery_list"] = []
if "iot_stream" not in st.session_state: st.session_state["iot_stream"] = False

# Backend Assets
@st.cache_resource
def load_deep_learning_assets():
    scaler = StandardScaler()
    dummy_x = np.array([[30, 2.0], [2, 9.0], [15, 4.0]])
    dummy_y = np.array([0, 1, 0])
    scaler.fit(dummy_x)
    habit_classifier = LogisticRegression().fit(scaler.transform(dummy_x), dummy_y)
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    sentiment_engine = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    return scaler, habit_classifier, embedder, sentiment_engine

scaler, habit_classifier, embedder, sentiment_engine = load_deep_learning_assets()

# =========================================================
# 2. INTERFACE LAYER
# =========================================================
if st.session_state["active_module"] == "Menu":
    st.markdown("<h1 style='text-align: center; color: #4A5568;'>⚡ ATHLETIC OPERATIONAL HUB</h1>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    cards = [("🤸‍♂️", "Gym Trainer", "Module 1"), ("🥗", "Dietician", "Module 2"), 
             ("🔌", "IoT Assistant", "Module 3"), ("📈", "Habit Tracker", "Module 4"),
             ("💬", "Gym Buddy", "Module 5"), ("🎯", "Pose Analyzer", "Module 6"),
             ("🗺️", "Recommender", "Module 7")]
    
    for i, (icon, label, mod) in enumerate(cards):
        with [col1, col2, col3][i % 3]:
            st.markdown(f"<div class='big-card'><h2>{icon}</h2><h3>{label}</h3></div>", unsafe_allow_html=True)
            if st.button(f"Open {label}", key=f"btn_{mod}"): 
                st.session_state["active_module"] = mod
                st.rerun()

else:
    if st.button("← Back to Menu"): st.session_state["active_module"] = "Menu"; st.rerun()
    dev_mode = st.toggle("🛡️ Enable Architectural Defense Mode", value=False)
    
    # --- MODULE 1: GYM TRAINER ---
    if st.session_state["active_module"] == "Module 1":
        st.markdown("<div class='module-strip'><h3>Module 1: AI Gym Trainer</h3></div>", unsafe_allow_html=True)
        st.camera_input("Optical Capture Array")
        
    # --- MODULE 2: DIETICIAN ---
    elif st.session_state["active_module"] == "Module 2":
        st.markdown("<div class='module-strip'><h3>Module 2: AI Dietician Engine</h3></div>", unsafe_allow_html=True)
        
    # --- MODULE 3: IoT ---
    elif st.session_state["active_module"] == "Module 3":
        st.markdown("<div class='module-strip'><h3>Module 3: Smart Gym Assistant IoT</h3></div>", unsafe_allow_html=True)
        
    # --- MODULE 4: HABIT TRACKER ---
    elif st.session_state["active_module"] == "Module 4":
        st.markdown("<div class='module-strip'><h3>Module 4: Fitness Habit Tracker</h3></div>", unsafe_allow_html=True)
        
    # --- MODULE 5: GYM BUDDY (UPDATED LOGIC) ---
    elif st.session_state["active_module"] == "Module 5":
        st.markdown("<div class='module-strip'><h3>💬 Virtual Gym Buddy</h3></div>", unsafe_allow_html=True)
        user_input = st.text_input("Tell your AI how your training feels:")
        if st.button("Send Message"):
            if user_input:
                msg_lower = user_input.lower()
                sentiment = sentiment_engine(user_input)[0]['label']
                if any(word in msg_lower for word in ["squat", "knee", "form", "technique", "lift"]):
                    reply = "Focus on keeping your knees tracking over your toes—alignment is everything for that movement!"
                elif sentiment == "NEGATIVE":
                    reply = "I hear you. Let's pivot to mobility and stretching—your long-term progress depends on recovery."
                else:
                    reply = "Fantastic energy! That mindset is perfect for a PR attempt today. Let's get after it!"
                st.session_state["chat_history"].append(("You", user_input))
                st.session_state["chat_history"].append(("Buddy AI", reply))
                st.rerun()
        for sender, msg in st.session_state["chat_history"]:
            st.markdown(f"<div class='terminal-card'>**{sender}:** {msg}</div>", unsafe_allow_html=True)

    # --- MODULE 6: POSE ANALYZER ---
    elif st.session_state["active_module"] == "Module 6":
        st.markdown("<div class='module-strip'><h3>Module 6: Pose-to-Performance Unit</h3></div>", unsafe_allow_html=True)
        
    # --- MODULE 7: RECOMMENDER ---
    elif st.session_state["active_module"] == "Module 7":
        st.markdown("<div class='module-strip'><h3>Module 7: Gym Recommender Gateway</h3></div>", unsafe_allow_html=True)
