import streamlit as st
import cv2
import tempfile
import os
from surveillance_core import BorderSurveillanceSystem
from ml_module import generate_simulated_data, train_anomaly_model
from eda_module import plot_class_distribution, plot_threat_vs_normal, plot_time_of_day_threats, plot_spatial_heatmap

st.set_page_config(page_title="Aegis Border Surveillance", layout="wide", page_icon="🛡️")

@st.cache_data
def get_ml_data():
    df = generate_simulated_data(800)
    df, _ = train_anomaly_model(df)
    return df

def main():
    st.sidebar.title("🛡️ Aegis Command Center")
    app_mode = st.sidebar.selectbox("Select Module", ["Live Surveillance", "Behavioral Anomaly ML", "Heatmaps & Analytics"])

    if app_mode == "Live Surveillance":
        st.title("🔴 Live Border Feed Analysis")
        st.write("Upload a drone/CCTV video feed to test real-time Intrusion & Loitering detection.")
        
        video_file_buffer = st.file_uploader("Upload Video", type=['mp4', 'mov', 'avi'])
        tffile = tempfile.NamedTemporaryFile(delete=False)
        
        if not video_file_buffer:
            st.info("Please upload a video to start the surveillance feed. (E.g. a video of people walking/cars driving)")
        else:
            tffile.write(video_file_buffer.read())
            
            # Initialize System
            surveillance_sys = BorderSurveillanceSystem()
            
            stframe = st.empty()
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write("**Processed Feed**")
                video_placeholder = st.empty()
            with col2:
                st.write("**Real-time Alerts**")
                alerts_placeholder = st.empty()
            
            cap = cv2.VideoCapture(tffile.name)
            
            all_threats = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Resize for optimization
                frame = cv2.resize(frame, (640, 480))
                
                annotated_frame, threats = surveillance_sys.process_frame(frame)
                
                # Update Video
                video_placeholder.image(annotated_frame, channels="BGR")
                
                # Update Alerts Log
                for t in threats:
                    if t not in all_threats:
                        all_threats.append(t)
                
                if len(all_threats) > 0:
                    with alerts_placeholder.container():
                        for tr in all_threats[-5:]: # show last 5
                            if tr['reason'] == 'BORDER CROSSED':
                                st.error(f"🚨 Intrusion: {tr['class'].upper()} (ID: {tr['id']})")
                            else:
                                st.warning(f"⚠️ Loitering: {tr['class'].upper()} (ID: {tr['id']})")
            cap.release()

    elif app_mode == "Behavioral Anomaly ML":
        st.title("🧠 Anomaly Detection Engine")
        st.write("Using Scikit-Learn **Isolation Forest** to separate innocent movement (e.g. Wildlife) from hostile threats based on historical metadata.")
        
        df = get_ml_data()
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Total Logged Events", len(df))
            st.metric("Detected Threats", len(df[df['threat_level'] == 1]))
            
            st.write("### Model Sample Output")
            st.dataframe(df[['object_class', 'velocity_kmh', 'time_in_zone_sec', 'threat_level']].head(10))
            
        with col2:
            st.pyplot(plot_threat_vs_normal(df))
            
        st.write("---")
        st.pyplot(plot_time_of_day_threats(df))

    elif app_mode == "Heatmaps & Analytics":
        st.title("📊 Spatial Analytics & Heatmaps")
        st.write("Exploratory Data Analysis over border incident data to formulate risk profiles for different sectors.")
        
        df = get_ml_data()
        
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(plot_class_distribution(df))
        with col2:
            st.pyplot(plot_spatial_heatmap(df))
            
        st.write("### Vulnerability Matrix by Sector")
        sector_stats = df[df['threat_level'] == 1].groupby('sector').size().reset_index(name='Intrusion Count')
        st.table(sector_stats)

if __name__ == "__main__":
    main()
