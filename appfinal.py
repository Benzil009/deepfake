import streamlit as st
import cv2
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
from transformers import AutoImageProcessor, AutoModelForImageClassification, pipeline
from PIL import Image

# Set page config for better layout
st.set_page_config(layout="wide", page_title="DeepGuard AI", page_icon="🛡️")

# Custom CSS for improved UI - enhanced for better visibility and contrast
st.markdown("""
    <style>
    /* Main theme colors - stronger blue palette with better contrast */
    :root {
        --primary-color: #1565C0;
        --secondary-color: #0D47A1;
        --accent-color: #42A5F5;
        --background-color: #FFFFFF;
        --card-bg-color: #F5F7FA;
        --text-color: #212529;
        --card-border: #E0E0E0;
    }
      
    /* Cards */
    .card {
            
        background-color: var(--card-bg-color);
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        border: 1px solid var(--card-border);
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    /* Improved metric cards with better contrast */
    .metric-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border: 1px solid var(--card-border);
        box-shadow: 0 3px 10px rgba(0,0,0,0.08);
        text-align: center;
        transition: transform 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    /* Section headers */
    .section-header {
        margin: 2rem 0 1.2rem 0;
        padding: 0.8rem 0;
        border-bottom: 3px solid var(--accent-color);
    }
    
    /* Upload area */
    .uploadArea {
        background-color: #F8F9FA;
        border-radius: 8px;
        border: 2px dashed #CCCCCC;
        padding: 2.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .uploadArea:hover {
        border-color: var(--accent-color);
        background-color: #F0F4F8;
    }
    
    /* Verdict banner */
    .verdict-banner {
        text-align: center;
        padding: 2rem;
        color: white;
        border-radius: 8px;
        margin-top: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #6c757d;
        margin-top: 3rem;
        border-top: 1px solid #e9ecef;
    }
    
    /* Fix for Streamlit's default styles to improve text visibility */
    .stText p, .stText span {
        color: var(--text-color) !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #0D47A1 !important;
        font-weight: 600 !important;
    }
    
    /* Improved metric display */
    .metric-value {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: var(--primary-color) !important;
        margin: 0.5rem 0 !important;
    }
    
    .metric-label {
        font-size: 1rem !important;
        font-weight: 500 !important;
        color: #495057 !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Button improvements */
    div.stButton > button {
        background-color: var(--primary-color);
        color: white;
        border: none;
        padding: 0.7rem 1.2rem;
        font-weight: 500;
        border-radius: 4px;
        transition: all 0.2s ease;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    div.stButton > button:hover {
        background-color: var(--secondary-color);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        transform: translateY(-2px);
    }
    
    /* Fix tabs */
    .stTabs [data-baseweb="tab"] {
        height: auto;
        white-space: normal;
        padding: 0.7rem 1.2rem;
        font-weight: 500;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        margin-bottom: 1rem;
    }
    
    /* Image frames display - LARGER SIZE */
    .frame-container img {
        min-width: 350px !important;
        max-width: 100% !important;
        border-radius: 4px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 0.5rem;
        border: 1px solid #e0e0e0;
    }
    
    /* Add some hover effects to frames */
    .frame-container img:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        transition: all 0.3s ease;
    }
    
    /* Better styling for charts */
    .stPlotlyChart {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource()
def load_model():
    processor = AutoImageProcessor.from_pretrained("prithivMLmods/Deep-Fake-Detector-Model")
    model = AutoModelForImageClassification.from_pretrained("prithivMLmods/Deep-Fake-Detector-Model")
    return pipeline("image-classification", model=model, feature_extractor=processor)

pipe = load_model()

def calculate_advanced_metrics(real_scores, fake_scores, real_count, fake_count):
    total_frames = len(real_scores)
    metrics = {
        'confidence_volatility': np.std(real_scores + fake_scores),
        'avg_real_confidence': np.mean(real_scores),
        'avg_fake_confidence': np.mean(fake_scores),
        'detection_stability': 1 - (np.std(real_scores) / np.mean(real_scores)) if np.mean(real_scores) != 0 else 0,
        'frame_consistency': abs(real_count - fake_count) / total_frames if total_frames > 0 else 0,
        'deepfake_probability': (fake_count / total_frames) * 100 if total_frames > 0 else 0
    }
    return metrics

def process_video(video_path, frame_skip=10):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    frames = []
    real_scores = []
    fake_scores = []
    frame_indices = []
    real_count, fake_count = 0, 0
    threshold = 0.6

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    frame_classifications = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = pipe(pil_image)
            
            if result:
                real_score = next((x["score"] for x in result if x["label"] == "Real"), 0)
                fake_score = next((x["score"] for x in result if x["label"] == "Fake"), 0)

                real_scores.append(real_score)
                fake_scores.append(fake_score)
                frame_indices.append(frame_count)

                label = "Real" if real_score > fake_score and real_score >= threshold else "Fake"
                score = max(real_score, fake_score)
                color = (0, 255, 0) if label == "Real" else (0, 0, 255)

                # Create improved overlay for better visualization
                overlay = frame.copy()
                cv2.rectangle(overlay, (10, 10), (320, 110), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
                cv2.putText(frame, f"{label}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                cv2.putText(frame, f"Confidence: {score:.2f}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                frame_classifications.append({
                    'frame': frame_count,
                    'label': label,
                    'confidence': score,
                    'timestamp': frame_count/fps if fps > 0 else 0
                })

                # Store higher resolution frames - 450x338 instead of 300x225
                frames.append(cv2.resize(frame, (450, 338)))

                if label == "Real":
                    real_count += 1
                else:
                    fake_count += 1

        frame_count += 1

    cap.release()
    video_metadata = {
        'fps': fps,
        'duration': duration,
        'total_frames': total_frames
    }
    return frames, real_scores, fake_scores, frame_indices, real_count, fake_count, frame_classifications, video_metadata

# Header
st.markdown('<div class="header">', unsafe_allow_html=True)
st.title("🛡️ DeepGuard AI")
st.subheader("Advanced Deepfake Detection Platform")
st.markdown('</div>', unsafe_allow_html=True)

# App description in a card
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("""
### How It Works
DeepGuard AI analyzes video content frame by frame using advanced deep learning models to identify potential
deepfake manipulations. Our technology examines visual inconsistencies, unnatural movements, and digital artifacts 
that are typically invisible to the human eye.
""")
st.markdown('</div>', unsafe_allow_html=True)

# Create columns for upload
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<div class="uploadArea">', unsafe_allow_html=True)
    uploaded_video = st.file_uploader("Upload a video or image for analysis", type=["mp4", "avi", "mov", "png", "jpg", "jpeg"])
    if not uploaded_video:
        st.markdown("Supported formats: MP4, AVI, MOV, PNG, JPG, JPEG")
    st.markdown('</div>', unsafe_allow_html=True)

if uploaded_video:
    # Create columns for video display and info
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="section-header">', unsafe_allow_html=True)
        st.subheader("📼 Input Media")
        st.markdown('</div>', unsafe_allow_html=True)
        
        video_path = "uploaded_video.mp4"
        with open(video_path, "wb") as f:
            f.write(uploaded_video.read())
            
        # Display file info
        file_size = os.path.getsize(video_path) / (1024 * 1024)  # Convert to MB
        st.markdown(f"**File Name:** {uploaded_video.name}")
        st.markdown(f"**File Size:** {file_size:.2f} MB")
        
        # Create a larger video display
        st.video(video_path)
    
    with col2:
        st.markdown('<div class="section-header">', unsafe_allow_html=True)
        st.subheader("🔍 Analysis Controls")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Add some configuration options
        st.markdown('<div class="card">', unsafe_allow_html=True)
        analysis_detail = st.select_slider(
            "Analysis Detail",
            options=["Quick Scan", "Standard", "Detailed"],
            value="Standard"
        )
        
        frame_skip_values = {"Quick Scan": 45, "Standard": 30, "Detailed": 15}
        frame_skip = frame_skip_values[analysis_detail]
        
        st.markdown("#### Processing Options")
        col1a, col1b = st.columns(2)
        with col1a:
            confidence_threshold = st.slider("Confidence Threshold", 0.5, 0.9, 0.6, 0.05)
        with col1b:
            show_advanced = st.checkbox("Show Advanced Metrics", True)
            
        analyze_button = st.button("🔍 Analyze Media", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if analyze_button:
        with st.spinner("🔄 Processing media... This may take a few minutes depending on the content."):
            frames, real_scores, fake_scores, frame_indices, real_count, fake_count, frame_classifications, video_metadata = process_video(video_path, frame_skip=frame_skip)
            advanced_metrics = calculate_advanced_metrics(real_scores, fake_scores, real_count, fake_count)

        # Create a progress indicator
        total_frames = len(real_scores)
        if total_frames > 0:
            real_percentage = (real_count / total_frames) * 100
            fake_percentage = (fake_count / total_frames) * 100
            
        # Results Dashboard
        st.markdown('<div class="section-header">', unsafe_allow_html=True)
        st.subheader("📊 Analysis Results")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Key Metrics in cards with improved visibility
        st.markdown("### Key Indicators")
        metric_cols = st.columns(4)
        
        if total_frames > 0:
            with metric_cols[0]:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown(f'<p class="metric-label">Authenticity Score</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="metric-value">{real_percentage:.1f}%</p>', unsafe_allow_html=True)
                st.markdown(f"<p><strong>{real_count}</strong> frames</p>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
            with metric_cols[1]:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown(f'<p class="metric-label">Manipulation Score</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="metric-value">{fake_percentage:.1f}%</p>', unsafe_allow_html=True)
                st.markdown(f"<p><strong>{fake_count}</strong> frames</p>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
            with metric_cols[2]:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown(f'<p class="metric-label">Detection Confidence</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="metric-value">{advanced_metrics["avg_real_confidence"]*100:.1f}%</p>', unsafe_allow_html=True)
                st.markdown('<p>Average confidence level</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
            with metric_cols[3]:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown(f'<p class="metric-label">Analysis Reliability</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="metric-value">{advanced_metrics["detection_stability"]*100:.1f}%</p>', unsafe_allow_html=True)
                st.markdown('<p>Consistency score</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

        # Sample Frames Analysis
        st.markdown('<div class="section-header">', unsafe_allow_html=True)
        st.subheader("🖼️ Visual Analysis")
        st.markdown('</div>', unsafe_allow_html=True)
        
        frame_tabs = st.tabs(["Sample Frames", "Timeline Analysis", "Confidence Distribution"])
        
        with frame_tabs[0]:
            if len(frames) > 0:
                # Display frames with larger size and better styling
                st.markdown('<div class="frame-container">', unsafe_allow_html=True)
                st.image(frames[:5], channels="BGR", caption=[f"Frame {i+1}" for i in range(min(5, len(frames)))], width=450)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("No frames were processed. Try a different file or analysis settings.")
            
        with frame_tabs[1]:
            # Timeline visualization with improved styling
            timeline_data = pd.DataFrame(frame_classifications)
            if not timeline_data.empty:
                fig = px.scatter(timeline_data, x='timestamp', y='confidence',
                               color='label', title='Detection Confidence Over Time',
                               color_discrete_map={'Real': '#1565C0', 'Fake': '#E53935'},
                               labels={'timestamp': 'Time (seconds)', 'confidence': 'Confidence Score'})
                
                fig.update_layout(
                    plot_bgcolor='rgba(250,250,250,0.9)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(family="Arial", size=12),
                    height=450,  # Increased height
                    hovermode='closest',
                    margin=dict(l=20, r=20, t=50, b=30)
                )
                
                fig.update_traces(marker=dict(size=10, opacity=0.8, line=dict(width=1, color='white')))
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Timeline visualization requires processed frames.")
                
        with frame_tabs[2]:
            # Confidence distribution with improved styling
            if len(real_scores) > 0 and len(fake_scores) > 0:
                fig = go.Figure()
                
                fig.add_trace(go.Violin(
                    y=real_scores, 
                    box_visible=True, 
                    line_color='#1565C0',
                    fillcolor='rgba(21, 101, 192, 0.5)',
                    opacity=0.6,
                    name='Real Score',
                    side='negative'
                ))
                
                fig.add_trace(go.Violin(
                    y=fake_scores,
                    box_visible=True,
                    line_color='#E53935',
                    fillcolor='rgba(229, 57, 53, 0.5)',
                    opacity=0.6,
                    name='Fake Score',
                    side='positive'
                ))
                
                fig.update_layout(
                    title='Confidence Score Distribution',
                    yaxis_title='Confidence Score',
                    violingap=0, 
                    violingroupgap=0,
                    violinmode='overlay',
                    plot_bgcolor='rgba(250,250,250,0.9)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(family="Arial", size=12),
                    height=450,  # Increased height
                    margin=dict(l=20, r=20, t=50, b=30)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Confidence distribution requires processed frames.")

        if show_advanced and total_frames > 0:
            # Advanced Analytics with improved styling
            st.markdown('<div class="section-header">', unsafe_allow_html=True)
            st.subheader("🔬 Technical Analysis")
            st.markdown('</div>', unsafe_allow_html=True)
            
            advanced_cols = st.columns(2)
            
            with advanced_cols[0]:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("#### Media Metadata")
                st.markdown(f"- **Duration:** {video_metadata['duration']:.2f} seconds")
                st.markdown(f"- **Frame Rate:** {video_metadata['fps']:.1f} FPS")
                st.markdown(f"- **Total Frames:** {video_metadata['total_frames']}")
                st.markdown(f"- **Analyzed Frames:** {total_frames} ({(total_frames/video_metadata['total_frames']*100):.1f}%)")
                st.markdown('</div>', unsafe_allow_html=True)

            with advanced_cols[1]:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("#### Analysis Metrics")
                st.markdown(f"- **Confidence Volatility:** {advanced_metrics['confidence_volatility']:.3f}")
                st.markdown(f"- **Frame Consistency:** {advanced_metrics['frame_consistency']:.3f}")
                st.markdown(f"- **Average Real Confidence:** {advanced_metrics['avg_real_confidence']:.3f}")
                st.markdown(f"- **Average Fake Confidence:** {advanced_metrics['avg_fake_confidence']:.3f}")
                st.markdown('</div>', unsafe_allow_html=True)
                
            # Add a gauge chart for deepfake probability with improved styling
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = advanced_metrics['deepfake_probability'],
                title = {'text': "Deepfake Probability", 'font': {'size': 24, 'color': '#212529'}},
                gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#212529'},
                    'bar': {'color': "rgba(50,50,50,0.2)"},
                    'steps': [
                        {'range': [0, 30], 'color': "#43A047"},
                        {'range': [30, 70], 'color': "#FB8C00"},
                        {'range': [70, 100], 'color': "#E53935"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 3},
                        'thickness': 0.8,
                        'value': advanced_metrics['deepfake_probability']
                    }
                },
                number = {'font': {'size': 40, 'color': '#212529'}, 'suffix': '%'}
            ))
            
            fig.update_layout(
                height=350,  # Increased height
                margin=dict(l=20, r=20, t=50, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
            )
            
            st.plotly_chart(fig, use_container_width=True)

        # Final Verdict with improved styling
        st.markdown('<div class="section-header">', unsafe_allow_html=True)
        st.subheader("🎯 Final Verdict")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if total_frames > 0:
            final_result = "FAKE" if fake_count > (real_count * 1.2) else "REAL"
            verdict_color = "#E53935" if final_result == "FAKE" else "#43A047"
            
            verdict_text = "This content appears to be authentic" if final_result == "REAL" else "This content shows signs of manipulation"
            
            st.markdown(f"""
                <div class="verdict-banner" style="background-color: {verdict_color};">
                    <h2 style="font-size: 2.2rem; margin-bottom: 1rem;">Video Classification: {final_result}</h2>
                    <p style="font-size: 1.2rem; margin-bottom: 0.5rem;">{verdict_text}</p>
                    <p style="font-size: 0.9rem;">Analysis completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Add explanation based on verdict with improved styling
            if final_result == "FAKE":
                st.markdown("""
                    <div class="card">
                    <h4>📋 Explanation</h4>
                    <p style="font-size: 1.05rem; line-height: 1.6;">The analysis found significant inconsistencies across multiple frames that are typical of digitally manipulated content. 
                    These may include unnatural facial movements, lighting inconsistencies, or digital artifacts.</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class="card">
                    <h4>📋 Explanation</h4>
                    <p style="font-size: 1.05rem; line-height: 1.6;">The analysis found consistent patterns typical of authentic content. The confidence scores
                    remain stable across frames, and no significant manipulation markers were detected.</p>
                    </div>
                """, unsafe_allow_html=True)

else:
    # Empty state with feature highlights - improved styling
    st.markdown('<div class="section-header">', unsafe_allow_html=True)
    st.subheader("DeepGuard AI Features")
    st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class="card">
                <h4>🔍 Advanced Detection</h4>
                <p style="line-height: 1.6;">Our AI model analyzes visual inconsistencies, unnatural movements, and digital artifacts that humans typically miss when viewing manipulated content.</p>
            </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
            <div class="card">
                <h4>📊 Comprehensive Analytics</h4>
                <p style="line-height: 1.6;">Get frame-by-frame breakdown with confidence scores, timeline analysis, and visualizations of potential manipulations in your media.</p>
            </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
            <div class="card">
                <h4>🛡️ Secure Processing</h4>
                <p style="line-height: 1.6;">All analysis is performed locally in your browser - your data stays private with no external storage or server processing required.</p>
            </div>
        """, unsafe_allow_html=True)
    
# Footer with improved styling
st.markdown('<div class="footer">', unsafe_allow_html=True)
st.markdown("""
    <p style="font-size: 1.1rem; font-weight: 500;"><strong>DeepGuard AI</strong> - Advanced deepfake detection powered by state-of-the-art computer vision</p>
    <p style="font-size: 0.8rem; margin-top: 0.8rem;">© 2025 DeepGuard AI • All rights reserved</p>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)