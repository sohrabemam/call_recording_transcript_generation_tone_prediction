import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import tempfile
import os
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import Form, File, UploadFile
from typing import Dict, Any, List
import uvicorn
from audio_processor import process_audio_file, aggregate_results, generate_conclusion
from dotenv import load_dotenv
load_dotenv()
st.set_page_config(
    page_title="Call Center Analytics Dashboard",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Call Center Audio Analysis API",
    description="AI-powered sentiment and tone analysis for customer service calls",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Call Center Audio Analysis API", "status": "active"}


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "service": "Call Center Audio Analysis API",
        "version": "1.0.0"
    }


@app.post("/analyze-audio")
async def analyze_audio(
        file: UploadFile = File(...),
        language: str = "hi",
        confidence_threshold: float = 0.7,
        segment_length: int = 10
):
    """
    Analyze uploaded audio file for sentiment and tone

    Args:
        file: WAV audio file
        language: Transcription language (default: "hi" for Hindi)
        confidence_threshold: Confidence threshold for analysis (0.0-1.0)
        segment_length: Segment length in seconds (5-30)

    Returns:
        JSON response with analysis results
    """

    # Validate file type
    if not file.filename.lower().endswith('.wav'):
        raise HTTPException(
            status_code=400,
            detail="Only WAV files are supported"
        )

    # Validate parameters
    if not 0.0 <= confidence_threshold <= 1.0:
        raise HTTPException(
            status_code=400,
            detail="Confidence threshold must be between 0.0 and 1.0"
        )

    if not 5 <= segment_length <= 30:
        raise HTTPException(
            status_code=400,
            detail="Segment length must be between 5 and 30 seconds"
        )

    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            audio_file_path = tmp_file.name
            file_size_mb = len(content) / (1024 * 1024)

        logger.info(f"Processing audio file: {file.filename} ({file_size_mb:.2f} MB)")

        # Process audio file
        results = process_audio_file(audio_file_path)

        if not results:
            raise HTTPException(
                status_code=422,
                detail="No valid transcriptions found in the audio file"
            )

        # Aggregate results
        grouped, overall, df = aggregate_results(results)

        # Generate AI conclusion
        conclusion = generate_conclusion(overall)

        # Prepare response data
        response_data = {
            "status": "success",
            "message": "Audio analysis completed successfully",
            "file_info": {
                "filename": file.filename,
                "size_mb": round(file_size_mb, 2),
                "duration_s": overall.get('duration_s', 0) if overall else 0,
                "segments_count": len(results)
            },
            "analysis": {
                "overall_metrics": overall,
                "speaker_summary": grouped.to_dict('records') if grouped is not None else [],
                "detailed_segments": [
                    {
                        "segment_id": i,
                        "speaker": segment.get('speaker'),
                        "start_time_s": segment.get('start_time_s'),
                        "sentiment": segment.get('sentiment'),
                        "tone": segment.get('tone'),
                        "sentiment_score": segment.get('sentiment_score'),
                        "tone_score": segment.get('score'),
                        "transcription": segment.get('transcription'),
                        "acoustic_features": {
                            "intensity_db": segment.get('intensity_db'),
                            "pitch_mean_hz": segment.get('pitch_mean_hz'),
                            "pitch_std_hz": segment.get('pitch_std_hz'),
                            "speech_rate_wpm": segment.get('speech_rate_wpm'),
                            "vocal_effort": segment.get('vocal_effort')
                        }
                    }
                    for i, segment in enumerate(results)
                ],
                "insights": {
                    "conclusion": conclusion,
                    "quality_score": round(
                        (overall.get('avg_sentiment_score', 0) + overall.get('avg_tone_score', 0)) / 2 * 20, 1
                    ) if overall else 0,
                    "recommendations": generate_recommendations(overall)
                }
            },
            "processing_info": {
                "language": language,
                "confidence_threshold": confidence_threshold,
                "segment_length": segment_length,
                "processing_time_s": 12.3  # Mock processing time
            }
        }

        # Clean up temporary file
        os.unlink(audio_file_path)

        logger.info(f"Analysis completed for {file.filename}")
        return JSONResponse(content=response_data)

    except Exception as e:
        # Clean up temporary file if it exists
        if 'audio_file_path' in locals():
            try:
                os.unlink(audio_file_path)
            except:
                pass

        logger.error(f"Error processing audio: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing audio file: {str(e)}"
        )


def generate_recommendations(overall: Dict[str, Any]) -> List[str]:
    """Generate actionable recommendations based on analysis"""
    if not overall:
        return ["Upload an audio file to get personalized recommendations"]

    recommendations = []

    # Sentiment-based recommendations
    sentiment_score = overall.get('avg_sentiment_score', 0)
    if sentiment_score > 0.5:
        recommendations.append("Excellent customer sentiment detected - maintain current approach")
    elif sentiment_score < -0.3:
        recommendations.append("Consider addressing customer concerns more proactively")
    else:
        recommendations.append("Monitor sentiment trends and adjust communication style")

    # Tone-based recommendations
    tone_score = overall.get('avg_tone_score', 0)
    if tone_score > 3.5:
        recommendations.append("Great energy levels maintained throughout the call")
    else:
        recommendations.append("Consider tone improvement training for better engagement")

    # Speech rate recommendations
    speech_rate = overall.get('speech_rate_wpm', 0)
    if speech_rate > 160:
        recommendations.append("Consider slowing down speech rate for better comprehension")
    elif speech_rate < 120:
        recommendations.append("Increase speech pace to maintain engagement")
    else:
        recommendations.append("Optimal speech rate maintained")

    # Audio quality recommendations
    intensity = overall.get('intensity_db', 0)
    if intensity < 50:
        recommendations.append("Improve audio setup and microphone quality")
    else:
        recommendations.append("Good audio quality maintained")

    return recommendations


@app.get("/analysis-status/{task_id}")
async def get_analysis_status(task_id: str):
    """Get status of analysis task (for future async processing)"""
    # This would be used for async processing with task queues
    return {"task_id": task_id, "status": "completed", "progress": 100}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
#
# # Enhanced Dashboard CSS
# st.markdown("""
#     <style>
#     /* Import Google Fonts */
#     @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
#
#     /* Global Styles */
#     .main {
#         background: #f8fafc;
#         font-family: 'Inter', sans-serif;
#         padding: 0;
#     }
#
#     .stApp {
#         background: #f8fafc;
#     }
#
#     /* Hide Streamlit elements */
#     #MainMenu {visibility: hidden;}
#     footer {visibility: hidden;}
#     header {visibility: hidden;}
#     .stDeployButton {display: none;}
#
#     /* Dashboard Header */
#     .dashboard-header {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         padding: 1.5rem 2rem;
#         margin: -1rem -1rem 2rem -1rem;
#         color: white;
#         display: flex;
#         justify-content: space-between;
#         align-items: center;
#         box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
#     }
#
#     .dashboard-title {
#         font-size: 2rem;
#         font-weight: 700;
#         margin: 0;
#         display: flex;
#         align-items: center;
#         gap: 0.75rem;
#     }
#
#     .logo-space {
#         width: 120px;
#         height: 60px;
#         background: rgba(255, 255, 255, 0.1);
#         border: 2px dashed rgba(255, 255, 255, 0.3);
#         border-radius: 8px;
#         display: flex;
#         align-items: center;
#         justify-content: center;
#         font-size: 0.8rem;
#         color: rgba(255, 255, 255, 0.7);
#     }
#
#     /* Sidebar Styling */
#     .css-1d391kg {
#         background: #1e293b;
#         padding-top: 2rem;
#     }
#
#     .sidebar-content {
#         background: #1e293b;
#         color: white;
#     }
#
#     .upload-section {
#         background: #334155;
#         border-radius: 12px;
#         padding: 1.5rem;
#         margin-bottom: 2rem;
#         border: 2px dashed #64748b;
#         text-align: center;
#         transition: all 0.3s ease;
#     }
#
#     .upload-section:hover {
#         border-color: #667eea;
#         background: #3f4b5f;
#     }
#
#     .upload-title {
#         color: #e2e8f0;
#         font-size: 1.1rem;
#         font-weight: 600;
#         margin-bottom: 0.5rem;
#     }
#
#     .upload-subtitle {
#         color: #94a3b8;
#         font-size: 0.9rem;
#         margin-bottom: 1rem;
#     }
#
#     /* Sidebar sections */
#     .sidebar-section {
#         background: #334155;
#         border-radius: 8px;
#         padding: 1rem;
#         margin-bottom: 1rem;
#     }
#
#     .sidebar-section h3 {
#         color: #e2e8f0;
#         font-size: 1rem;
#         font-weight: 600;
#         margin-bottom: 1rem;
#         border-bottom: 1px solid #475569;
#         padding-bottom: 0.5rem;
#     }
#
#     /* Main Dashboard Grid */
#     .dashboard-grid {
#         display: grid;
#         gap: 1.5rem;
#         margin-bottom: 2rem;
#     }
#
#     /* Metric Cards */
#     .metric-card {
#         background: white;
#         border-radius: 12px;
#         padding: 1.5rem;
#         box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
#         border: 1px solid #e2e8f0;
#         transition: all 0.3s ease;
#     }
#
#     .metric-card:hover {
#         transform: translateY(-2px);
#         box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
#     }
#
#     .metric-header {
#         display: flex;
#         justify-content: space-between;
#         align-items: center;
#         margin-bottom: 1rem;
#     }
#
#     .metric-title {
#         font-size: 0.875rem;
#         font-weight: 500;
#         color: #64748b;
#         text-transform: uppercase;
#         letter-spacing: 0.5px;
#     }
#
#     .metric-icon {
#         width: 40px;
#         height: 40px;
#         border-radius: 8px;
#         display: flex;
#         align-items: center;
#         justify-content: center;
#         font-size: 1.2rem;
#     }
#
#     .metric-value {
#         font-size: 2.5rem;
#         font-weight: 700;
#         color: #1e293b;
#         margin-bottom: 0.25rem;
#     }
#
#     .metric-change {
#         font-size: 0.875rem;
#         font-weight: 500;
#     }
#
#     .metric-change.positive {
#         color: #10b981;
#     }
#
#     .metric-change.negative {
#         color: #ef4444;
#     }
#
#     .metric-change.neutral {
#         color: #64748b;
#     }
#
#     /* Chart Cards */
#     .chart-card {
#         background: white;
#         border-radius: 12px;
#         padding: 1.5rem;
#         box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
#         border: 1px solid #e2e8f0;
#         margin-bottom: 1.5rem;
#     }
#
#     .chart-header {
#         display: flex;
#         justify-content: space-between;
#         align-items: center;
#         margin-bottom: 1.5rem;
#         padding-bottom: 1rem;
#         border-bottom: 1px solid #e2e8f0;
#     }
#
#     .chart-title {
#         font-size: 1.25rem;
#         font-weight: 600;
#         color: #1e293b;
#     }
#
#     .chart-subtitle {
#         font-size: 0.875rem;
#         color: #64748b;
#         margin-top: 0.25rem;
#     }
#
#     /* Status Indicators */
#     .status-indicator {
#         display: inline-flex;
#         align-items: center;
#         gap: 0.5rem;
#         padding: 0.5rem 1rem;
#         border-radius: 20px;
#         font-size: 0.875rem;
#         font-weight: 500;
#     }
#
#     .status-success {
#         background: #dcfce7;
#         color: #166534;
#     }
#
#     .status-warning {
#         background: #fef3c7;
#         color: #92400e;
#     }
#
#     .status-error {
#         background: #fee2e2;
#         color: #991b1b;
#     }
#
#     .status-processing {
#         background: #dbeafe;
#         color: #1e40af;
#     }
#
#     /* Progress Bar */
#     .progress-container {
#         background: #f1f5f9;
#         border-radius: 8px;
#         padding: 1rem;
#         margin: 1rem 0;
#     }
#
#     .stProgress > div > div > div > div {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         border-radius: 4px;
#     }
#
#     /* Data Table Styling */
#     .stDataFrame {
#         border-radius: 8px;
#         overflow: hidden;
#         border: 1px solid #e2e8f0;
#     }
#
#     /* Button Styling */
#     .stButton > button {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         color: white;
#         border: none;
#         border-radius: 8px;
#         padding: 0.75rem 1.5rem;
#         font-weight: 500;
#         transition: all 0.3s ease;
#         box-shadow: 0 2px 8px rgba(102, 126, 234, 0.2);
#     }
#
#     .stButton > button:hover {
#         transform: translateY(-1px);
#         box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
#     }
#
#     /* Tabs Styling */
#     .stTabs [data-baseweb="tab-list"] {
#         gap: 8px;
#         background: #f8fafc;
#         padding: 0.5rem;
#         border-radius: 8px;
#     }
#
#     .stTabs [data-baseweb="tab"] {
#         background: white;
#         border-radius: 6px;
#         padding: 0.75rem 1.5rem;
#         border: 1px solid #e2e8f0;
#         font-weight: 500;
#     }
#
#     .stTabs [aria-selected="true"] {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         color: white;
#         border-color: transparent;
#     }
#
#     /* Responsive Grid */
#     @media (min-width: 768px) {
#         .metrics-grid {
#             display: grid;
#             grid-template-columns: repeat(2, 1fr);
#             gap: 1.5rem;
#         }
#     }
#
#     @media (min-width: 1024px) {
#         .metrics-grid {
#             grid-template-columns: repeat(4, 1fr);
#         }
#     }
#
#     @media (min-width: 1280px) {
#         .charts-grid {
#             display: grid;
#             grid-template-columns: repeat(2, 1fr);
#             gap: 1.5rem;
#         }
#     }
#     </style>
# """, unsafe_allow_html=True)
#
# # Dashboard Header
# st.markdown("""
#     <div class="dashboard-header">
#         <h1 class="dashboard-title">
#             🎙️ Call Center Analytics Dashboard
#         </h1>
#         <div class="logo-space">
#             Your Logo Here
#         </div>
#     </div>
# """, unsafe_allow_html=True)
#
# # Sidebar Configuration
# with st.sidebar:
#     st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
#
#     # Upload Section
#     st.markdown("""
#         <div class="upload-section">
#             <div class="upload-title">📂 Upload Audio File</div>
#             <div class="upload-subtitle">Drag and drop your WAV file here</div>
#         </div>
#     """, unsafe_allow_html=True)
#
#     uploaded_file = st.file_uploader(
#         "",
#         type=["wav"],
#         help="Upload a WAV file for analysis",
#         label_visibility="collapsed"
#     )
#
#     # Analysis Settings
#     st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
#     st.markdown("### ⚙️ Analysis Settings")
#
#     language_option = st.selectbox(
#         "Transcription Language",
#         ["Hindi (hi)", "Auto-detect"],
#         help="Select language for Whisper transcription"
#     )
#
#     confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.7, 0.1)
#     segment_length = st.slider("Segment Length (seconds)", 5, 30, 10, 5)
#     st.markdown('</div>', unsafe_allow_html=True)
#
#     # Processing Status
#     if 'processing_status' in st.session_state:
#         st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
#         st.markdown("### 📊 Processing Status")
#         status = st.session_state.processing_status
#         if status == "complete":
#             st.markdown('<div class="status-indicator status-success">✅ Analysis Complete</div>',
#                         unsafe_allow_html=True)
#         elif status == "processing":
#             st.markdown('<div class="status-indicator status-processing">⏳ Processing...</div>', unsafe_allow_html=True)
#         elif status == "error":
#             st.markdown('<div class="status-indicator status-error">❌ Error Occurred</div>', unsafe_allow_html=True)
#         st.markdown('</div>', unsafe_allow_html=True)
#
#     # Quick Stats
#     if 'results' in st.session_state and st.session_state.results:
#         st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
#         st.markdown("### 📈 Quick Stats")
#         st.metric("Total Segments", len(st.session_state.results))
#         st.metric("Processing Time", f"{st.session_state.get('processing_time', 0):.1f}s")
#         st.metric("File Size", f"{st.session_state.get('file_size', 0):.1f} MB")
#         st.markdown('</div>', unsafe_allow_html=True)
#
#     st.markdown('</div>', unsafe_allow_html=True)
#
# # Main Dashboard Content
# if uploaded_file is not None:
#     # Initialize session state
#     if 'processing_status' not in st.session_state:
#         st.session_state.processing_status = "processing"
#
#     # Save uploaded file temporarily
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
#         tmp_file.write(uploaded_file.read())
#         audio_file_path = tmp_file.name
#         st.session_state.file_size = len(uploaded_file.read()) / (1024 * 1024)  # MB
#
#     # Processing Section
#     if st.session_state.processing_status == "processing":
#         st.markdown('<div class="progress-container">', unsafe_allow_html=True)
#         progress_bar = st.progress(0)
#         status_placeholder = st.empty()
#
#         steps = [
#             "🎵 Loading audio file...",
#             "🗣️ Transcribing speech...",
#             "🧠 Analyzing sentiment...",
#             "📊 Extracting features...",
#             "📈 Generating insights..."
#         ]
#
#         try:
#             for i, step in enumerate(steps):
#                 status_placeholder.markdown(f'<div class="status-indicator status-processing">{step}</div>',
#                                             unsafe_allow_html=True)
#                 progress_bar.progress((i + 1) * 20)
#
#                 if i == 2:  # Actual processing
#                     results = process_audio_file(audio_file_path)
#                     grouped, overall, df = aggregate_results(results)
#                     st.session_state.results = results
#                     st.session_state.grouped = grouped
#                     st.session_state.overall = overall
#                     st.session_state.df = df
#                     st.session_state.processing_time = 12.3
#                     st.session_state.processing_status = "complete"
#
#             status_placeholder.markdown('<div class="status-indicator status-success">✅ Processing Complete!</div>',
#                                         unsafe_allow_html=True)
#             st.rerun()
#
#         except Exception as e:
#             st.session_state.processing_status = "error"
#             st.error(f"❌ Error processing audio: {e}")
#             logging.error(f"Processing error: {e}")
#
#         st.markdown('</div>', unsafe_allow_html=True)
#
#     # Clean up temporary file
#     os.unlink(audio_file_path)
#
#     # Dashboard Content (when processing is complete)
#     if st.session_state.processing_status == "complete":
#         results = st.session_state.results
#         grouped = st.session_state.grouped
#         overall = st.session_state.overall
#         df = st.session_state.df
#
#         # Key Metrics Row
#         st.markdown("## 📊 Key Performance Indicators")
#
#         if overall is not None:
#             col1, col2, col3, col4 = st.columns(4)
#
#             with col1:
#                 st.markdown(f"""
#                     <div class="metric-card">
#                         <div class="metric-header">
#                             <div class="metric-title">Avg Sentiment Score</div>
#                             <div class="metric-icon" style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white;">😊</div>
#                         </div>
#                         <div class="metric-value">{overall.get('avg_sentiment_score', 0):.2f}</div>
#                         <div class="metric-change positive">+12% from last call</div>
#                     </div>
#                 """, unsafe_allow_html=True)
#
#             with col2:
#                 st.markdown(f"""
#                     <div class="metric-card">
#                         <div class="metric-header">
#                             <div class="metric-title">Avg Tone Score</div>
#                             <div class="metric-icon" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;">🎭</div>
#                         </div>
#                         <div class="metric-value">{overall.get('avg_tone_score', 0):.2f}</div>
#                         <div class="metric-change neutral">Stable</div>
#                     </div>
#                 """, unsafe_allow_html=True)
#
#             with col3:
#                 st.markdown(f"""
#                     <div class="metric-card">
#                         <div class="metric-header">
#                             <div class="metric-title">Total Segments</div>
#                             <div class="metric-icon" style="background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); color: white;">📊</div>
#                         </div>
#                         <div class="metric-value">{int(overall.get('segment_count', 0))}</div>
#                         <div class="metric-change neutral">Normal range</div>
#                     </div>
#                 """, unsafe_allow_html=True)
#
#             with col4:
#                 st.markdown(f"""
#                     <div class="metric-card">
#                         <div class="metric-header">
#                             <div class="metric-title">Call Duration</div>
#                             <div class="metric-icon" style="background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); color: white;">⏱️</div>
#                         </div>
#                         <div class="metric-value">{overall.get('duration_s', 0):.0f}s</div>
#                         <div class="metric-change positive">Efficient</div>
#                     </div>
#                 """, unsafe_allow_html=True)
#
#         # Secondary Metrics Row
#         col5, col6 = st.columns(2)
#
#         with col5:
#             st.markdown(f"""
#                 <div class="metric-card">
#                     <div class="metric-header">
#                         <div class="metric-title">Speech Rate (WPM)</div>
#                         <div class="metric-icon" style="background: linear-gradient(135deg, #06b6d4 0%, #0891b2 100%); color: white;">🗣️</div>
#                     </div>
#                     <div class="metric-value">{overall.get('speech_rate_wpm', 0):.1f}</div>
#                     <div class="metric-change positive">Optimal pace</div>
#                 </div>
#             """, unsafe_allow_html=True)
#
#         with col6:
#             st.markdown(f"""
#                 <div class="metric-card">
#                     <div class="metric-header">
#                         <div class="metric-title">Vocal Effort</div>
#                         <div class="metric-icon" style="background: linear-gradient(135deg, #ec4899 0%, #db2777 100%); color: white;">🔊</div>
#                     </div>
#                     <div class="metric-value">{overall.get('vocal_effort', 0):.1f}</div>
#                     <div class="metric-change neutral">Balanced</div>
#                 </div>
#             """, unsafe_allow_html=True)
#
#         # Charts Section
#         st.markdown("## 📈 Analytics Dashboard")
#
#         # Create tabs for different views
#         tab1, tab2, tab3, tab4 = st.tabs(
#             ["📊 Overview", "🎭 Sentiment Analysis", "🔊 Acoustic Analysis", "⏱️ Timeline View"])
#
#         with tab1:
#             col1, col2 = st.columns(2)
#
#             with col1:
#                 st.markdown('<div class="chart-card">', unsafe_allow_html=True)
#                 st.markdown("""
#                     <div class="chart-header">
#                         <div>
#                             <div class="chart-title">Sentiment Distribution</div>
#                             <div class="chart-subtitle">Overall call sentiment breakdown</div>
#                         </div>
#                     </div>
#                 """, unsafe_allow_html=True)
#
#                 try:
#                     sentiment_counts = df.groupby('sentiment').size().reset_index(name='count')
#                     fig_sentiment = px.pie(
#                         sentiment_counts,
#                         values='count',
#                         names='sentiment',
#                         color_discrete_map={
#                             'POSITIVE': '#10b981',
#                             'NEGATIVE': '#ef4444',
#                             'NEUTRAL': '#f59e0b'
#                         },
#                         hole=0.4
#                     )
#                     fig_sentiment.update_layout(
#                         plot_bgcolor="rgba(0,0,0,0)",
#                         paper_bgcolor="rgba(0,0,0,0)",
#                         height=350,
#                         showlegend=True,
#                         font=dict(size=12)
#                     )
#                     st.plotly_chart(fig_sentiment, use_container_width=True)
#                 except Exception as e:
#                     st.error(f"Error plotting sentiment: {e}")
#
#                 st.markdown('</div>', unsafe_allow_html=True)
#
#             with col2:
#                 st.markdown('<div class="chart-card">', unsafe_allow_html=True)
#                 st.markdown("""
#                     <div class="chart-header">
#                         <div>
#                             <div class="chart-title">Speaker Performance</div>
#                             <div class="chart-subtitle">Tone scores by speaker</div>
#                         </div>
#                     </div>
#                 """, unsafe_allow_html=True)
#
#                 try:
#                     speaker_scores = df.groupby('speaker')['score'].mean().reset_index()
#                     fig_speaker = px.bar(
#                         speaker_scores,
#                         x='speaker',
#                         y='score',
#                         color='score',
#                         color_continuous_scale='Viridis'
#                     )
#                     fig_speaker.update_layout(
#                         plot_bgcolor="rgba(0,0,0,0)",
#                         paper_bgcolor="rgba(0,0,0,0)",
#                         height=350,
#                         showlegend=False
#                     )
#                     st.plotly_chart(fig_speaker, use_container_width=True)
#                 except Exception as e:
#                     st.error(f"Error plotting speaker performance: {e}")
#
#                 st.markdown('</div>', unsafe_allow_html=True)
#
#         with tab2:
#             st.markdown('<div class="chart-card">', unsafe_allow_html=True)
#             st.markdown("""
#                 <div class="chart-header">
#                     <div>
#                         <div class="chart-title">Sentiment Analysis by Speaker</div>
#                         <div class="chart-subtitle">Detailed sentiment breakdown across speakers</div>
#                     </div>
#                 </div>
#             """, unsafe_allow_html=True)
#
#             try:
#                 sentiment_speaker = df.groupby(['speaker', 'sentiment']).size().reset_index(name='count')
#                 fig_sentiment_speaker = px.bar(
#                     sentiment_speaker,
#                     x='speaker',
#                     y='count',
#                     color='sentiment',
#                     color_discrete_map={
#                         'POSITIVE': '#10b981',
#                         'NEGATIVE': '#ef4444',
#                         'NEUTRAL': '#f59e0b'
#                     },
#                     barmode='group'
#                 )
#                 fig_sentiment_speaker.update_layout(
#                     plot_bgcolor="rgba(0,0,0,0)",
#                     paper_bgcolor="rgba(0,0,0,0)",
#                     height=400
#                 )
#                 st.plotly_chart(fig_sentiment_speaker, use_container_width=True)
#             except Exception as e:
#                 st.error(f"Error plotting sentiment analysis: {e}")
#
#             st.markdown('</div>', unsafe_allow_html=True)
#
#         with tab3:
#             st.markdown('<div class="chart-card">', unsafe_allow_html=True)
#             st.markdown("""
#                 <div class="chart-header">
#                     <div>
#                         <div class="chart-title">3D Acoustic Feature Analysis</div>
#                         <div class="chart-subtitle">Intensity, pitch variation, and tone scores</div>
#                     </div>
#                 </div>
#             """, unsafe_allow_html=True)
#
#             try:
#                 fig_3d = px.scatter_3d(
#                     df,
#                     x='intensity_db',
#                     y='pitch_std_hz',
#                     z='score',
#                     color='tone',
#                     size='score',
#                     hover_data=['speaker', 'transcription'],
#                     color_discrete_map={
#                         'ENTHUSIASTIC': '#10b981',
#                         'LAZY': '#ef4444'
#                     }
#                 )
#                 fig_3d.update_layout(
#                     plot_bgcolor="rgba(0,0,0,0)",
#                     paper_bgcolor="rgba(0,0,0,0)",
#                     height=500
#                 )
#                 st.plotly_chart(fig_3d, use_container_width=True)
#             except Exception as e:
#                 st.error(f"Error plotting 3D analysis: {e}")
#
#             st.markdown('</div>', unsafe_allow_html=True)
#
#         with tab4:
#             st.markdown('<div class="chart-card">', unsafe_allow_html=True)
#             st.markdown("""
#                 <div class="chart-header">
#                     <div>
#                         <div class="chart-title">Timeline Analysis</div>
#                         <div class="chart-subtitle">Tone score evolution throughout the call</div>
#                     </div>
#                 </div>
#             """, unsafe_allow_html=True)
#
#             try:
#                 fig_timeline = go.Figure()
#
#                 for speaker in df['speaker'].unique():
#                     speaker_data = df[df['speaker'] == speaker]
#                     fig_timeline.add_trace(go.Scatter(
#                         x=speaker_data['start_time_s'],
#                         y=speaker_data['score'],
#                         mode='lines+markers',
#                         name=speaker,
#                         line=dict(width=3),
#                         marker=dict(size=8)
#                     ))
#
#                 fig_timeline.update_layout(
#                     title="",
#                     xaxis_title="Time (seconds)",
#                     yaxis_title="Tone Score",
#                     plot_bgcolor="rgba(0,0,0,0)",
#                     paper_bgcolor="rgba(0,0,0,0)",
#                     height=400
#                 )
#                 st.plotly_chart(fig_timeline, use_container_width=True)
#             except Exception as e:
#                 st.error(f"Error plotting timeline: {e}")
#
#             st.markdown('</div>', unsafe_allow_html=True)
#
#         # Data Tables Section
#         st.markdown("## 📋 Detailed Analysis")
#
#         col1, col2 = st.columns(2)
#
#         with col1:
#             st.markdown('<div class="chart-card">', unsafe_allow_html=True)
#             st.markdown("""
#                 <div class="chart-header">
#                     <div>
#                         <div class="chart-title">Speaker Summary</div>
#                         <div class="chart-subtitle">Aggregated metrics by speaker</div>
#                     </div>
#                 </div>
#             """, unsafe_allow_html=True)
#
#             if grouped is not None:
#                 st.dataframe(grouped, use_container_width=True, height=300)
#             else:
#                 st.info("No speaker data available")
#
#             st.markdown('</div>', unsafe_allow_html=True)
#
#         with col2:
#             st.markdown('<div class="chart-card">', unsafe_allow_html=True)
#             st.markdown("""
#                 <div class="chart-header">
#                     <div>
#                         <div class="chart-title">Overall Metrics</div>
#                         <div class="chart-subtitle">Call-level aggregated data</div>
#                     </div>
#                 </div>
#             """, unsafe_allow_html=True)
#
#             if overall is not None:
#                 overall_df = pd.DataFrame([overall])
#                 st.dataframe(overall_df, use_container_width=True, height=300)
#             else:
#                 st.info("No overall data available")
#
#             st.markdown('</div>', unsafe_allow_html=True)
#
#         # AI Insights Section
#         st.markdown("## 🤖 AI-Generated Insights")
#         st.markdown('<div class="chart-card">', unsafe_allow_html=True)
#
#         conclusion = generate_conclusion(overall)
#
#         col1, col2 = st.columns([2, 1])
#
#         with col1:
#             st.markdown(f"""
#             ### 📝 Analysis Summary
#             {conclusion}
#
#             ### 💡 Key Recommendations
#             Based on the comprehensive analysis:
#
#             - **Agent Performance**: {"Excellent tone consistency maintained" if overall.get('avg_tone_score', 0) > 3 else "Consider tone improvement training sessions"}
#             - **Customer Satisfaction**: {"Positive interaction patterns detected" if overall.get('avg_sentiment_score', 0) > 0.5 else "Address customer concerns proactively"}
#             - **Call Quality**: {"High engagement and clarity levels" if overall.get('intensity_db', 0) > 50 else "Improve audio setup and clarity"}
#             - **Communication**: {"Optimal speech rate maintained" if 120 <= overall.get('speech_rate_wpm', 0) <= 160 else "Adjust speaking pace for better comprehension"}
#             """)
#
#         with col2:
#             # Quality Score Gauge
#             quality_score = (overall.get('avg_sentiment_score', 0) + overall.get('avg_tone_score', 0)) / 2 * 20
#             st.markdown(f"""
#                 <div style="text-align: center; padding: 2rem; background: #f8fafc; border-radius: 12px;">
#                     <div style="font-size: 3rem; font-weight: 700; color: {'#10b981' if quality_score > 70 else '#f59e0b' if quality_score > 50 else '#ef4444'};">
#                         {quality_score:.0f}%
#                     </div>
#                     <div style="font-size: 1.1rem; font-weight: 600; color: #64748b; margin-top: 0.5rem;">
#                         Overall Quality Score
#                     </div>
#                     <div style="margin-top: 1rem;">
#                         <div class="status-indicator {'status-success' if quality_score > 70 else 'status-warning' if quality_score > 50 else 'status-error'}">
#                             {'Excellent' if quality_score > 70 else 'Good' if quality_score > 50 else 'Needs Improvement'}
#                         </div>
#                     </div>
#                 </div>
#             """, unsafe_allow_html=True)
#
#         st.markdown('</div>', unsafe_allow_html=True)
#
#         # Export Section
#         st.markdown("## 📥 Export Data")
#         col1, col2, col3 = st.columns(3)
#
#         with col1:
#             csv_data = df.to_csv(index=False)
#             st.download_button(
#                 label="📊 Download Full Analysis",
#                 data=csv_data,
#                 file_name="call_analysis_complete.csv",
#                 mime="text/csv"
#             )
#
#         with col2:
#             summary_data = grouped.to_csv(index=False) if grouped is not None else ""
#             st.download_button(
#                 label="👥 Download Speaker Summary",
#                 data=summary_data,
#                 file_name="speaker_summary.csv",
#                 mime="text/csv"
#             )
#
#         with col3:
#             st.button("📧 Email Report", help="Send analysis report via email")
#
# else:
#     # Welcome Dashboard
#     st.markdown("""
#         <div style="text-align: center; padding: 4rem 2rem; background: white; border-radius: 16px; margin: 2rem 0; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);">
#             <div style="font-size: 4rem; margin-bottom: 1rem;">🎧</div>
#             <h2 style="color: #1e293b; margin-bottom: 1rem;">Welcome to Call Center Analytics</h2>
#             <p style="font-size: 1.2rem; color: #64748b; margin-bottom: 2rem; max-width: 600px; margin-left: auto; margin-right: auto;">
#                 Upload your audio file using the sidebar to get started with AI-powered sentiment and tone analysis
#             </p>
#
#             <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 2rem; margin-top: 3rem; max-width: 800px; margin-left: auto; margin-right: auto;">
#                 <div style="text-align: center; padding: 2rem; background: #f8fafc; border-radius: 12px;">
#                     <div style="font-size: 2.5rem; margin-bottom: 1rem;">🎯</div>
#                     <div style="font-weight: 600; color: #1e293b; margin-bottom: 0.5rem;">Accurate Analysis</div>
#                     <div style="color: #64748b; font-size: 0.9rem;">Advanced AI models for precise sentiment detection</div>
#                 </div>
#
#                 <div style="text-align: center; padding: 2rem; background: #f8fafc; border-radius: 12px;">
#                     <div style="font-size: 2.5rem; margin-bottom: 1rem;">⚡</div>
#                     <div style="font-weight: 600; color: #1e293b; margin-bottom: 0.5rem;">Fast Processing</div>
#                     <div style="color: #64748b; font-size: 0.9rem;">Real-time analysis with instant results</div>
#                 </div>
#
#                 <div style="text-align: center; padding: 2rem; background: #f8fafc; border-radius: 12px;">
#                     <div style="font-size: 2.5rem; margin-bottom: 1rem;">📊</div>
#                     <div style="font-weight: 600; color: #1e293b; margin-bottom: 0.5rem;">Rich Insights</div>
#                     <div style="color: #64748b; font-size: 0.9rem;">Comprehensive analytics and visualizations</div>
#                 </div>
#
#                 <div style="text-align: center; padding: 2rem; background: #f8fafc; border-radius: 12px;">
#                     <div style="font-size: 2.5rem; margin-bottom: 1rem;">🤖</div>
#                     <div style="font-weight: 600; color: #1e293b; margin-bottom: 0.5rem;">AI-Powered</div>
#                     <div style="color: #64748b; font-size: 0.9rem;">Machine learning algorithms for smart analysis</div>
#                 </div>
#             </div>
#         </div>
#     """, unsafe_allow_html=True)
#
#     # Feature Overview
#     st.markdown("## 🚀 Platform Features")
#
#     col1, col2, col3 = st.columns(3)
#
#     with col1:
#         st.markdown("""
#             <div class="chart-card">
#                 <h4>🎭 Sentiment Analysis</h4>
#                 <ul style="color: #64748b; line-height: 1.6;">
#                     <li>Real-time emotion detection</li>
#                     <li>Positive/Negative/Neutral classification</li>
#                     <li>Confidence scoring</li>
#                     <li>Speaker-specific analysis</li>
#                 </ul>
#             </div>
#         """, unsafe_allow_html=True)
#
#     with col2:
#         st.markdown("""
#             <div class="chart-card">
#                 <h4>🔊 Acoustic Features</h4>
#                 <ul style="color: #64748b; line-height: 1.6;">
#                     <li>Pitch and intensity analysis</li>
#                     <li>Speech rate calculation</li>
#                     <li>Vocal effort measurement</li>
#                     <li>Audio quality assessment</li>
#                 </ul>
#             </div>
#         """, unsafe_allow_html=True)
#
#     with col3:
#         st.markdown("""
#             <div class="chart-card">
#                 <h4>📈 Advanced Analytics</h4>
#                 <ul style="color: #64748b; line-height: 1.6;">
#                     <li>Interactive visualizations</li>
#                     <li>Timeline analysis</li>
#                     <li>Performance metrics</li>
#                     <li>Exportable reports</li>
#                 </ul>
#             </div>
#         """, unsafe_allow_html=True)