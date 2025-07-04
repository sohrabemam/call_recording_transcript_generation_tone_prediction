import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import tempfile
import os
import logging
from audio_processor import process_audio_file, aggregate_results, generate_conclusion
from dotenv import load_dotenv
load_dotenv()
# Configure page
st.set_page_config(
    page_title="Call Center Audio Analysis",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Enhanced Custom CSS
st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global Styles */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: transparent;
    }

    /* Header Styles */
    .main-header {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    .main-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }

    .main-subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 1.2rem;
        font-weight: 400;
    }

    /* Card Styles */
    .analysis-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Upload Area */
    .upload-area {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border: 2px dashed #667eea;
        border-radius: 16px;
        padding: 3rem;
        text-align: center;
        margin-bottom: 2rem;
        transition: all 0.3s ease;
    }

    .upload-area:hover {
        border-color: #764ba2;
        background: rgba(255, 255, 255, 1);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    }

    /* Button Styles */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }

    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }

    /* Sidebar */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
    }

    /* Data Tables */
    .stDataFrame {
        background: white;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 12px;
        font-weight: 600;
    }

    /* Status Messages */
    .status-success {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        font-weight: 600;
    }

    .status-processing {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        font-weight: 600;
    }

    /* Chart Containers */
    .chart-container {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
    }

    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("""
    <div class="main-header">
        <h1 class="main-title">🎙️ Call Center Audio Analysis</h1>
        <p class="main-subtitle">Advanced AI-powered sentiment and tone analysis for customer service calls</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.markdown("### ⚙️ Analysis Settings")

    # Language Selection
    language_option = st.selectbox(
        "Transcription Language",
        ["Hindi (hi)", "Auto-detect"],
        help="Select language for Whisper transcription"
    )

    # Analysis Parameters
    st.markdown("### 📊 Parameters")
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.7, 0.1)
    segment_length = st.slider("Segment Length (seconds)", 5, 30, 10, 5)

    # Quick Stats (placeholder)
    st.markdown("### 📈 Quick Stats")
    if 'results' in st.session_state and st.session_state.results:
        st.metric("Total Segments", len(st.session_state.results))
        st.metric("Processing Time", f"{st.session_state.get('processing_time', 0):.1f}s")

# Main Content Area
col1, col2 = st.columns([2, 1])

with col1:
    # File Upload Section
    st.markdown("""
        <div class="upload-area">
            <h3>📂 Upload Audio File</h3>
            <p>Drag and drop your WAV file here or click to browse</p>
        </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "",
        type=["wav"],
        help="Upload a WAV file for tone and sentiment analysis. Ensure the audio is clear for best results.",
        label_visibility="collapsed"
    )

with col2:
    # Quick Info Panel
    st.markdown("""
        <div class="analysis-card">
            <h4>📋 Analysis Features</h4>
            <ul style="list-style: none; padding: 0;">
                <li>✅ Sentiment Analysis</li>
                <li>✅ Tone Detection</li>
                <li>✅ Speaker Identification</li>
                <li>✅ Acoustic Features</li>
                <li>✅ Timeline Analysis</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

# Processing Section
if uploaded_file is not None:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        audio_file_path = tmp_file.name

    # Processing UI
    st.markdown('<div class="analysis-card">', unsafe_allow_html=True)

    # Progress tracking
    progress_container = st.container()
    with progress_container:
        progress_bar = st.progress(0)
        status_placeholder = st.empty()

        # Processing steps
        steps = [
            "🎵 Loading audio file...",
            "🗣️ Transcribing speech...",
            "🧠 Analyzing sentiment...",
            "📊 Extracting features...",
            "📈 Generating insights..."
        ]

        try:
            for i, step in enumerate(steps):
                status_placeholder.markdown(f'<div class="status-processing">{step}</div>', unsafe_allow_html=True)
                progress_bar.progress((i + 1) * 20)

                if i == 2:  # Actual processing happens here
                    results = process_audio_file(audio_file_path)
                    grouped, overall, df = aggregate_results(results)
                    st.session_state.results = results
                    st.session_state.processing_time = 12.3  # Placeholder (to be replaced with actual time)

            status_placeholder.markdown('<div class="status-success">✅ Processing Complete!</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Error processing audio: {e}")
            logging.error(f"Processing error: {e}")
            results = None

    st.markdown('</div>', unsafe_allow_html=True)

    # Clean up temporary file
    os.unlink(audio_file_path)

    if results:
        # Metrics Overview (Enhanced Analysis Overview)
        st.markdown("## 📊 Analysis Overview")

        if overall is not None:
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{overall.get('avg_sentiment_score', 0):.2f}</div>
                        <div class="metric-label">Avg Sentiment Score</div>
                    </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{overall.get('avg_tone_score', 0):.2f}</div>
                        <div class="metric-label">Avg Tone Score</div>
                    </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{int(overall.get('segment_count', 0))}</div>
                        <div class="metric-label">Segments</div>
                    </div>
                """, unsafe_allow_html=True)

            with col4:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{overall.get('duration_s', 0):.0f}s</div>
                        <div class="metric-label">Duration</div>
                    </div>
                """, unsafe_allow_html=True)

            # Additional Acoustic Metrics
            col5, col6 = st.columns(2)
            with col5:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{overall.get('speech_rate_wpm', 0):.1f}</div>
                        <div class="metric-label">Avg Speech Rate</div>
                    </div>
                """, unsafe_allow_html=True)

            with col6:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{overall.get('vocal_effort', 0):.1f}</div>
                        <div class="metric-label">Vocal Effort</div>
                    </div>
                """, unsafe_allow_html=True)

        # Summary Tables
        st.markdown("## 📋 Summary Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.subheader("👥 Speaker Analysis")
            if grouped is not None:
                st.dataframe(grouped, use_container_width=True, height=300)
            else:
                default_grouped = pd.DataFrame({
                    'speaker': ['Agent', 'Customer'],
                    'sentiment': ['NEUTRAL', 'NEUTRAL'],
                    'tone': ['ENTHUSIASTIC', 'ENTHUSIASTIC'],
                    'OPENAI_TONE': ['ENTHUSIASTIC', 'ENTHUSIASTIC'],
                    'score': [0.0, 0.0],
                    'sentiment_score': [0.0, 0.0],
                    'intensity_db': [0.0, 0.0],
                    'pitch_mean_hz': [0.0, 0.0],
                    'pitch_std_hz': [0.0, 0.0],
                    'pitch_range': [0.0, 0.0],
                    'jitter_percent': [0.0, 0.0],
                    'shimmer_percent': [0.0, 0.0],
                    'spectral_tilt_db': [0.0, 0.0],
                    'speech_rate_wpm': [0.0, 0.0],
                    'avg_pause_duration_s': [0.0, 0.0],
                    'vocal_fry_ratio': [0.0, 0.0],
                    'vocal_effort': [0.0, 0.0],
                    'speech_clarity': [0.0, 0.0]
                })
                st.dataframe(default_grouped, use_container_width=True, height=300)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.subheader("🎯 Overall Metrics")
            if overall is not None:
                overall_df = pd.DataFrame([overall])
                st.dataframe(overall_df, use_container_width=True, height=300)
            else:
                default_overall = pd.DataFrame({
                    'sentiment': ['NEUTRAL'],
                    'tone': ['ENTHUSIASTIC'],
                    'OPENAI_TONE': ['ENTHUSIASTIC'],
                    'score': [0.0],
                    'sentiment_score': [0.0],
                    'intensity_db': [0.0],
                    'pitch_mean_hz': [0.0],
                    'pitch_std_hz': [0.0],
                    'pitch_range': [0.0],
                    'jitter_percent': [0.0],
                    'shimmer_percent': [0.0],
                    'spectral_tilt_db': [0.0],
                    'speech_rate_wpm': [0.0],
                    'avg_pause_duration_s': [0.0],
                    'vocal_fry_ratio': [0.0],
                    'vocal_effort': [0.0],
                    'speech_clarity': [0.0],
                    'avg_sentiment_score': [0.0],
                    'avg_tone_score': [0.0],
                    'segment_count': [0],
                    'duration_s': [0.0]
                })
                st.dataframe(default_overall, use_container_width=True, height=300)
            st.markdown('</div>', unsafe_allow_html=True)

        # Visualizations
        st.markdown("## 📈 Interactive Visualizations")

        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Sentiment", "🎭 Tone", "🔊 Acoustics", "⏱️ Timeline"])

        with tab1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            try:
                sentiment_counts = df.groupby(['speaker', 'sentiment']).size().reset_index(name='count')
                fig_sentiment = px.bar(
                    sentiment_counts,
                    x='speaker',
                    y='count',
                    color='sentiment',
                    title="Sentiment Distribution by Speaker",
                    color_discrete_map={
                        'POSITIVE': '#10b981',
                        'NEGATIVE': '#ef4444',
                        'NEUTRAL': '#f59e0b'
                    },
                    barmode='group'
                )
                fig_sentiment.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(size=12),
                    height=400
                )
                st.plotly_chart(fig_sentiment, use_container_width=True)
            except Exception as e:
                st.error(f"Error plotting sentiment: {e}")
            st.markdown('</div>', unsafe_allow_html=True)

        with tab2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            try:
                tone_counts = df.groupby(['speaker', 'tone']).size().reset_index(name='count')
                fig_tone = px.pie(
                    tone_counts,
                    values='count',
                    names='tone',
                    title="Overall Tone Distribution",
                    color_discrete_map={
                        'ENTHUSIASTIC': '#10b981',
                        'LAZY': '#ef4444'
                    }
                )
                fig_tone.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    height=400
                )
                st.plotly_chart(fig_tone, use_container_width=True)
            except Exception as e:
                st.error(f"Error plotting tone: {e}")
            st.markdown('</div>', unsafe_allow_html=True)

        with tab3:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            try:
                fig_3d = px.scatter_3d(
                    df,
                    x='intensity_db',
                    y='pitch_std_hz',
                    z='score',
                    color='tone',
                    size='score',
                    hover_data=['speaker', 'transcription'],
                    title="3D Acoustic Feature Analysis",
                    color_discrete_map={
                        'ENTHUSIASTIC': '#10b981',
                        'LAZY': '#ef4444'
                    }
                )
                fig_3d.update_layout(height=500)
                st.plotly_chart(fig_3d, use_container_width=True)
            except Exception as e:
                st.error(f"Error plotting 3D scatter: {e}")
            st.markdown('</div>', unsafe_allow_html=True)

        with tab4:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            try:
                df['color'] = df['tone'].apply(lambda x: '#10b981' if x == 'ENTHUSIASTIC' else '#ef4444')
                fig_timeline = go.Figure()

                # Add line for each speaker
                for speaker in df['speaker'].unique():
                    speaker_data = df[df['speaker'] == speaker]
                    fig_timeline.add_trace(go.Scatter(
                        x=speaker_data['start_time_s'],
                        y=speaker_data['score'],
                        mode='lines+markers',
                        name=speaker,
                        line=dict(width=3),
                        marker=dict(size=8)
                    ))

                fig_timeline.update_layout(
                    title="Tone Score Evolution Over Time",
                    xaxis_title="Time (seconds)",
                    yaxis_title="Tone Score",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    height=400
                )
                st.plotly_chart(fig_timeline, use_container_width=True)
            except Exception as e:
                st.error(f"Error plotting timeline: {e}")
            st.markdown('</div>', unsafe_allow_html=True)

        # Detailed Analysis Expander
        with st.expander("🔍 Detailed Analysis Data", expanded=False):
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)

            # Add search and filter options
            col1, col2, col3 = st.columns(3)
            with col1:
                speaker_filter = st.selectbox("Filter by Speaker", ["All"] + list(df['speaker'].unique()))
            with col2:
                sentiment_filter = st.selectbox("Filter by Sentiment", ["All"] + list(df['sentiment'].unique()))
            with col3:
                tone_filter = st.selectbox("Filter by Tone", ["All"] + list(df['tone'].unique()))

            # Apply filters
            filtered_df = df.copy()
            if speaker_filter != "All":
                filtered_df = filtered_df[filtered_df['speaker'] == speaker_filter]
            if sentiment_filter != "All":
                filtered_df = filtered_df[filtered_df['sentiment'] == sentiment_filter]
            if tone_filter != "All":
                filtered_df = filtered_df[filtered_df['tone'] == tone_filter]

            st.dataframe(
                filtered_df.drop(columns=['start_time_s'], errors='ignore'),
                use_container_width=True,
                height=400
            )

            # Download button
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Filtered Data",
                data=csv,
                file_name="call_analysis_filtered.csv",
                mime="text/csv"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        # AI-Generated Insights
        st.markdown("## 🤖 AI Insights & Recommendations")
        st.markdown('<div class="analysis-card">', unsafe_allow_html=True)

        conclusion = generate_conclusion(overall)

        # Enhanced conclusion with recommendations
        st.markdown(f"""
        ### 📝 Analysis Summary
        {conclusion}

        ### 💡 Recommendations
        Based on the analysis, here are some actionable insights:

        - **Agent Performance**: {"Excellent tone consistency" if overall.get('avg_tone_score', 0) > 3 else "Consider tone improvement training"}
        - **Customer Satisfaction**: {"Positive interaction detected" if overall.get('avg_sentiment_score', 0) > 0.5 else "Address customer concerns"}
        - **Call Quality**: {"High engagement levels" if overall.get('intensity_db', 0) > 50 else "Improve audio clarity"}
        """)

        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.error(
            "❌ No valid transcriptions found in the audio file. Please ensure the audio is clear and contains speech.")

else:
    # Welcome message when no file is uploaded
    st.markdown("""
        <div class="analysis-card" style="text-align: center; padding: 3rem;">
            <h2>🎧 Ready to Analyze Your Call?</h2>
            <p style="font-size: 1.1rem; color: #6b7280; margin-bottom: 2rem;">
                Upload a WAV audio file to get started with AI-powered sentiment and tone analysis
            </p>
            <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 2rem;">
                <div style="text-align: center;">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">🎯</div>
                    <div style="font-weight: 600;">Accurate Analysis</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">⚡</div>
                    <div style="font-weight: 600;">Fast Processing</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">📊</div>
                    <div style="font-weight: 600;">Rich Insights</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
    <div style="text-align: center; padding: 2rem; color: rgba(255,255,255,0.7);">
        <p>Built with ❤️ using Streamlit • AI-Powered Audio Analysis</p>
    </div>
""", unsafe_allow_html=True)