# 📞 Call Recording Transcript & Tone Prediction

This project processes `.wav` call recordings from customer service conversations to:

- 🎙️ Diarize speakers (Customer vs Agent)
- 📝 Generate line-by-line transcripts
- 😊 Analyze sentiments (positive / neutral / negative)
- 🗣️ Classify tone (energetic / neutral / lazy)

---

## 🚀 How to Set Up and Run the Project

### ✅ Step 1: Clone the Repository

```bash
git clone https://github.com/sohrabemam/call_recording_transcript_generation_tone_prediction.git
cd call_recording_transcript_generation_tone_prediction

# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

✅ Step 5: Install FFmpeg (Required for Audio Processing)
Download from: https://ffmpeg.org/download.html

Add FFmpeg to your system PATH
ffmpeg -version

✅ Step 6: Create a .env File with Your OpenAI API Key
In the root project folder, create a file named .env:
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

✅ Step 7: Run the Streamlit App
streamlit run app1.py


📁 call_recording_transcript_generation_tone_prediction/
├── app1.py               # Streamlit app
├── audio_processor.py    # Audio transcription & tone logic
├── requirements.txt
├── .env                  # Your API key (not pushed)
├── .gitignore
