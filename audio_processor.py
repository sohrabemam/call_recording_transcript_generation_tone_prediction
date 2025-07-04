import librosa
import numpy as np
import soundfile as sf
import pandas as pd
from pydub import AudioSegment
from pydub.silence import split_on_silence, detect_silence
import os
import re
from langdetect import detect
import yake
from scipy.fft import fft
from scipy.signal import butter, lfilter
from openai import OpenAI
import logging
from dotenv import load_dotenv
load_dotenv()
# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it to your OpenAI API key.")
# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# Initialize keyword extractor
kw_extractor = yake.KeywordExtractor(lan="en", n=1, dedupLim=0.9, top=1)


def butter_lowpass(cutoff, fs, order=5):
    """Design a lowpass Butterworth filter."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def lowpass_filter(data, cutoff, fs, order=5):
    """Apply a lowpass Butterworth filter to data."""
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def split_audio_by_silence(audio_file, min_silence_len=500, silence_thresh=-30):
    """Split audio into segments based on silence using pydub.silence."""
    logging.info(f"Splitting audio file: {audio_file}")
    try:
        audio = AudioSegment.from_wav(audio_file)
        audio = audio.set_channels(1)
        chunks = split_on_silence(
            audio,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh,
            keep_silence=200
        )
        segments = []
        start_times = []
        current_time = 0
        for i, chunk in enumerate(chunks):
            chunk_path = f"temp_segment_{i}.wav"
            chunk.export(chunk_path, format="wav")
            segments.append(chunk_path)
            start_times.append(current_time / 1000.0)
            current_time += len(chunk)
        logging.debug(f"Created {len(segments)} segments")
        return segments, start_times
    except Exception as e:
        logging.error(f"Error splitting audio: {e}")
        return [], []


def transcribe_audio(audio_path):
    """Transcribe audio file using OpenAI Whisper."""
    logging.info(f"Transcribing audio: {audio_path}")
    try:
        with open(audio_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        text = transcription.text.strip()
        logging.debug(f"Transcription for {audio_path}: {text}")
        return text
    except Exception as e:
        logging.error(f"Transcription error for {audio_path}: {e}")
        return ""


def analyze_sentiment(text):
    """Analyze sentiment using OpenAI GPT-3.5-turbo."""
    if not text or text.strip().isdigit():
        logging.debug(f"No valid text for sentiment analysis: {text}")
        return "NEUTRAL", 0.0
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": "You are a sentiment analysis expert. Classify the sentiment of the given text as POSITIVE, NEGATIVE, or NEUTRAL. Return only the sentiment label in uppercase."},
                {"role": "user", "content": text}
            ],
            max_tokens=10
        )
        sentiment = response.choices[0].message.content.strip().upper()
        if sentiment not in ["POSITIVE", "NEGATIVE", "NEUTRAL"]:
            sentiment = "NEUTRAL"
        logging.debug(f"Sentiment for text '{text}': {sentiment}")
        return sentiment, 1.0
    except Exception as e:
        logging.error(f"Sentiment analysis error for text '{text}': {e}")
        return "NEUTRAL", 0.0


def detect_language_openai(text):
    """Detect language using OpenAI GPT-3.5-turbo."""
    if not text or text.strip().isdigit():
        logging.debug(f"No valid text for language detection: {text}")
        return "UNKNOWN"
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": "You are a language detection expert. Identify the primary language of the given text. Return only the language name in uppercase (e.g., HINDI, ENGLISH)."},
                {"role": "user", "content": text}
            ],
            max_tokens=10
        )
        language = response.choices[0].message.content.strip().upper()
        logging.debug(f"Language for text '{text}': {language}")
        return language
    except Exception as e:
        logging.error(f"Language detection error for text '{text}': {e}")
        return "UNKNOWN"


def analyze_tone_openai(transcription, acoustic_features):
    """Classify tone using OpenAI GPT-3.5-turbo based on transcription and acoustic features."""
    if not transcription:
        logging.debug("No transcription for tone analysis")
        return "LAZY"
    features_summary = (
        f"Intensity: {acoustic_features['intensity_db']:.2f} dB, "
        f"Pitch Mean: {acoustic_features['pitch_mean_hz']:.2f} Hz, "
        f"Pitch Std: {acoustic_features['pitch_std_hz']:.2f} Hz, "
        f"Jitter: {acoustic_features['jitter_percent']:.2f}%, "
        f"Shimmer: {acoustic_features['shimmer_percent']:.2f}%, "
        f"Spectral Tilt: {acoustic_features['spectral_tilt_db']:.2f} dB/octave, "
        f"Speech Rate: {acoustic_features['speech_rate_wpm']:.2f} WPM, "
        f"Avg Pause Duration: {acoustic_features['avg_pause_duration_s']:.2f} s, "
        f"Vocal Fry Ratio: {acoustic_features['vocal_fry_ratio']:.4f}"
    )
    prompt = (
        f"Classify the tone of the following speech as LAZY or ENTHUSIASTIC based on the transcription and acoustic features.\n"
        f"Transcription: {transcription}\n"
        f"Acoustic Features: {features_summary}\n"
        f"LAZY tone typically has low intensity (<60 dB), monotone pitch (std <20 Hz), high jitter/shimmer (>5%), "
        f"steep spectral tilt (<-10 dB/octave), slow speech rate (<120 WPM), long pauses (>0.5 s), and high vocal fry (>0.2).\n"
        f"ENTHUSIASTIC tone has higher intensity, varied pitch, low jitter/shimmer, flatter spectral tilt, faster speech, shorter pauses, and low vocal fry.\n"
        f"Return only the tone in uppercase (LAZY or ENTHUSIASTIC)."
    )
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a tone analysis expert."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=10
        )
        tone = response.choices[0].message.content.strip().upper()
        if tone not in ["LAZY", "ENTHUSIASTIC"]:
            tone = "LAZY"
        logging.debug(f"Tone for transcription '{transcription}': {tone}")
        return tone
    except Exception as e:
        logging.error(f"Tone analysis error for transcription '{transcription}': {e}")
        return "LAZY"


def analyze_acoustic_features(audio_path, transcription):
    """Extract acoustic features for laziness detection."""
    try:
        y, sr = librosa.load(audio_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)

        # 1. Intensity (dB)
        rms = np.mean(librosa.feature.rms(y=y))
        intensity_db = 20 * np.log10(rms + 1e-10) if rms > 0 else -100

        # 2. Pitch (F₀) + Std Dev and Range
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = pitches[magnitudes > 0]
        pitch_values = pitch_values[(pitch_values >= 50) & (pitch_values <= 300)]
        pitch_mean = np.mean(pitch_values) if len(pitch_values) > 0 else 0
        pitch_std = np.std(pitch_values) if len(pitch_values) > 0 else 0
        pitch_range = np.max(pitch_values) - np.min(pitch_values) if len(pitch_values) > 1 else 0

        # 3. Jitter & Shimmer
        if len(pitch_values) > 1:
            jitter = np.mean(np.abs(np.diff(pitch_values)) / (pitch_values[:-1] + 1e-10)) * 100
        else:
            jitter = 0
        amplitudes = np.abs(y) / (np.max(np.abs(y)) + 1e-10)  # Normalize amplitudes
        if len(amplitudes) > 1:
            shimmer = np.mean(np.abs(np.diff(amplitudes)) / (amplitudes[:-1] + 1e-10)) * 100
            shimmer = min(shimmer, 100)  # Cap shimmer
        else:
            shimmer = 0

        # 4. Spectral Tilt
        spectrum = np.abs(fft(y))
        freqs = np.fft.fftfreq(len(spectrum), 1 / sr)
        pos_mask = freqs > 0
        log_freqs = np.log10(freqs[pos_mask] + 1e-10)
        log_spectrum = 20 * np.log10(spectrum[pos_mask] + 1e-10)
        if len(log_freqs) > 1:
            spectral_tilt = np.polyfit(log_freqs, log_spectrum, 1)[0]
            if not np.isfinite(spectral_tilt):
                spectral_tilt = 0
        else:
            spectral_tilt = 0

        # 5. Speech Rate (WPM)
        word_count = len(transcription.split()) if transcription else 0
        speech_rate_wpm = (word_count / duration) * 60 if duration > 0 else 0
        speech_rate_wpm = min(speech_rate_wpm, 300)

        # 6. Pause Patterns
        audio_segment = AudioSegment.from_wav(audio_path)
        silence_regions = detect_silence(audio_segment, min_silence_len=100, silence_thresh=-40)
        pause_durations = [end - start for start, end in silence_regions]
        avg_pause_duration_s = np.mean(pause_durations) / 1000 if pause_durations else 0

        # 7. Vocal Fry Presence and Vocal Effort
        y_low = lowpass_filter(y, cutoff=200, fs=sr)
        low_energy = np.mean(y_low ** 2)
        total_energy = np.mean(y ** 2)
        vocal_fry_ratio = low_energy / (total_energy + 1e-10) if total_energy > 0 else 0
        vocal_effort = intensity_db * (1 - vocal_fry_ratio) if intensity_db > 0 else 0

        # 8. Speech Clarity (based on jitter and shimmer)
        speech_clarity = 100 - (jitter + shimmer) / 2 if jitter >= 0 and shimmer >= 0 else 0

        features = {
            "intensity_db": float(intensity_db),
            "pitch_mean_hz": float(pitch_mean),
            "pitch_std_hz": float(pitch_std),
            "pitch_range": float(pitch_range),
            "jitter_percent": float(jitter),
            "shimmer_percent": float(shimmer),
            "spectral_tilt_db": float(spectral_tilt),
            "speech_rate_wpm": float(speech_rate_wpm),
            "avg_pause_duration_s": float(avg_pause_duration_s),
            "vocal_fry_ratio": float(vocal_fry_ratio),
            "vocal_effort": float(vocal_effort),
            "speech_clarity": float(speech_clarity)
        }
        logging.debug(f"Acoustic features for {audio_path}: {features}")
        return features
    except Exception as e:
        logging.error(f"Error extracting acoustic features for {audio_path}: {e}")
        return {
            "intensity_db": 0.0,
            "pitch_mean_hz": 0.0,
            "pitch_std_hz": 0.0,
            "pitch_range": 0.0,
            "jitter_percent": 0.0,
            "shimmer_percent": 0.0,
            "spectral_tilt_db": 0.0,
            "speech_rate_wpm": 0.0,
            "avg_pause_duration_s": 0.0,
            "vocal_fry_ratio": 0.0,
            "vocal_effort": 0.0,
            "speech_clarity": 0.0
        }


def analyze_tone(features):
    """Classify tone based on acoustic features and return score."""
    score = 0
    if features["intensity_db"] < 60:
        score += 1
    if features["pitch_std_hz"] < 20:
        score += 1
    if features["jitter_percent"] > 5 or features["shimmer_percent"] > 5:
        score += 1
    if features["spectral_tilt_db"] < -10:
        score += 1
    if features["speech_rate_wpm"] < 120:
        score += 1
    if features["avg_pause_duration_s"] > 0.5:
        score += 1
    if features["vocal_fry_ratio"] > 0.2:
        score += 1

    tone = "LAZY" if score >= 4 else "ENTHUSIASTIC"
    return tone, score


def detect_language(text):
    """Detect language using langdetect (for comparison)."""
    if not text or text.strip().isdigit():
        logging.debug(f"No valid text for langdetect: {text}")
        return "UNKNOWN"
    try:
        lang = detect(text)
        if lang in ["id", "hr", "de", "af"]:
            lang = "HI"
        logging.debug(f"Langdetect for text '{text}': {lang.upper()}")
        return lang.upper()
    except Exception as e:
        logging.error(f"Langdetect error for text '{text}': {e}")
        return "UNKNOWN"


def extract_main_keyword(text):
    """Extract the main keyword from the transcribed text."""
    if not text or text.strip().isdigit():
        logging.debug(f"No valid text for keyword extraction: {text}")
        return "NONE"
    try:
        keywords = kw_extractor.extract_keywords(text)
        keyword = keywords[0][0].upper() if keywords else "NONE"
        logging.debug(f"Keyword for text '{text}': {keyword}")
        return keyword
    except Exception as e:
        logging.error(f"Keyword extraction error for text '{text}': {e}")
        return "NONE"


def assign_speaker(transcriptions, segment_index):
    """Assign speaker (Customer/Agent) based on simple alternation heuristic."""
    return "Agent" if segment_index % 2 == 0 else "Customer"


def process_audio_file(audio_file_path):
    """Process the audio file and generate the required output."""
    logging.info(f"Processing audio file: {audio_file_path}")
    segments, start_times = split_audio_by_silence(audio_file_path)

    results = []

    for i, (segment_path, start_time) in enumerate(zip(segments, start_times)):
        logging.info(f"Processing segment {i}: {segment_path}")
        transcription = transcribe_audio(segment_path)
        if not transcription:
            logging.warning(f"No transcription for segment {segment_path}")
            continue

        sentiment_result = analyze_sentiment(transcription)
        if not isinstance(sentiment_result, tuple) or len(sentiment_result) != 2:
            logging.error(f"Invalid sentiment result for segment {i}: {sentiment_result}")
            sentiment, sentiment_score = "NEUTRAL", 0.0
        else:
            sentiment, sentiment_score = sentiment_result

        acoustic_features = analyze_acoustic_features(segment_path, transcription)
        tone, score = analyze_tone(acoustic_features)
        openai_tone = analyze_tone_openai(transcription, acoustic_features)
        language = detect_language(transcription)
        openai_lang = detect_language_openai(transcription)
        main_keyword = extract_main_keyword(transcription)
        speaker = assign_speaker(transcriptions=[t['transcription'] for t in results if 'transcription' in t],
                                 segment_index=i)

        result = {
            "speaker": speaker,
            "transcription": transcription,
            "sentiment": sentiment,
            "sentiment_score": sentiment_score,
            "tone": tone,
            "score": score,
            "LANGUAGE": language,
            "OPENAI_LANG": openai_lang,
            "OPENAI_TONE": openai_tone,
            "MAIN KEYWORD": main_keyword,
            "intensity_db": f"{acoustic_features['intensity_db']:.2f}",
            "pitch_mean_hz": f"{acoustic_features['pitch_mean_hz']:.2f}",
            "pitch_std_hz": f"{acoustic_features['pitch_std_hz']:.2f}",
            "pitch_range": f"{acoustic_features['pitch_range']:.2f}",
            "jitter_percent": f"{acoustic_features['jitter_percent']:.2f}",
            "shimmer_percent": f"{acoustic_features['shimmer_percent']:.2f}",
            "spectral_tilt_db": f"{acoustic_features['spectral_tilt_db']:.2f}",
            "speech_rate_wpm": f"{acoustic_features['speech_rate_wpm']:.2f}",
            "avg_pause_duration_s": f"{acoustic_features['avg_pause_duration_s']:.2f}",
            "vocal_fry_ratio": f"{acoustic_features['vocal_fry_ratio']:.4f}",
            "vocal_effort": f"{acoustic_features['vocal_effort']:.2f}",
            "speech_clarity": f"{acoustic_features['speech_clarity']:.2f}",
            "start_time_s": start_time
        }
        logging.debug(f"Result for segment {i}: {result}")
        results.append(result)

        try:
            os.remove(segment_path)
        except Exception as e:
            logging.error(f"Error removing segment {segment_path}: {e}")

    logging.info(f"Processed {len(results)} segments")

    # Save results to CSV
    if results:
        df = pd.DataFrame(results)
        csv_path = "call_center_analysis.csv"
        try:
            df.to_csv(csv_path, index=False)
            logging.info(f"Saved results to {csv_path}")
        except Exception as e:
            logging.error(f"Error saving CSV: {e}")

    return results


def aggregate_results(results):
    """Aggregate sentiment, tone, and acoustic features for Agent and Customer."""
    if not results:
        logging.warning("No results to aggregate")
        return None, None, None

    df = pd.DataFrame(results)
    logging.debug(f"DataFrame columns: {list(df.columns)}")

    # Verify required columns
    required_columns = ['speaker', 'transcription', 'sentiment', 'sentiment_score', 'tone', 'OPENAI_TONE',
                        'score', 'LANGUAGE', 'OPENAI_LANG', 'MAIN KEYWORD', 'intensity_db', 'pitch_mean_hz',
                        'pitch_std_hz', 'pitch_range', 'jitter_percent', 'shimmer_percent', 'spectral_tilt_db',
                        'speech_rate_wpm', 'avg_pause_duration_s', 'vocal_fry_ratio', 'vocal_effort',
                        'speech_clarity', 'start_time_s']
    for col in required_columns:
        if col not in df.columns:
            logging.warning(f"Column '{col}' missing in DataFrame. Filling with default values.")
            default_value = 'NEUTRAL' if col in ['sentiment', 'tone', 'OPENAI_TONE', 'LANGUAGE', 'OPENAI_LANG',
                                                 'MAIN KEYWORD'] else 0.0
            df[col] = default_value

    # Convert numeric columns and handle invalid values
    numeric_cols = [
        'score', 'sentiment_score', 'intensity_db', 'pitch_mean_hz', 'pitch_std_hz', 'pitch_range',
        'jitter_percent', 'shimmer_percent', 'spectral_tilt_db', 'speech_rate_wpm', 'avg_pause_duration_s',
        'vocal_fry_ratio', 'vocal_effort', 'speech_clarity'
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    # Sentiment aggregation
    def most_common_sentiment(group):
        if 'sentiment' not in group.columns:
            logging.error("Sentiment column missing in group")
            return "NEUTRAL"
        counts = group['sentiment'].value_counts()
        return counts.idxmax() if not counts.empty else "NEUTRAL"

    # Tone aggregation
    def most_common_tone(group):
        if 'tone' not in group.columns:
            logging.error("Tone column missing in group")
            return "ENTHUSIASTIC"
        counts = group['tone'].value_counts()
        return counts.idxmax() if not counts.empty else "ENTHUSIASTIC"

    def most_common_openai_tone(group):
        if 'OPENAI_TONE' not in group.columns:
            logging.error("OPENAI_TONE column missing in group")
            return "ENTHUSIASTIC"
        counts = group['OPENAI_TONE'].value_counts()
        return counts.idxmax() if not counts.empty else "ENTHUSIASTIC"

    # Group by speaker
    try:
        grouped = df.groupby('speaker').agg({
            'sentiment': most_common_sentiment,
            'tone': most_common_tone,
            'OPENAI_TONE': most_common_openai_tone,
            **{col: 'mean' for col in numeric_cols}
        }).reset_index()
        # Ensure Agent and Customer are present
        speakers = ['Agent', 'Customer']
        existing_speakers = grouped['speaker'].tolist()
        for speaker in speakers:
            if speaker not in existing_speakers:
                default_row = {
                    'speaker': speaker,
                    'sentiment': 'NEUTRAL',
                    'tone': 'ENTHUSIASTIC',
                    'OPENAI_TONE': 'ENTHUSIASTIC',
                    **{col: 0.0 for col in numeric_cols}
                }
                grouped = pd.concat([grouped, pd.DataFrame([default_row])], ignore_index=True)
        grouped = grouped.sort_values(by='speaker', key=lambda x: x.map({'Agent': 0, 'Customer': 1}))
    except Exception as e:
        logging.error(f"Error during grouping: {e}")
        grouped = pd.DataFrame({
            'speaker': ['Agent', 'Customer'],
            'sentiment': ['NEUTRAL', 'NEUTRAL'],
            'tone': ['ENTHUSIASTIC', 'ENTHUSIASTIC'],
            'OPENAI_TONE': ['ENTHUSIASTIC', 'ENTHUSIASTIC'],
            **{col: [0.0, 0.0] for col in numeric_cols}
        })

    # Overall aggregation
    overall = {
        'sentiment': most_common_sentiment(df),
        'tone': most_common_tone(df),
        'OPENAI_TONE': most_common_openai_tone(df),
        **{col: df[col].mean() for col in numeric_cols if col in df.columns}
    }
    overall['avg_sentiment_score'] = df['sentiment_score'].mean()  # Add average sentiment score
    overall['avg_tone_score'] = df['score'].mean()  # Add average tone score
    overall['segment_count'] = len(df)
    overall['duration_s'] = df['start_time_s'].max() if not df.empty else 0.0

    return grouped, overall, df


def generate_conclusion(overall):
    """Generate a conclusion explaining the overall tone with additional acoustic insights."""
    if not overall:
        return "No data available for analysis."

    tone = overall['tone']
    reasons = []
    if overall['intensity_db'] < 60:
        reasons.append(f"Low intensity ({overall['intensity_db']:.2f} dB < 60 dB) suggests a lack of vocal energy.")
    if overall['pitch_std_hz'] < 20:
        reasons.append(f"Low pitch variation ({overall['pitch_std_hz']:.2f} Hz < 20 Hz) indicates monotony.")
    if overall['jitter_percent'] > 5 or overall['shimmer_percent'] > 5:
        reasons.append(
            f"High jitter ({overall['jitter_percent']:.2f}%) or shimmer ({overall['shimmer_percent']:.2f}%) suggests a rough voice.")
    if overall['spectral_tilt_db'] < -10:
        reasons.append(
            f"Steep spectral tilt ({overall['spectral_tilt_db']:.2f} dB/octave < -10) indicates a dull sound.")
    if overall['speech_rate_wpm'] < 120:
        reasons.append(f"Slow speech rate ({overall['speech_rate_wpm']:.2f} WPM < 120) suggests disengagement.")
    if overall['avg_pause_duration_s'] > 0.5:
        reasons.append(f"Long pauses ({overall['avg_pause_duration_s']:.2f} s > 0.5 s) indicate hesitation.")
    if overall['vocal_fry_ratio'] > 0.2:
        reasons.append(f"High vocal fry ratio ({overall['vocal_fry_ratio']:.4f} > 0.2) suggests a creaky voice.")

    additional_insights = []
    if overall['pitch_range'] < 50:
        additional_insights.append(
            f"Limited pitch range ({overall['pitch_range']:.2f} Hz) indicates reduced expressiveness.")
    if overall['vocal_effort'] < 20:
        additional_insights.append(
            f"Low vocal effort ({overall['vocal_effort']:.2f}) suggests minimal energy exertion.")
    if overall['speech_clarity'] < 80:
        additional_insights.append(
            f"Low speech clarity ({overall['speech_clarity']:.2f}%) suggests potential articulation issues.")

    conclusion = f"The overall tone is {tone}."
    if tone == "LAZY" and reasons:
        conclusion += f" Reasons: {' '.join(reasons)}"
        if additional_insights:
            conclusion += f" Additional insights: {' '.join(additional_insights)}"
    elif tone == "ENTHUSIASTIC":
        conclusion += f" The speech exhibits high energy, varied pitch, and fluent delivery, indicating engagement."
        if additional_insights:
            conclusion += f" Additional insights: {' '.join(additional_insights)}"
    return conclusion