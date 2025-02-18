import speech_recognition as sr
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import librosa
import numpy as np
import torch
from transformers import pipeline

# Step 1: Voice Input Collection
def record_audio():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Please speak...")
        audio = recognizer.listen(source)
        print("Recording complete.")
    return audio

# Step 2: Speech-to-Text Conversion
def speech_to_text(audio):
    recognizer = sr.Recognizer()
    try:
        text = recognizer.recognize_google(audio)
        print(f"Transcribed Text: {text}")
        return text
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand the audio.")
        return None
    except sr.RequestError:
        print("Could not request results from Google Speech Recognition service.")
        return None

# Step 3: Sentiment Analysis with VADER
def analyze_sentiment(text):
    if text:
        analyzer = SentimentIntensityAnalyzer()
        sentiment = analyzer.polarity_scores(text)
        print(f"Sentiment: {sentiment}")
        return sentiment
    return None

# Step 4: Emotion Detection with Hugging Face Transformers
def detect_emotion(text):
    if text:
        emotion_classifier = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions")
        emotions = emotion_classifier(text)
        print(f"Detected Emotions: {emotions}")
        return emotions
    return None

# Step 5: Voice Analysis (Advanced Features)
def analyze_voice(audio_file):
    # Load audio file
    y, sr = librosa.load(audio_file, sr=None)
    
    # Extract features
    pitch = librosa.yin(y, fmin=50, fmax=2000)  # Pitch estimation
    pitch_mean = np.mean(pitch)
    pitch_std = np.std(pitch)
    
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)  # Tempo estimation
    
    # Additional features
    jitter = np.mean(np.abs(np.diff(pitch)))  # Jitter (pitch variability)
    shimmer = librosa.feature.rms(y=y)[0].mean()  # Shimmer (amplitude variability)
    
    print(f"Voice Analysis: Mean Pitch={pitch_mean}, Pitch Std={pitch_std}, Tempo={tempo}, Jitter={jitter}, Shimmer={shimmer}")
    return {
        "mean_pitch": pitch_mean,
        "pitch_std": pitch_std,
        "tempo": tempo,
        "jitter": jitter,
        "shimmer": shimmer
    }

# Step 6: Mental Health Assessment (Improved)
def mental_health_assessment(sentiment, emotions, voice_features):
    if sentiment is None or emotions is None or voice_features is None:
        return "Insufficient data for assessment."
    
    # Mood based on sentiment
    if sentiment["compound"] >= 0.05:
        mood = "Positive"
    elif sentiment["compound"] <= -0.05:
        mood = "Negative"
    else:
        mood = "Neutral"
    
    # Emotion detection
    top_emotion = emotions[0]["label"]
    
    # Voice analysis
    if voice_features["jitter"] > 0.1 or voice_features["shimmer"] > 0.1:
        voice_state = "High Stress"
    else:
        voice_state = "Normal"
    
    return f"Mood: {mood}, Top Emotion: {top_emotion}, Voice State: {voice_state}"

# Main Function
def main():
    # Step 1: Record audio
    audio = record_audio()
    
    # Save audio to a file (optional)
    with open("recorded_audio.wav", "wb") as f:
        f.write(audio.get_wav_data())
    
    # Step 2: Convert speech to text
    text = speech_to_text(audio)
    
    # Step 3: Analyze sentiment
    sentiment = analyze_sentiment(text)
    
    # Step 4: Detect emotions
    emotions = detect_emotion(text)
    
    # Step 5: Analyze voice
    voice_features = analyze_voice("recorded_audio.wav")
    
    # Step 6: Mental health assessment
    assessment = mental_health_assessment(sentiment, emotions, voice_features)
    print(f"Mental Health Assessment: {assessment}")

if __name__ == "__main__":
    main()