import cv2
import json
import queue
import wave
import threading
import numpy as np
import tensorflow as tf
import sounddevice as sd

from pathlib import Path
from collections import deque, Counter
from vosk import Model, KaldiRecognizer

# =========================================================
# PATHS
# =========================================================
MODEL_PATH = Path("models/emotion_cnn.keras")
LABELS_PATH = Path("models/labels.txt")
VOSK_MODEL_PATH = Path("models/vosk-model-small-en-us-0.15")

# =========================================================
# FACE CONFIG
# =========================================================
IMG_W, IMG_H = 48, 48

MIN_FACE_SIZE = 120
MIN_FACE_AREA_RATIO = 0.06
ASPECT_MIN, ASPECT_MAX = 0.75, 1.35
MIN_MEAN_INTENSITY = 35
MIN_STD_INTENSITY = 18
SKIN_YCRCB_MIN = 0.18
EYE_REQUIRED = False

# =========================================================
# AUDIO CONFIG
# =========================================================
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"
AUDIO_BLOCKSIZE = 1000

# low-voice boost
MIC_GAIN = 5.0
MIN_AUDIO_LEVEL = 0.3
NORMALIZE_AUDIO = True

# =========================================================
# GLOBALS
# =========================================================
audio_queue = queue.Queue()
audio_chunks = []
stop_audio_event = threading.Event()

last_mic_volume = 0.0
selected_mic_name = "Unknown"

# =========================================================
# HELPERS
# =========================================================
def load_labels():
    if not LABELS_PATH.exists():
        raise FileNotFoundError(f"labels.txt not found at: {LABELS_PATH}")
    return [line.strip() for line in LABELS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def model_has_rescaling(model: tf.keras.Model) -> bool:
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Rescaling):
            return True
    return False


def preprocess_face(gray_face, use_internal_rescaling: bool):
    face = cv2.resize(gray_face, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    if use_internal_rescaling:
        face = face.astype(np.float32)
    else:
        face = face.astype(np.float32) / 255.0
    face = np.expand_dims(face, axis=-1)
    face = np.expand_dims(face, axis=0)
    return face


def skin_ratio_ycrcb(bgr_roi):
    ycrcb = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower, upper)
    return float(np.count_nonzero(mask)) / float(mask.size + 1e-6)


def passes_face_filters(frame_bgr, gray_eq, x1, y1, x2, y2, frame_area, eye_cascade=None):
    w = x2 - x1
    h = y2 - y1

    if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
        return False

    if (w * h) / float(frame_area) < MIN_FACE_AREA_RATIO:
        return False

    ar = w / float(h + 1e-6)
    if not (ASPECT_MIN <= ar <= ASPECT_MAX):
        return False

    roi_gray = gray_eq[y1:y2, x1:x2]
    if roi_gray.size == 0:
        return False

    mean_i = float(np.mean(roi_gray))
    std_i = float(np.std(roi_gray))
    if mean_i < MIN_MEAN_INTENSITY or std_i < MIN_STD_INTENSITY:
        return False

    roi_bgr = frame_bgr[y1:y2, x1:x2]
    if roi_bgr.size == 0:
        return False

    if skin_ratio_ycrcb(roi_bgr) < SKIN_YCRCB_MIN:
        return False

    if EYE_REQUIRED and eye_cascade is not None:
        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.2,
            minNeighbors=8,
            minSize=(30, 30)
        )
        if len(eyes) < 2:
            return False

    return True


# =========================================================
# AUDIO DEVICE SELECTION
# =========================================================
def choose_input_device():
    global selected_mic_name

    devices = sd.query_devices()

    # 1) default input if valid
    default_in = sd.default.device[0]
    if default_in is not None and default_in >= 0:
        d = devices[default_in]
        if d["max_input_channels"] > 0:
            selected_mic_name = d["name"]
            return default_in

    # 2) prefer built-in/macbook mic
    for i, d in enumerate(devices):
        if d["max_input_channels"] <= 0:
            continue
        name = d["name"].lower()
        if ("microphone" in name or "macbook" in name or "built-in" in name) and ("teams" not in name):
            selected_mic_name = d["name"]
            return i

    # 3) fallback to any input except teams
    for i, d in enumerate(devices):
        if d["max_input_channels"] <= 0:
            continue
        name = d["name"].lower()
        if "teams" not in name:
            selected_mic_name = d["name"]
            return i

    raise RuntimeError("No usable input microphone found.")


# =========================================================
# TRANSCRIPT CLEANUP + DISPLAY HELPERS
# =========================================================
def clean_transcript(text: str):
    if not text:
        return ""

    text = text.lower().strip()

    # remove consecutive repeated words
    words = text.split()
    cleaned_words = []
    for w in words:
        if not cleaned_words or cleaned_words[-1] != w:
            cleaned_words.append(w)
    text = " ".join(cleaned_words)

    # common fixes
    replacements = {
        "im ": "i am ",
        "i m ": "i am ",
        "dont": "don't",
        "cant": "can't",
        "wont": "won't",
        "ive": "i have",
        "id": "i would",
        "doesnt": "doesn't",
        "isnt": "isn't",
        "arent": "aren't",
        "wasnt": "wasn't",
        "werent": "weren't",
        "couldnt": "couldn't",
        "shouldnt": "shouldn't",
        "wouldnt": "wouldn't",
    }

    text = " " + text + " "
    for old, new in replacements.items():
        text = text.replace(f" {old}", f" {new}")
    text = text.strip()

    # capitalize first letter
    if text:
        text = text[0].upper() + text[1:]

    return text


def wrap_text(text, max_chars=72):
    words = text.split()
    if not words:
        return [""]

    lines = []
    current = ""

    for word in words:
        trial = word if not current else current + " " + word
        if len(trial) <= max_chars:
            current = trial
        else:
            lines.append(current)
            current = word

    if current:
        lines.append(current)

    return lines


# =========================================================
# TEXT EMOTION ANALYSIS
# =========================================================
def analyze_text_emotion(text: str):
    text = text.lower().strip()

    if not text:
        return "unknown", {}, 0.0

    emotion_keywords = {
        "happy": [
            "happy", "good", "great", "excited", "joy", "love", "fine",
            "better", "calm", "relaxed", "nice", "amazing", "wonderful"
        ],
        "sad": [
            "sad", "depressed", "depression", "low", "lonely", "cry",
            "crying", "hurt", "upset", "hopeless", "miserable",
            "heartbroken", "empty", "bad"
        ],
        "angry": [
            "angry", "mad", "annoyed", "frustrated", "hate",
            "irritated", "furious", "rage"
        ],
        "fear": [
            "afraid", "scared", "fear", "worried", "anxious", "anxiety",
            "panic", "nervous", "stress", "stressed", "pressure",
            "overwhelmed", "exam", "exams", "tension"
        ],
        "surprise": [
            "surprised", "shock", "shocked", "unexpected", "wow", "omg"
        ],
        "neutral": [
            "okay", "ok", "normal", "usual", "alright"
        ]
    }

    scores = {k: 0 for k in emotion_keywords}
    words = text.split()

    for emotion, keywords in emotion_keywords.items():
        for w in words:
            if w in keywords:
                scores[emotion] += 1

    phrase_rules = [
        ("i feel very depressed", "sad", 8),
        ("i feel depressed", "sad", 7),
        ("i am depressed", "sad", 7),
        ("very depressed", "sad", 7),
        ("i feel very sad", "sad", 8),
        ("i feel sad", "sad", 7),
        ("i am sad", "sad", 7),
        ("very sad", "sad", 7),
        ("i feel lonely", "sad", 7),
        ("i feel low", "sad", 6),
        ("i feel anxious", "fear", 7),
        ("i am anxious", "fear", 7),
        ("i feel stressed", "fear", 7),
        ("i am stressed", "fear", 7),
        ("very stressed", "fear", 7),
        ("exam stress", "fear", 7),
        ("stressed about exams", "fear", 8),
        ("i am worried", "fear", 7),
        ("i feel worried", "fear", 7),
        ("i am nervous", "fear", 7),
        ("i feel happy", "happy", 7),
        ("i am happy", "happy", 7),
        ("very happy", "happy", 7),
        ("i am angry", "angry", 7),
        ("i feel angry", "angry", 7),
        ("very angry", "angry", 7),
    ]

    for phrase, emotion, boost in phrase_rules:
        if phrase in text:
            scores[emotion] += boost

    sadness_terms = ["depressed", "depression", "sad", "lonely", "hopeless", "crying", "upset"]
    if any(term in text for term in sadness_terms):
        scores["sad"] += 5

    stress_terms = ["exam", "exams", "stressed", "stress", "pressure", "anxious", "worried", "nervous"]
    if any(term in text for term in stress_terms):
        scores["fear"] += 4

    if "depressed" in text or "very sad" in text or "i feel sad" in text or "i am sad" in text:
        scores["sad"] += 3

    best_emotion = max(scores, key=scores.get)
    best_score = scores[best_emotion]
    sorted_scores = sorted(scores.values(), reverse=True)
    second_score = sorted_scores[1] if len(sorted_scores) > 1 else 0

    if best_score == 0:
        return "unknown", scores, 0.0

    gap = best_score - second_score
    word_count = len(words)

    confidence = 0.45 + min(gap * 0.08, 0.35)

    if word_count < 4:
        confidence -= 0.20
    elif word_count < 7:
        confidence -= 0.10

    confidence = max(0.0, min(confidence, 0.95))
    return best_emotion, scores, confidence


def fuse_emotions(face_emotion: str, voice_emotion: str, voice_confidence: float):
    if voice_emotion not in ("unknown", "neutral") and voice_confidence >= 0.15:
        return voice_emotion

    if voice_emotion not in ("unknown", "") and face_emotion in ("unknown", "neutral"):
        return voice_emotion

    if face_emotion not in ("unknown", ""):
        return face_emotion

    if voice_emotion not in ("unknown", ""):
        return voice_emotion

    return "unknown"


# =========================================================
# AUDIO RECORDING
# =========================================================
def audio_callback(indata, frames, time_info, status):
    global last_mic_volume

    if status:
        print("Audio status:", status)

    audio = indata.copy().astype(np.float32)

    raw_volume = float(np.linalg.norm(audio) / max(len(audio), 1))
    last_mic_volume = raw_volume

    # boost low voice
    audio *= MIC_GAIN

    # normalize softly
    if NORMALIZE_AUDIO:
        peak = np.max(np.abs(audio))
        if peak > 0:
            target_peak = 12000.0
            scale = min(target_peak / peak, 8.0)
            audio *= scale

    audio = np.clip(audio, -32768, 32767).astype(np.int16)

    boosted_volume = float(np.linalg.norm(audio.astype(np.float32)) / max(len(audio), 1))
    if boosted_volume >= MIN_AUDIO_LEVEL:
        audio_queue.put(audio)


def audio_recorder_worker():
    try:
        mic_idx = choose_input_device()
        print(f"🎤 Using microphone device: {mic_idx} -> {selected_mic_name}")

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=AUDIO_BLOCKSIZE,
            dtype=DTYPE,
            channels=CHANNELS,
            device=mic_idx,
            callback=audio_callback
        ):
            print("🎤 Microphone listening...")

            while not stop_audio_event.is_set():
                try:
                    chunk = audio_queue.get(timeout=0.2)
                    audio_chunks.append(chunk.copy())
                except queue.Empty:
                    continue

    except Exception as e:
        print("Audio recorder error:", e)


# =========================================================
# FULL-SESSION TRANSCRIPTION
# =========================================================
def transcribe_recorded_audio():
    if not VOSK_MODEL_PATH.exists():
        raise FileNotFoundError(f"Vosk model folder not found: {VOSK_MODEL_PATH}")

    if not audio_chunks:
        return ""

    vosk_model = Model(str(VOSK_MODEL_PATH))
    recognizer = KaldiRecognizer(vosk_model, SAMPLE_RATE)
    recognizer.SetWords(True)

    parts = []
    seen = set()

    for chunk in audio_chunks:
        data_bytes = chunk.tobytes()
        if recognizer.AcceptWaveform(data_bytes):
            result = json.loads(recognizer.Result())
            text = result.get("text", "").strip()
            if text and text not in seen:
                parts.append(text)
                seen.add(text)

    final_result = json.loads(recognizer.FinalResult())
    final_text = final_result.get("text", "").strip()
    if final_text and final_text not in seen:
        parts.append(final_text)

    transcript = " ".join(parts).strip()
    return clean_transcript(transcript)


def save_debug_wav(path="debug_recording.wav"):
    if not audio_chunks:
        return

    all_audio = np.concatenate(audio_chunks, axis=0).astype(np.float32)

    peak = np.max(np.abs(all_audio))
    if peak > 0:
        all_audio *= min(15000.0 / peak, 6.0)

    all_audio = np.clip(all_audio, -32768, 32767).astype(np.int16)

    with wave.open(path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(all_audio.tobytes())


# =========================================================
# MAIN
# =========================================================
def main():
    global last_mic_volume, audio_queue

    audio_chunks.clear()
    last_mic_volume = 0.0
    stop_audio_event.clear()
    audio_queue = queue.Queue()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Emotion model not found: {MODEL_PATH}")
    if not LABELS_PATH.exists():
        raise FileNotFoundError(f"Labels file not found: {LABELS_PATH}")
    if not VOSK_MODEL_PATH.exists():
        raise FileNotFoundError(f"Vosk model folder not found: {VOSK_MODEL_PATH}")

    labels = load_labels()
    model = tf.keras.models.load_model(MODEL_PATH)
    use_internal_rescaling = model_has_rescaling(model)

    print("✅ Emotion model loaded")
    print("✅ Labels:", labels)
    print("✅ Internal rescaling layer:", use_internal_rescaling)
    print("\nAvailable audio devices:")
    print(sd.query_devices())

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if face_cascade.empty():
        raise RuntimeError("Failed to load face cascade.")

    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    if eye_cascade.empty():
        eye_cascade = None

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    audio_thread = threading.Thread(target=audio_recorder_worker, daemon=True)
    audio_thread.start()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        stop_audio_event.set()
        raise RuntimeError("Cannot access webcam.")

    smooth_N = 8
    prob_queue = deque(maxlen=smooth_N)
    face_emotion_history = []

    print("✅ Webcam started. Speak while camera is open.")
    print("Press 'q' to stop and get final fused emotion.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_area = frame.shape[0] * frame.shape[1]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_eq = clahe.apply(gray)

        faces = face_cascade.detectMultiScale(
            gray_eq,
            scaleFactor=1.05,
            minNeighbors=6,
            minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE)
        )

        best = None
        best_area = 0

        for (x, y, w, h) in faces:
            pad = int(0.20 * w)
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(gray_eq.shape[1], x + w + pad)
            y2 = min(gray_eq.shape[0], y + h + pad)

            if not passes_face_filters(frame, gray_eq, x1, y1, x2, y2, frame_area, eye_cascade):
                continue

            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                best = (x1, y1, x2, y2)

        if best is not None:
            x1, y1, x2, y2 = best
            face_gray = gray_eq[y1:y2, x1:x2]
            inp = preprocess_face(face_gray, use_internal_rescaling)

            probs = model.predict(inp, verbose=0)[0].astype(np.float32)
            prob_queue.append(probs)
            probs_smoothed = np.mean(np.stack(prob_queue, axis=0), axis=0)

            idx = int(np.argmax(probs_smoothed))
            conf = float(probs_smoothed[idx])
            label = labels[idx] if idx < len(labels) else str(idx)
            face_emotion_history.append(label)

            top2 = np.argsort(probs_smoothed)[-2:][::-1]
            top2_text = " | ".join([f"{labels[i]} {probs_smoothed[i]*100:.0f}%" for i in top2])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Face: {label} ({conf*100:.1f}%)", (x1, max(25, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, top2_text, (x1, min(frame.shape[0] - 10, y2 + 22)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            prob_queue.clear()
            cv2.putText(frame, "No HUMAN face detected", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        mic_text = f"Mic: {selected_mic_name}"
        vol_text = f"Mic level: {last_mic_volume:.2f}"

        cv2.putText(frame, mic_text[:70], (10, frame.shape[0] - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
        cv2.putText(frame, vol_text, (10, frame.shape[0] - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
        cv2.putText(frame, "Speak while camera is open. Press q to finish", (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        cv2.imshow("Multimodal Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    stop_audio_event.set()
    audio_thread.join(timeout=2.0)

    save_debug_wav("debug_recording.wav")
    transcript = transcribe_recorded_audio()

    if face_emotion_history:
        dominant_face_emotion = Counter(face_emotion_history).most_common(1)[0][0]
    else:
        dominant_face_emotion = "unknown"

    voice_emotion, voice_scores, voice_confidence = analyze_text_emotion(transcript)
    final_emotion = fuse_emotions(dominant_face_emotion, voice_emotion, voice_confidence)

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print("Transcript:", transcript if transcript else "[No speech captured]")
    print("Dominant facial emotion:", dominant_face_emotion)
    print("Voice/text emotion:", voice_emotion)
    print("Voice confidence:", round(voice_confidence, 3))
    print("Final fused emotion:", final_emotion)
    print("=" * 60)

    result = np.zeros((560, 1000, 3), dtype=np.uint8)

    cv2.putText(result, "Session Result", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    cv2.putText(result, f"Facial Emotion : {dominant_face_emotion}", (30, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.putText(result, f"Voice Emotion  : {voice_emotion}", (30, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 200, 0), 2)

    cv2.putText(result, f"Voice Confidence : {voice_confidence:.2f}", (30, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (180, 180, 255), 2)

    cv2.putText(result, f"Final Emotion  : {final_emotion}", (30, 310),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3)

    shown_transcript = transcript if transcript else "No speech captured"
    chunks = wrap_text(shown_transcript, max_chars=72)[:5]

    cv2.putText(result, "Transcript:", (30, 380),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

    y = 420
    for line in chunks:
        cv2.putText(result, line, (30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
        y += 28

    cv2.putText(result, "Audio saved as debug_recording.wav", (30, 545),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 255), 2)

    cv2.imshow("Multimodal Emotion Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
