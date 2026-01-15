import os
import time
import uuid
import re
import json
import requests
import multiprocessing as mp
import azure.cognitiveservices.speech as speechsdk
import torch
import subprocess
import pyaudio
import numpy as np

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

secrets = json.load(open("secrets.json", "r"))

SPEECH_KEY = secrets["speech_key"]
SPEECH_REGION = secrets.get("SERVICE_LOCATION", "eastus")

TRANSLATOR_KEY = secrets["translate_key"]
TRANSLATOR_REGION = secrets.get("translate_region", "global")

SOURCE_LANGUAGE = "en-US"
TARGET_LANGUAGE = "zh-CN"
TTS_VOICE = "zh-CN-XiaoxiaoNeural"
# TARGET_LANGUAGE = "pt-BR"
# TTS_VOICE = "pt-BR-FranciscaNeural"

AUDIO_FILE = r"C:\Research\TranslateCapstone\browning.mp4"

# --- PROCESSING CONFIG ---
MIN_WORDS_FOR_SILERO = 6   
LAG_BUFFER_WORDS = 1       

# --- DYNAMIC SPEED CONFIG ---
BASE_TTS_RATE = "1.4"
FAST_TTS_RATE = "1.9"
QUEUE_THRESHOLD = 1        

import csv
from pathlib import Path

CSV_PATH = Path(f"latency_log{TARGET_LANGUAGE}.csv")

CSV_HEADER = [
    "id",
    "reason",
    "asr_latency_ms",
    "post_asr_latency_ms",
    "playback_latency_ms",
    "total_latency_ms"
]

if not CSV_PATH.exists():
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(CSV_HEADER)


# -------------------------------------------------------------------
# SILERO
# -------------------------------------------------------------------

def load_silero():
    print("[INIT] Loading Silero punctuation...")
    model, _, _, _, apply_te = torch.hub.load(
        repo_or_dir="snakers4/silero-models",
        model="silero_te",
        trust_repo=True,
        verbose=False
    )
    return apply_te

# -------------------------------------------------------------------
# AUDIO PUMP (Main Process)
# -------------------------------------------------------------------

def pump_audio_from_file(push_stream, filename):
    print(f"[STREAM] Extracting audio from {filename}...")
    
    command = ['ffmpeg', '-i', filename, '-f', 's16le', '-ac', '1', '-ar', '16000', '-vn', '-']
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    
    CHUNK_SIZE = 3200 
    SLEEP_DURATION = (CHUNK_SIZE / 32000) * 0.85 

    try:
        while True:
            chunk = process.stdout.read(CHUNK_SIZE)
            if not chunk: break
            if np.max(np.frombuffer(chunk, dtype=np.int16)) < 200:
                noise = (np.random.randn(len(chunk)//2) * 50).astype(np.int16)
                chunk = noise.tobytes() # simulate noise
            push_stream.write(chunk)
            time.sleep(SLEEP_DURATION) 
    except Exception as e:
        print(f"[STREAM ERROR] {e}")
    finally:
        push_stream.close()
        process.terminate()
        print("[STREAM] Audio finished.")

# -------------------------------------------------------------------
# WORKERS (Independent Processes)
# -------------------------------------------------------------------

def mt_worker(mt_q, tts_q):
    # Re-init session inside process
    session = requests.Session()
    print("✅ MT Worker Started")
    
    while True:
        item = mt_q.get()
        if item is None: break
        
        text = item["text"]
        if not text.strip():
            tts_q.put(item) # Pass empty logic
            continue

        try:
            r = session.post(
                "https://api.cognitive.microsofttranslator.com/translate",
                params={"api-version": "3.0", "to": TARGET_LANGUAGE},
                headers={
                    "Ocp-Apim-Subscription-Key": TRANSLATOR_KEY,
                    "Ocp-Apim-Subscription-Region": TRANSLATOR_REGION,
                    "Content-Type": "application/json"
                },
                json=[{"text": text}],
                timeout=5
            )
            r.raise_for_status()
            item["translated"] = r.json()[0]["translations"][0]["text"]
        except Exception as e:
            print(f"[MT FAIL] {e}")
            item["translated"] = text # Fallback

        tts_q.put(item)

def tts_worker(tts_q, audio_q):
    print("✅ TTS Worker Started")
    
    # Init Azure inside the process
    speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    speech_config.speech_synthesis_voice_name = TTS_VOICE
    
    # FIX: Set audio_config to None for in-memory generation
    synth = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)

    while True:
        item = tts_q.get()
        if item is None: break

        backlog = tts_q.qsize()
        current_rate = FAST_TTS_RATE if backlog > QUEUE_THRESHOLD else BASE_TTS_RATE
        
        txt = item["translated"]
        print(f"[TTS] Generating: '{txt[:15]}...' (Backlog: {backlog})")
        
        ssml = f"""
        <speak version="1.0" xml:lang="{TARGET_LANGUAGE}">
            <voice name="{TTS_VOICE}">
            <prosody rate="{current_rate}">{txt}</prosody>
            </voice>
        </speak>
        """
        
        # This blocks only for network download (~200ms), NOT playback (5s)
        result = synth.speak_ssml_async(ssml).get()
        
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            # Strip 44-byte WAV header for smoother streaming
            raw_bytes = result.audio_data[44:] if len(result.audio_data) > 44 else result.audio_data
            item["audio"] = raw_bytes
            # audio_q.put(item)
            
            # Log latency NOW (when bytes are ready)
            t_tts_ready = time.perf_counter()
            item["t_tts_ready"] = t_tts_ready

            audio_q.put(item)  # send metadata + audio together
        else:
            print(f"[TTS ERROR] {result.cancellation_details.error_details}")

def playback_worker(audio_q):
    print("✅ Playback Worker Started")
    p = pyaudio.PyAudio()
    
    # Open stream safely
    try:
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, output=True)
    except Exception as e:
        print(f"[PLAYBACK INIT ERROR] {e}")
        return

    while True:
        item = audio_q.get()
        if item is None:
            break

        audio = item["audio"]
        t_play_start = time.perf_counter()

        stream.write(audio)

        # ⏱️ Compute latencies
        asr_latency = (item["t_asr_flush"] - item["t_asr_start"]) * 1000
        post_asr_latency = (item["t_tts_ready"] - item["t_asr_flush"]) * 1000
        playback_latency = (t_play_start - item["t_tts_ready"]) * 1000
        total_latency = (t_play_start - item["t_asr_start"]) * 1000

        # 🧾 Write CSV row
        with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                item["id"],
                item["reason"],
                f"{asr_latency:.1f}",
                f"{post_asr_latency:.1f}",
                f"{playback_latency:.1f}",
                f"{total_latency:.1f}",
            ])

        print(
            f"[{item['id']}] ⏱️ "
            f"ASR {asr_latency:.0f}ms | "
            f"Post-ASR {post_asr_latency:.0f}ms | "
            f"Play {playback_latency:.0f}ms | "
            f"Total {total_latency:.0f}ms"
        )

    
    stream.stop_stream()
    stream.close()
    p.terminate()

# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------

def main():
    apply_te = load_silero()

    # Use standard Multiprocessing Queues
    mt_q = mp.Queue()
    tts_q = mp.Queue() 
    audio_q = mp.Queue() 

    # Launch Processes
    p_mt = mp.Process(target=mt_worker, args=(mt_q, tts_q))
    p_tts = mp.Process(target=tts_worker, args=(tts_q, audio_q))
    p_play = mp.Process(target=playback_worker, args=(audio_q,))

    p_mt.start()
    p_tts.start()
    p_play.start()

    # Setup ASR
    speech_cfg = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    speech_cfg.speech_recognition_language = SOURCE_LANGUAGE
    speech_cfg.set_property(speechsdk.PropertyId.SpeechServiceResponse_StablePartialResultThreshold, "1")

    # Audio Input Pump (Thread in Main Process)
    push_stream = speechsdk.audio.PushAudioInputStream()
    audio_cfg = speechsdk.audio.AudioConfig(stream=push_stream)
    
    import threading
    threading.Thread(target=pump_audio_from_file, args=(push_stream, AUDIO_FILE), daemon=True).start()

    recognizer = speechsdk.SpeechRecognizer(speech_cfg, audio_cfg)

    state = {
        "words": [],
        "idx": 0,
        "t0": None,
        "last_clean": "",
        "last_punct": ""
    }

    def clean(t): return re.sub(r"[^\w\s]", "", t).lower()

    def flush(text, reason):
        now = time.perf_counter()

        item = {
            "id": str(uuid.uuid4())[:6],
            "text": text,
            "reason": reason,

            # ⏱️ ASR timing
            "t_asr_start": state["t0"],
            "t_asr_flush": now
        }

        mt_q.put(item)

        print(f"[{reason}] {text}")



    def process(text, final=False):
        if state["t0"] is None:
            state["t0"] = time.perf_counter()

        words = text.split()
        if len(words) <= state["idx"]:
            return

        new = words[state["idx"]:]
        combined = " ".join(new)

        if len(new) >= MIN_WORDS_FOR_SILERO:
            c = clean(combined)
            if c == state["last_clean"]:
                punct = state["last_punct"]
            else:
                punct = apply_te(c, lan="en")
                state["last_clean"] = c
                state["last_punct"] = punct
        else:
            punct = combined

        flushed_any = False

        # for m in re.finditer(r"[,.?!]", punct):
        for m in re.finditer(r"[.?!]", punct): # Testing chinese
            tail = punct[m.end():].split()

            if not final and len(tail) < LAG_BUFFER_WORDS:
                continue

            segment = punct[:m.end()].strip()
            wc = len(segment.split())

            state["idx"] += wc
            flush(segment, "EARLY")
            state["t0"] = time.perf_counter()
            flushed_any = True

        # 🔑 FINAL must only flush what remains
        if final:
            remaining_words = words[state["idx"]:]
            if remaining_words:
                remaining = " ".join(remaining_words).strip()
                if remaining:
                    flush(remaining, "FINAL")

            # Reset for next utterance
            state["idx"] = 0
            state["t0"] = None
            state["last_clean"] = ""
            state["last_punct"] = ""


    recognizer.recognizing.connect(lambda e: process(e.result.text))
    recognizer.recognized.connect(lambda e: process(e.result.text, True))

    print("🚀 Live translation started (Fully Decoupled)")
    recognizer.start_continuous_recognition_async().get()

    try:
        while True: time.sleep(0.5)
    except KeyboardInterrupt:
        recognizer.stop_continuous_recognition_async().get()
        mt_q.put(None)
        tts_q.put(None)
        audio_q.put(None)
        p_mt.join()
        p_tts.join()
        p_play.join()

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()