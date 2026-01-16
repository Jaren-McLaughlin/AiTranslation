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
import csv
from pathlib import Path
import threading

# -------------------------------------------------------------------
# GLOBAL CONFIG
# -------------------------------------------------------------------

secrets = json.load(open("secrets.json", "r"))
SPEECH_KEY = secrets["speech_key"]
SPEECH_REGION = secrets.get("SERVICE_LOCATION", "eastus")
TRANSLATOR_KEY = secrets["translate_key"]
TRANSLATOR_REGION = secrets.get("translate_region", "global")
SOURCE_LANGUAGE = "en-US"
AUDIO_FILE = r"C:\Research\TranslateCapstone\browning.mp4"

# --- TEST PARAMETERS ---
TEST_LANGUAGES = [
    {"lang": "zh-CN", "voice": "zh-CN-XiaoxiaoNeural"},
    {"lang": "pt-BR", "voice": "pt-BR-FranciscaNeural"},
    {"lang": "es-ES", "voice": "es-ES-ElviraNeural"},
    {"lang": "fr-FR", "voice": "fr-FR-DeniseNeural"},
    {"lang": "ja-JP", "voice": "ja-JP-NanamiNeural"},
    {"lang": "ko-KR", "voice": "ko-KR-InJoonNeural"},
    # Add more as needed
]

# We will test these speeds. 
# We want to find the "Break Point" where latency starts accumulating.
TEST_SPEEDS = ["1.2", "1.3", "1.4", "1.5", "1.6", "1.7", "1.8"] 

SEGMENTS_PER_TEST = 50  # Number of sentences to process before stopping a test run
MAX_TEST_DURATION_SEC = 420 # Safety timeout per test

# -------------------------------------------------------------------
# SHARED RESOURCES (Silero)
# -------------------------------------------------------------------
# Load once globally to save time
print("[INIT] Loading Silero...")
silero_model, _, _, _, apply_te = torch.hub.load(
    repo_or_dir="snakers4/silero-models",
    model="silero_te",
    trust_repo=True,
    verbose=False
)

# -------------------------------------------------------------------
# WORKER FUNCTIONS
# -------------------------------------------------------------------

def pump_audio_from_file(push_stream, filename, stop_event):
    # Sends audio to Azure
    command = ['ffmpeg', '-i', filename, '-f', 's16le', '-ac', '1', '-ar', '16000', '-vn', '-']
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    CHUNK_SIZE = 3200 
    # Speed up input slightly (0.85x sleep) to stress test the system
    SLEEP_DURATION = (CHUNK_SIZE / 32000) * 0.90 

    try:
        while not stop_event.is_set():
            chunk = process.stdout.read(CHUNK_SIZE)
            if not chunk: break
            push_stream.write(chunk)
            time.sleep(SLEEP_DURATION) 
    except Exception:
        pass
    finally:
        push_stream.close()
        process.terminate()

def mt_worker(mt_q, tts_q, target_lang):
    session = requests.Session()
    while True:
        item = mt_q.get()
        if item is None: break
        
        text = item["text"]
        if not text.strip():
            tts_q.put(item)
            continue

        try:
            r = session.post(
                "https://api.cognitive.microsofttranslator.com/translate",
                params={"api-version": "3.0", "to": target_lang},
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
        except Exception:
            item["translated"] = text 
        
        tts_q.put(item)

def tts_worker(tts_q, audio_q, tts_voice, tts_rate):
    speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    speech_config.speech_synthesis_voice_name = tts_voice
    synth = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)

    while True:
        item = tts_q.get()
        if item is None: break

        txt = item["translated"]
        ssml = f"""
        <speak version="1.0" xml:lang="{item['lang']}">
            <voice name="{tts_voice}">
            <prosody rate="{tts_rate}">{txt}</prosody>
            </voice>
        </speak>
        """
        
        result = synth.speak_ssml_async(ssml).get()
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            raw_bytes = result.audio_data[44:] if len(result.audio_data) > 44 else result.audio_data
            item["audio"] = raw_bytes
            item["t_tts_ready"] = time.perf_counter()
            audio_q.put(item)

def playback_worker(audio_q, results_q):
    p = pyaudio.PyAudio()
    try:
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, output=True)
    except:
        return

    while True:
        item = audio_q.get()
        if item is None: break

        t_play_start = time.perf_counter()
        stream.write(item["audio"])
        
        # Calculate final metrics
        total_latency = (t_play_start - item["t_asr_start"]) * 1000
        
        # Send metrics back to main process for logging
        results_q.put({
            "lang": item["lang"],
            "rate": item["rate"],
            "id": item["id"],
            "total_latency": total_latency,
            "asr_latency": (item["t_asr_flush"] - item["t_asr_start"]) * 1000,
            "playback_latency": (t_play_start - item["t_tts_ready"]) * 1000
        })

    stream.stop_stream()
    stream.close()
    p.terminate()

# -------------------------------------------------------------------
# SINGLE TEST RUNNER
# -------------------------------------------------------------------

def run_test_session(lang_config, speed, results_writer):
    target_lang = lang_config["lang"]
    voice = lang_config["voice"]
    
    print(f"\n--- STARTING TEST: {target_lang} @ {speed}x ---")
    
    # Queues
    mt_q = mp.Queue()
    tts_q = mp.Queue() 
    audio_q = mp.Queue()
    results_q = mp.Queue()
    stop_event = mp.Event() # To stop pump

    # Processes
    p_mt = mp.Process(target=mt_worker, args=(mt_q, tts_q, target_lang))
    p_tts = mp.Process(target=tts_worker, args=(tts_q, audio_q, voice, speed))
    p_play = mp.Process(target=playback_worker, args=(audio_q, results_q))

    p_mt.start()
    p_tts.start()
    p_play.start()

    # ASR Setup
    speech_cfg = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    speech_cfg.speech_recognition_language = SOURCE_LANGUAGE
    speech_cfg.set_property(speechsdk.PropertyId.SpeechServiceResponse_StablePartialResultThreshold, "1")
    
    push_stream = speechsdk.audio.PushAudioInputStream()
    audio_cfg = speechsdk.audio.AudioConfig(stream=push_stream)
    
    pump_thread = threading.Thread(target=pump_audio_from_file, args=(push_stream, AUDIO_FILE, stop_event), daemon=True)
    pump_thread.start()

    recognizer = speechsdk.SpeechRecognizer(speech_cfg, audio_cfg)

    # Local State
    state = {"words": [], "idx": 0, "t0": None, "last_clean": "", "last_punct": ""}
    processed_count = 0

    def clean(t): return re.sub(r"[^\w\s]", "", t).lower()

    def flush(text, reason):
        mt_q.put({
            "id": str(uuid.uuid4())[:6],
            "text": text,
            "t_asr_start": state["t0"],
            "t_asr_flush": time.perf_counter(),
            "reason": reason,
            "lang": target_lang,
            "rate": speed
        })

    def process(text, final=False):
        if state["t0"] is None: state["t0"] = time.perf_counter()
        words = text.split()
        if len(words) <= state["idx"]: return
        
        new = words[state["idx"]:]
        combined = " ".join(new)
        
        # Silero Logic
        punct = combined
        if len(new) >= 6: # MIN_WORDS
            c = clean(combined)
            if c != state["last_clean"]:
                punct = apply_te(c, lan="en")
                state["last_clean"] = c
                state["last_punct"] = punct
        
        # Punctuation Check
        punc_regex = r"[,.?!]" 
        for m in re.finditer(punc_regex, punct):
            tail = punct[m.end():].split()
            if not final and len(tail) < 1: continue # LAG_BUFFER
            
            segment = punct[:m.end()].strip()
            state["idx"] += len(segment.split())
            flush(segment, "EARLY")
            state["t0"] = time.perf_counter()
            return

        if final:
            flush(combined, "FINAL")
            state["idx"] = 0
            state["t0"] = None

    recognizer.recognizing.connect(lambda e: process(e.result.text))
    recognizer.recognized.connect(lambda e: process(e.result.text, True))

    recognizer.start_continuous_recognition_async().get()

    # Monitoring Loop
    start_time = time.time()
    try:
        while processed_count < SEGMENTS_PER_TEST:
            if time.time() - start_time > MAX_TEST_DURATION_SEC:
                print("⚠️ Timeout reached.")
                break
            
            # Check for results
            while not results_q.empty():
                res = results_q.get()
                processed_count += 1
                print(f"[{processed_count}/{SEGMENTS_PER_TEST}] Latency: {res['total_latency']:.0f}ms")
                
                # Write to CSV
                results_writer.writerow([
                    res["lang"], res["rate"], res["id"], 
                    f"{res['asr_latency']:.0f}", 
                    f"{res['playback_latency']:.0f}", 
                    f"{res['total_latency']:.0f}"
                ])
            time.sleep(0.1)
            
    finally:
        # Cleanup
        stop_event.set()
        recognizer.stop_continuous_recognition_async().get()
        mt_q.put(None)
        tts_q.put(None)
        audio_q.put(None)
        
        p_mt.join()
        p_tts.join()
        p_play.join()
        pump_thread.join(timeout=1)

# -------------------------------------------------------------------
# MAIN LOOP
# -------------------------------------------------------------------

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    
    # Init CSV
    csv_file = open("benchmark_results.csv", "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow(["Language", "Speed", "ID", "ASR_Latency", "Playback_Queue_Latency", "Total_Latency"])
    
    try:
        for lang_conf in TEST_LANGUAGES:
            for speed in TEST_SPEEDS:
                run_test_session(lang_conf, speed, writer)
                # Cool down between tests
                time.sleep(2) 
    except KeyboardInterrupt:
        print("\nBenchmark Aborted.")
    finally:
        csv_file.close()
        print("\n✅ Benchmark Complete. Results saved to benchmark_results.csv")