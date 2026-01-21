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
import queue

# -------------------------------------------------------------------
# GLOBAL CONFIG
# -------------------------------------------------------------------

try:
    secrets = json.load(open("secrets.json", "r"))
except FileNotFoundError:
    print("⚠️ 'secrets.json' not found. Please create it.")
    exit(1)

SPEECH_KEY = secrets["speech_key"]
SPEECH_REGION = secrets.get("SERVICE_LOCATION", "eastus")
TRANSLATOR_KEY = secrets["translate_key"]
TRANSLATOR_REGION = secrets.get("translate_region", "global")
SOURCE_LANGUAGE = "en-US"
AUDIO_FILE = r"C:\Research\TranslateCapstone\christofferson.mp4"

# --- TEST PARAMETERS ---
TEST_LANGUAGES = [
    # {"lang": "pt-BR", "voice": "pt-BR-FranciscaNeural"},
    # {"lang": "es-ES", "voice": "es-ES-ElviraNeural"},
    # {"lang": "fr-FR", "voice": "fr-FR-DeniseNeural"},
    {"lang": "ja-JP", "voice": "ja-JP-NanamiNeural"},
    {"lang": "ko-KR", "voice": "ko-KR-InJoonNeural"},
    {"lang": "zh-CN", "voice": "zh-CN-XiaoxiaoNeural"},
]

# Optimized speeds for efficient testing
TEST_SPEEDS = ["1.3", "1.5", "1.7"] 

SEGMENTS_PER_TEST = 25  
MAX_TEST_DURATION_SEC = 420 

# -------------------------------------------------------------------
# SHARED RESOURCES (Silero)
# -------------------------------------------------------------------
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
    command = ['ffmpeg', '-i', filename, '-f', 's16le', '-ac', '1', '-ar', '16000', '-vn', '-']
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    CHUNK_SIZE = 3200 
    # Use 1.0x Real Time speed for accurate latency testing
    SLEEP_DURATION = (CHUNK_SIZE / 32000) 

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
    
    # ---------------------------------------------------------
    # NEW: State for Deduplication
    # ---------------------------------------------------------
    last_processed_text = ""
    
    while True:
        item = mt_q.get()
        if item is None: break
        
        text = item["text"]
        if not text.strip():
            tts_q.put(item)
            continue

        # ---------------------------------------------------------
        # NEW: Deduplication Logic
        # ---------------------------------------------------------
        clean_new = text.strip().lower()
        clean_last = last_processed_text.strip().lower()
        
        # 1. Exact Match (Prevents double looping)
        if clean_new == clean_last:
            print(f"   [MT SKIP] Exact duplicate: '{text}'")
            continue
            
        # 2. Overlap Match (Prevents "555" vs "five five five" overlap)
        # If the new text is fully contained at the end of the last text, skip it.
        if len(clean_new) > 5 and clean_new in clean_last:
             print(f"   [MT SKIP] Overlap duplicate: '{text}'")
             continue

        last_processed_text = text
        # ---------------------------------------------------------

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
    stream = None

    def _open_stream():
        """Helper to safely open/re-open the stream"""
        try:
            return p.open(format=pyaudio.paInt16, channels=1, rate=16000, output=True)
        except Exception as e:
            print(f"[PLAYBACK INIT ERROR] {e}")
            return None

    # Initial open
    stream = _open_stream()

    while True:
        item = audio_q.get()
        if item is None: break

        audio_data = item.get("audio")
        if not audio_data or len(audio_data) == 0:
            continue

        t_play_start = time.perf_counter()
        
        # --- ROBUST WRITE LOOP ---
        # If writing fails (Error -9999), we close the stream, re-open it, and try again.
        success = False
        retries = 0
        while not success and retries < 3:
            if stream is None:
                stream = _open_stream()
                if stream is None:
                    time.sleep(0.1)
                    retries += 1
                    continue

            try:
                stream.write(audio_data)
                success = True
            except OSError as e:
                # Catch the -9999 error here
                print(f"⚠️ [PLAYBACK DRIVER ERROR] {e}. Resetting Audio Stream...")
                try:
                    stream.close()
                except:
                    pass
                stream = None # Force re-open on next pass
                retries += 1
            except Exception as e:
                print(f"❌ [PLAYBACK UNKNOWN ERROR] {e}")
                break
        
        if not success:
            print(f"❌ [SKIP] Dropped audio segment {item['id']} due to driver failure.")
            continue
        # -------------------------

        # Calculate final metrics
        total_latency = (t_play_start - item["t_asr_start"]) * 1000
        
        results_q.put({
            "lang": item["lang"],
            "rate": item["rate"],
            "id": item["id"],
            "total_latency": total_latency,
            "asr_latency": (item["t_asr_flush"] - item["t_asr_start"]) * 1000,
            "playback_latency": (t_play_start - item["t_tts_ready"]) * 1000
        })

    if stream:
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
    
    mt_q = mp.Queue()
    tts_q = mp.Queue() 
    audio_q = mp.Queue()
    results_q = mp.Queue()
    stop_event = mp.Event()
    
    # NEW: Gate for start signal
    session_started_event = threading.Event()

    p_mt = mp.Process(target=mt_worker, args=(mt_q, tts_q, target_lang))
    p_tts = mp.Process(target=tts_worker, args=(tts_q, audio_q, voice, speed))
    p_play = mp.Process(target=playback_worker, args=(audio_q, results_q))

    p_mt.start()
    p_tts.start()
    p_play.start()

    asr_text_queue = queue.Queue()

    speech_cfg = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    speech_cfg.speech_recognition_language = SOURCE_LANGUAGE
    speech_cfg.set_property(speechsdk.PropertyId.SpeechServiceResponse_StablePartialResultThreshold, "3")
    
    push_stream = speechsdk.audio.PushAudioInputStream()
    audio_cfg = speechsdk.audio.AudioConfig(stream=push_stream)
    
    # ---------------------------------------------------------
    # NEW: Gated Pump to prevent "Buffer Bloat" / Fake Latency
    # ---------------------------------------------------------
    def gated_pump(stream, file, stop, start_signal):
        print("[PUMP] Waiting for Azure Session...")
        if not start_signal.wait(timeout=10):
            print("[PUMP] ⚠️ Timed out waiting for session start!")
            return
        print("[PUMP] Session Ready! Starting Stream (1.0x Real Time).")
        pump_audio_from_file(stream, file, stop)

    pump_thread = threading.Thread(
        target=gated_pump, 
        args=(push_stream, AUDIO_FILE, stop_event, session_started_event), 
        daemon=True
    )
    pump_thread.start()

    recognizer = speechsdk.SpeechRecognizer(speech_cfg, audio_cfg)

    def text_processing_worker():
        state = {"words": [], "idx": 0, "t0": None, "last_clean": "", "last_punct": ""}
        
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

        while not stop_event.is_set():
            try:
                item = asr_text_queue.get(timeout=0.1) 
            except queue.Empty:
                continue
            if item is None: break 

            text, is_final = item
            
            if state["t0"] is None: state["t0"] = time.perf_counter()
            words = text.split()
            if len(words) <= state["idx"]: 
                continue
            
            new = words[state["idx"]:]
            combined = " ".join(new)
            
            punct = combined
            if len(new) >= 4: 
                c = clean(combined)
                if c != state["last_clean"]:
                    try:
                        punct = apply_te(c, lan="en")
                        state["last_clean"] = c
                        state["last_punct"] = punct
                    except Exception:
                        punct = combined
            
            punc_regex = r"[,.?!]" if target_lang not in ["ja-JP", "ko-KR", "de-DE"] else r"[.?!]"
            for m in re.finditer(punc_regex, punct):
                tail = punct[m.end():].split()
                if not is_final and len(tail) < 1: continue 
                
                segment = punct[:m.end()].strip()
                state["idx"] += len(segment.split())
                flush(segment, "EARLY")
                state["t0"] = time.perf_counter()
                
                # IMPORTANT: Break so we re-evaluate from the new index
                break 

            if is_final:
                flush(combined, "FINAL")
                state["idx"] = 0
                state["t0"] = None

    t_processor = threading.Thread(target=text_processing_worker, daemon=True)
    t_processor.start()

    recognizer.recognizing.connect(lambda e: asr_text_queue.put((e.result.text, False)))
    recognizer.recognized.connect(lambda e: asr_text_queue.put((e.result.text, True)))
    
    # ---------------------------------------------------------
    # NEW: Signal pump to start ONLY when Azure is ready
    # ---------------------------------------------------------
    recognizer.session_started.connect(lambda e: session_started_event.set())
    recognizer.canceled.connect(lambda e: print(f"CANCELED: {e}"))

    recognizer.start_continuous_recognition_async().get()

    start_time = time.time()
    processed_count = 0
    try:
        while processed_count < SEGMENTS_PER_TEST:
            if time.time() - start_time > MAX_TEST_DURATION_SEC:
                print("⚠️ Timeout reached.")
                break
            
            while not results_q.empty():
                res = results_q.get()
                processed_count += 1
                print(f"[{processed_count}/{SEGMENTS_PER_TEST}] Latency: {res['total_latency']:.0f}ms")
                results_writer.writerow([
                    res["lang"], res["rate"], res["id"], 
                    f"{res['asr_latency']:.0f}", 
                    f"{res['playback_latency']:.0f}", 
                    f"{res['total_latency']:.0f}"
                ])
            time.sleep(0.1)
            
    finally:
        print(f"--- Stopping Test: {target_lang} @ {speed}x ---")
        stop_event.set() 
        try:
            recognizer.stop_continuous_recognition_async().get()
        except:
            pass
        p_mt.terminate()
        p_tts.terminate()
        p_play.terminate()
        try:
            push_stream.close()
        except:
            pass
        p_mt.join()
        p_tts.join()
        p_play.join()
        asr_text_queue.put(None)
        t_processor.join(timeout=1.0)
        pump_thread.join(timeout=1.0)
        print("--- Cleanup Complete ---")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    
    csv_file = open("benchmark_results.csv", "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow(["Language", "Speed", "ID", "ASR_Latency", "Playback_Queue_Latency", "Total_Latency"])
    
    try:
        for lang_conf in TEST_LANGUAGES:
            for speed in TEST_SPEEDS:
                run_test_session(lang_conf, speed, writer)
                time.sleep(2) 
    except KeyboardInterrupt:
        print("\nBenchmark Aborted.")
    finally:
        csv_file.close()
        print("\n✅ Benchmark Complete. Results saved to benchmark_results.csv")