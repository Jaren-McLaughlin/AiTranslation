import os
import time
import json
import subprocess
import threading
import azure.cognitiveservices.speech as speechsdk
import multiprocessing as mp

# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------

try:
    secrets = json.load(open("secrets.json", "r"))
except FileNotFoundError:
    print("⚠️ 'secrets.json' not found. Please create it.")
    exit(1)

SPEECH_KEY = secrets["speech_key"]
SPEECH_REGION = secrets.get("SERVICE_LOCATION", "eastus")
SOURCE_LANGUAGE = "en-US" 

# The file you want to verify
AUDIO_FILE = r"C:\Research\TranslateCapstone\browning.mp4"

# -------------------------------------------------------------------
# AUDIO PUMP (Exactly matching your live architecture)
# -------------------------------------------------------------------

def pump_audio_from_file(push_stream, filename, stop_event):
    print(f"[STREAM] Reading from {filename}...")
    command = ['ffmpeg', '-i', filename, '-f', 's16le', '-ac', '1', '-ar', '16000', '-vn', '-']
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    
    CHUNK_SIZE = 3200 
    # 1.0x Real Time Speed (Crucial for accurate accuracy testing)
    SLEEP_DURATION = (CHUNK_SIZE / 32000) 

    try:
        while not stop_event.is_set():
            chunk = process.stdout.read(CHUNK_SIZE)
            if not chunk: break
            push_stream.write(chunk)
            time.sleep(SLEEP_DURATION) 
    except Exception as e:
        print(f"[STREAM ERROR] {e}")
    finally:
        push_stream.close()
        process.terminate()
        print("[STREAM] Audio finished.")
        stop_event.set() # Signal main thread that audio is done

# -------------------------------------------------------------------
# MAIN VERIFICATION LOOP
# -------------------------------------------------------------------

def main():
    # 1. Setup Threading Events
    stop_event = threading.Event()
    session_started_event = threading.Event()
    
    # 2. Setup Azure ASR
    speech_cfg = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    speech_cfg.speech_recognition_language = SOURCE_LANGUAGE
    
    # Use the exact same properties as your production script
    speech_cfg.set_property(speechsdk.PropertyId.SpeechServiceResponse_StablePartialResultThreshold, "3")
    speech_cfg.set_property(speechsdk.PropertyId.SpeechServiceResponse_PostProcessingOption, "TrueText")
    
    push_stream = speechsdk.audio.PushAudioInputStream()
    audio_cfg = speechsdk.audio.AudioConfig(stream=push_stream)
    
    recognizer = speechsdk.SpeechRecognizer(speech_cfg, audio_cfg)

    # 3. Capture Results
    full_transcript = []

    def handle_final_result(evt):
        text = evt.result.text
        if text:
            print(f"[ASR] {text}")
            full_transcript.append(text)

    # Connect Callbacks
    recognizer.recognized.connect(handle_final_result)
    recognizer.session_started.connect(lambda e: session_started_event.set())
    recognizer.canceled.connect(lambda e: print(f"[CANCELED] {e}"))

    # 4. Gated Audio Pump
    def gated_pump():
        print("[PUMP] Waiting for Azure Session...")
        if not session_started_event.wait(timeout=10):
            print("⚠️ Timed out waiting for session start!")
            return
        print("[PUMP] Session Ready! Starting Stream.")
        pump_audio_from_file(push_stream, AUDIO_FILE, stop_event)

    pump_thread = threading.Thread(target=gated_pump, daemon=True)
    
    # 5. Run
    print(f"--- STARTING ASR VERIFICATION: {SOURCE_LANGUAGE} ---")
    pump_thread.start()
    recognizer.start_continuous_recognition_async().get()

    # Wait for audio to finish
    try:
        while not stop_event.is_set():
            time.sleep(0.5)
        
        # Give ASR a few seconds to process the very last buffer
        print("Audio finished. Waiting for final ASR results...")
        time.sleep(3) 
        
    except KeyboardInterrupt:
        print("\nAborted by user.")
    finally:
        recognizer.stop_continuous_recognition_async().get()
        stop_event.set()

    # 6. Output Results
    final_text = "\n".join(full_transcript)
    
    print("\n" + "="*40)
    print("FINAL ASR TRANSCRIPT")
    print("="*40)
    print(final_text)
    print("="*40)

    with open("asr_output.txt", "w", encoding="utf-8") as f:
        f.write(final_text)
    
    print(f"\n✅ Saved transcript to 'asr_output.txt'")

if __name__ == "__main__":
    main()