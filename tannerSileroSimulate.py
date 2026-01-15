import os
import time
import uuid
import re
import multiprocessing as mp
import requests
import azure.cognitiveservices.speech as speechsdk
import torch
import json
import subprocess
import threading
import pyaudio
import numpy as np
import queue # Standard queue for threading

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

secrets_dict = json.load(open("secrets.json", "r"))

SPEECH_KEY = secrets_dict.get("speech_key") or os.getenv("SPEECHKEY")
SPEECH_REGION = secrets_dict.get("SERVICE_LOCATION") or os.getenv("REGION") or "eastus"

TRANSLATOR_KEY = secrets_dict.get("translate_key") or os.getenv("TRANSLATEKEY")
TRANSLATOR_REGION = secrets_dict.get("translate_region") or os.getenv("TRANSLATEREGION") or "global"

SOURCE_LANGUAGE = "en-US"
# TARGET_LANGUAGE = "zh-CN"
# TTS_VOICE = "zh-CN-XiaoxiaoNeural"
TARGET_LANGUAGE = "pt-BR"
TTS_VOICE = "pt-BR-FranciscaNeural"

# --- INPUT CONFIG ---
USE_MIC = False  
VIDEO_FILE = r"C:\Research\TranslateCapstone\browning.mp4" 
file_name = os.path.basename(VIDEO_FILE).split("\\")[-1].split(".")[0]
ORIGINAL_VOLUME = 0.3

# Silero Configuration
MIN_WORDS_FOR_SILERO = 4        
MAX_SEGMENT_SILENCE = 2.5       
TTS_RATE = "1.3"

# --- ADAPTIVE SPLITTING ---
if TARGET_LANGUAGE in ["ja-JP", "ko-KR", "de-DE", "zh-CN"]: #Todo - split up chinese dialects
    SPLIT_PATTERN = r'[.?!](?=\s|$)' # Conservative
else:
    SPLIT_PATTERN = r'(?:[.?!,-]|\band\b)(?=\s|$)' # Aggressive

LAG_BUFFER_WORDS = 3

# -------------------------------------------------------------------
# STREAMING HELPER
# -------------------------------------------------------------------

def pump_audio_from_mp4(push_stream, filename):
    print(f"[STREAM] extracting audio from {filename}...")
    p = pyaudio.PyAudio()
    player = p.open(format=pyaudio.paInt16, channels=1, rate=16000, output=True)

    command = [
        'ffmpeg', '-i', filename, '-f', 's16le', '-ac', '1', '-ar', '16000', '-vn', '-'
    ]

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    BYTES_PER_SEC = 32000
    CHUNK_SIZE = 3200  
    SLEEP_DURATION = CHUNK_SIZE / BYTES_PER_SEC

    try:
        while True:
            chunk = process.stdout.read(CHUNK_SIZE)
            if not chunk: break
            push_stream.write(chunk)
            
            audio_data = np.frombuffer(chunk, dtype=np.int16)
            quieter_audio = (audio_data * ORIGINAL_VOLUME).astype(np.int16)
            player.write(quieter_audio.tobytes())
            
            time.sleep(SLEEP_DURATION) 
    except Exception as e:
        print(f"[STREAM ERROR] {e}")
    finally:
        push_stream.close()
        process.terminate()
        player.stop_stream()
        player.close()
        p.terminate()
        print("[STREAM] Audio finished.")

# -------------------------------------------------------------------
# MODEL LOADER
# -------------------------------------------------------------------

def load_silero_model():
    print("[INIT] Loading Silero punctuation model...")
    model, example_texts, languages, punct, apply_te = torch.hub.load(
        repo_or_dir='snakers4/silero-models',
        model='silero_te',
        trust_repo=True,
        verbose=False
    )
    return apply_te

# -------------------------------------------------------------------
# WORKERS
# -------------------------------------------------------------------
mt_session = requests.Session()

def translate_text(text: str, to_lang: str = TARGET_LANGUAGE) -> str:
    text = text.strip()
    if not text: return ""
    endpoint = "https://api.cognitive.microsofttranslator.com/translate"
    params = {"api-version": "3.0", "to": to_lang}
    headers = {
        "Ocp-Apim-Subscription-Key": TRANSLATOR_KEY,
        "Ocp-Apim-Subscription-Region": TRANSLATOR_REGION,
        "Content-Type": "application/json"
    }
    try:
        resp = mt_session.post(endpoint, params=params, headers=headers, json=[{"text": text}], timeout=5)
        resp.raise_for_status()
        return resp.json()[0]["translations"][0]["text"]
    except Exception as e:
        print(f"[MT ERROR] {e}")
        return ""

def create_tts_synthesizer_no_speaker():
    speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    speech_config.speech_synthesis_voice_name = TTS_VOICE
    # CRITICAL: We set audio_config to None so we get bytes back instead of auto-playing
    return speechsdk.SpeechSynthesizer(speech_config, audio_config=None)

def mt_worker(mt_in_queue, tts_out_queue):
    print("[MT WORKER] Ready")
    while True:
        item = mt_in_queue.get()
        if item is None: break
        translated = translate_text(item["text"])
        if translated:
            tts_out_queue.put({**item, "translated_text": translated})

# -------------------------------------------------------------------
# PLAYBACK THREAD LOGIC
# -------------------------------------------------------------------
def audio_player_thread(playback_queue):
    """
    Consumes audio bytes from the queue and plays them sequentially.
    Tracks the 'Gap' between the end of one segment and start of the next.
    """
    p = pyaudio.PyAudio()
    # Azure usually returns 16kHz 16-bit mono for default TTS (RIFF header included)
    # The header is small, PyAudio usually handles the stream okay, 
    # but strictly we should skip 44 bytes of WAV header. 
    # For simplicity, we just play the raw stream; PyAudio tolerates the header noise (it's a tiny click).
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, output=True)
    
    last_audio_end_perf = time.perf_counter()

    while True:
        item = playback_queue.get()
        if item is None: break # Poison pill

        audio_data = item['audio_data']
        meta = item['meta']

        # --- TIMING ---
        playback_start_time = time.perf_counter()
        
        # 1. Total Latency (To Ear)
        t0_recognition = meta.get("first_recognition_time", playback_start_time)
        total_latency = (playback_start_time - t0_recognition) * 1000

        # 2. Accumulation (User Speaking)
        t1_flush = meta.get("asr_flush_time", playback_start_time)
        accumulation_time = (t1_flush - t0_recognition) * 1000

        # 3. Processing (System)
        # Note: This now includes time spent sitting in the playback buffer!
        processing_time = (playback_start_time - t1_flush) * 1000

        # 4. Gap
        gap_ms = (playback_start_time - last_audio_end_perf) * 1000
        
        # Log
        print(f"[{meta['chunk_id']}] ⏱️ Total: {total_latency:.0f}ms (Gap: {gap_ms:.0f}ms)")
        with open(f"latency_log_{file_name}_{TARGET_LANGUAGE}.csv", "a") as log_file:
            log_file.write(f"{meta['chunk_id']},{playback_start_time},{accumulation_time:.0f},{processing_time:.0f},{total_latency:.0f},{gap_ms:.0f}\n")

        # PLAY
        # Skip WAV header (first 44 bytes) to avoid the "click"
        if len(audio_data) > 44:
            stream.write(audio_data[44:])
        
        last_audio_end_perf = time.perf_counter()

    stream.stop_stream()
    stream.close()
    p.terminate()

def tts_worker(tts_in_queue):
    print("[TTS WORKER] Ready (Prefetch Mode)")
    synthesizer = create_tts_synthesizer_no_speaker()
    
    # Internal queue for the playback thread
    playback_queue = queue.Queue()
    
    # Start Playback Thread
    t = threading.Thread(target=audio_player_thread, args=(playback_queue,), daemon=True)
    t.start()

    while True:
        item = tts_in_queue.get()
        if item is None: 
            playback_queue.put(None) # Kill player
            break

        ssml = f"""<speak version="1.0" xml:lang="{TARGET_LANGUAGE}">
                <voice name="{TTS_VOICE}">
                    <prosody rate="{TTS_RATE}">{item['translated_text']}</prosody>
                </voice></speak>"""
        try:
            # This call downloads the audio but does NOT play it.
            # It blocks only for the duration of the download (network speed).
            result = synthesizer.speak_ssml_async(ssml).get()
            
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                # Push raw bytes + metadata to the player immediately
                playback_queue.put({
                    "audio_data": result.audio_data,
                    "meta": item
                })
            elif result.reason == speechsdk.ResultReason.Canceled:
                cancellation_details = result.cancellation_details
                print(f"[TTS CANCELED] {cancellation_details.reason}")

        except Exception as e:
            print(f"[TTS ERROR] {e}")

# -------------------------------------------------------------------
# MAIN PROCESS STATE
# -------------------------------------------------------------------

state = {
    "word_index": 0,
    "last_flush_time": time.perf_counter(),
    "last_context_words": [],
    "first_recognition_time": None
}

def clean_text_for_silero(text):
    return text.replace(".", "").replace("?", "").replace("!", "").replace(",", "").lower().strip()

def count_words(text):
    return len(text.split())

# -------------------------------------------------------------------
# MAIN ASR LOOP
# -------------------------------------------------------------------

def main():
    # Updated CSV header
    with open(f"latency_log_{file_name}_{TARGET_LANGUAGE}.csv", "a") as log_file:
        log_file.write(f"chunk_id,playback_start_time,accumulation_time,processing_time,total_latency,gap_ms\n")

    if not SPEECH_KEY or not TRANSLATOR_KEY:
        raise RuntimeError("Missing Keys")

    apply_te = load_silero_model()

    mt_queue = mp.Queue()
    tts_queue = mp.Queue()

    mp.Process(target=mt_worker, args=(mt_queue, tts_queue), daemon=True).start()
    mp.Process(target=tts_worker, args=(tts_queue,), daemon=True).start()

    speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    speech_config.speech_recognition_language = SOURCE_LANGUAGE
    speech_config.set_property(speechsdk.PropertyId.SpeechServiceResponse_PostProcessingOption, "TrueText")
    
    if USE_MIC:
        audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    else:
        push_stream = speechsdk.audio.PushAudioInputStream()
        audio_config = speechsdk.audio.AudioConfig(stream=push_stream)
        stream_thread = threading.Thread(target=pump_audio_from_mp4, args=(push_stream, VIDEO_FILE))
        stream_thread.daemon = True
        stream_thread.start()

    recognizer = speechsdk.SpeechRecognizer(speech_config, audio_config)

    # ------------------------------------------------
    # SMART FLUSH LOGIC
    # ------------------------------------------------
    def smart_flush_check(full_raw_text, is_final=False):
        all_words = full_raw_text.split()
        total_len = len(all_words)
        start_idx = state["word_index"]

        if start_idx >= total_len: return

        new_words = all_words[start_idx:]
        
        context_words = state["last_context_words"]
        input_words = context_words + new_words
        
        input_str = " ".join(input_words)
        clean_input = clean_text_for_silero(input_str)
        
        try:
            punctuated_full = apply_te(clean_input, lan='en')
        except Exception:
            punctuated_full = input_str

        match_iter = re.finditer(SPLIT_PATTERN, punctuated_full)
        matches = list(match_iter)

        should_flush = False
        flush_text_punct = ""
        flush_num_new_words = 0
        reason = ""

        safe_punct_idx = -1
        for m in reversed(matches):
            p_idx = m.end()
            remainder = punctuated_full[p_idx:].strip()
            words_after = count_words(remainder)
            
            if not is_final:
                if words_after < LAG_BUFFER_WORDS:
                    continue 
            safe_punct_idx = p_idx
            break
        
        if safe_punct_idx != -1:
            full_sentence_punct = punctuated_full[:safe_punct_idx]
            total_punct_words = full_sentence_punct.split()
            num_words_context = len(context_words)
            num_new_committed = len(total_punct_words) - num_words_context
            
            if num_new_committed > 0:
                flush_text_punct = " ".join(total_punct_words[num_words_context:])
                flush_num_new_words = num_new_committed
                should_flush = True
                reason = "SENTENCE"

        elif is_final:
            flush_text_punct = " ".join(new_words)
            flush_num_new_words = len(new_words)
            should_flush = True
            reason = "FINAL_CATCHUP"
        
        if should_flush and flush_num_new_words > 0:
            if start_idx + flush_num_new_words > total_len:
                flush_num_new_words = total_len - start_idx

            state["word_index"] += flush_num_new_words
            state["last_flush_time"] = time.perf_counter()
            
            current_end_idx = state["word_index"]
            new_context_start = max(0, current_end_idx - 4)
            state["last_context_words"] = all_words[new_context_start:current_end_idx]

            print(f"[FLUSH:{reason}] '{flush_text_punct}'")
            
            t0 = state["first_recognition_time"] or time.perf_counter()

            mt_queue.put({
                "chunk_id": str(uuid.uuid4())[:6],
                "text": flush_text_punct,
                "reason": reason,
                "asr_flush_time": time.perf_counter(),
                "first_recognition_time": t0
            })

    def recognizing_handler(evt):
        if state["first_recognition_time"] is None:
            state["first_recognition_time"] = time.perf_counter()
        smart_flush_check(evt.result.text, is_final=False)

    def recognized_handler(evt):
        print(f"[ASR FINAL] {evt.result.text}")
        final_text = evt.result.text
        all_words = final_text.split()
        start_idx = state["word_index"]
        
        if start_idx < len(all_words):
            new_words = all_words[start_idx:]
            text_to_process = " ".join(new_words)
            
            # Simple punctuation split for final catchup
            sentences = re.split(r'([.?!])', text_to_process)
            
            chunks = []
            current_chunk = ""
            for part in sentences:
                current_chunk += part
                if part in ['.', '?', '!']:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
            if current_chunk: chunks.append(current_chunk.strip())

            t0 = state["first_recognition_time"] or time.perf_counter()
            for chunk in chunks:
                if not chunk: continue
                print(f"[FLUSH:FINAL] '{chunk}'")
                mt_queue.put({
                    "chunk_id": str(uuid.uuid4())[:6],
                    "text": chunk,
                    "reason": "FINAL_CATCHUP",
                    "asr_flush_time": time.perf_counter(),
                    "first_recognition_time": t0 
                })

        state["word_index"] = 0
        state["last_context_words"] = []
        state["first_recognition_time"] = None

    recognizer.recognizing.connect(recognizing_handler)
    recognizer.recognized.connect(recognized_handler)

    print(f"Pipeline Started. Source: {'Microphone' if USE_MIC else VIDEO_FILE}")
    recognizer.start_continuous_recognition_async().get()

    try:
        while True: time.sleep(0.5)
    except KeyboardInterrupt:
        recognizer.stop_continuous_recognition_async().get()
        mt_queue.put(None)
        tts_queue.put(None)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()