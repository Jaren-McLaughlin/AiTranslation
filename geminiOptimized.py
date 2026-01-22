import os
import time
import uuid
import re
import multiprocessing as mp
import requests
import azure.cognitiveservices.speech as speechsdk
import torch
import json
import pyaudio
import sys
import queue

# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------

try:
    secrets = json.load(open("secrets.json", "r"))
except FileNotFoundError:
    print("Error: secrets.json not found.")
    sys.exit(1)

SPEECH_KEY = secrets.get("speech_key") or os.getenv("SPEECHKEY")
SPEECH_REGION = secrets.get("SERVICE_LOCATION") or "eastus"
TRANSLATOR_KEY = secrets.get("translate_key") or os.getenv("TRANSLATEKEY")
TRANSLATOR_REGION = secrets.get("translate_region") or "global"

SOURCE_LANGUAGE = "en-US"
TARGET_LANGUAGE = "pt-BR"
TTS_VOICE = "pt-BR-FranciscaNeural"

USE_MIC = True
AUDIO_FILE = "locked.wav"

MIN_WORDS_FOR_SILERO = 4        
LAG_BUFFER_WORDS = 2            

# --- DYNAMIC SPEED CONFIG ---
BASE_TTS_RATE = "1.2"
FAST_TTS_RATE = "1.6"           

# Threshold: If latency > 2.5s for 3 segments in a row, speed up.
SAFE_LATENCY_MS = 2500          

# -------------------------------------------------------------------
# WORKER PROCESSES
# -------------------------------------------------------------------

def mt_worker(mt_in_queue, tts_out_queue, target_lang, key, region):
    """ Process 1: Translates text """
    print("[MT WORKER] Ready")
    session = requests.Session()
    
    endpoint = "https://api.cognitive.microsofttranslator.com/translate"
    headers = {
        "Ocp-Apim-Subscription-Key": key,
        "Ocp-Apim-Subscription-Region": region,
        "Content-Type": "application/json"
    }

    while True:
        item = mt_in_queue.get()
        if item is None: break

        text = item["text"].strip()
        if not text: continue

        try:
            resp = session.post(
                endpoint, 
                params={"api-version": "3.0", "to": target_lang}, 
                headers=headers, 
                json=[{"text": text}], 
                timeout=5
            )
            resp.raise_for_status()
            translated_text = resp.json()[0]["translations"][0]["text"]
            
            item["translated_text"] = translated_text
            tts_out_queue.put(item)
            
        except Exception as e:
            print(f"[MT ERROR] {e}")
            item["translated_text"] = text
            tts_out_queue.put(item)

def tts_worker(tts_in_queue, audio_out_queue, feedback_in_queue, speech_key, speech_region, voice, target_lang):
    """
    Process 2: Generates Audio.
    UPDATED LOGIC: Speeds up if 3 consecutive segments are > SAFE_LATENCY_MS.
    """
    print("[TTS WORKER] Ready")
    
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
    speech_config.speech_synthesis_voice_name = voice
    synth = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)

    # --- SPEED CONTROL STATE ---
    current_rate = BASE_TTS_RATE
    
    consecutive_over_threshold = 0
    consecutive_under_threshold = 0
    
    last_latency_val = 0

    while True:
        # 1. Process Feedback (Non-blocking drain)
        while True:
            try:
                lat_data = feedback_in_queue.get_nowait()
                last_latency_val = lat_data["total_latency"]
                
                if last_latency_val > SAFE_LATENCY_MS:
                    consecutive_over_threshold += 1
                    consecutive_under_threshold = 0 # Reset the "good" counter
                else:
                    consecutive_under_threshold += 1
                    consecutive_over_threshold = 0 # Reset the "bad" counter

            except queue.Empty:
                break
        
        # 2. Decision Time
        # Trigger: 3 consecutive segments ABOVE threshold
        if consecutive_over_threshold >= 3:
            if current_rate != FAST_TTS_RATE:
                print(f"   ⚡ Speeding up! (Last 3 segments > {SAFE_LATENCY_MS}ms)")
            current_rate = FAST_TTS_RATE
            
        # Trigger: 3 consecutive segments BELOW threshold
        elif consecutive_under_threshold >= 3:
            if current_rate != BASE_TTS_RATE:
                print(f"   🐢 Slowing down. (Last 3 segments < {SAFE_LATENCY_MS}ms)")
            current_rate = BASE_TTS_RATE

        # 3. Get Next Item to Speak
        item = tts_in_queue.get()
        if item is None: break

        text = item["translated_text"]
        
        ssml = f"""
        <speak version="1.0" xml:lang="{target_lang}">
            <voice name="{voice}">
                <prosody rate="{current_rate}">{text}</prosody>
            </voice>
        </speak>
        """

        result = synth.speak_ssml_async(ssml).get()

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            item["audio_data"] = result.audio_data
            audio_out_queue.put(item)
        else:
            print(f"[TTS ERROR] {result.cancellation_details.error_details}")

def playback_worker(audio_in_queue, feedback_out_queue):
    """ Process 3: Plays audio and reports latency back to TTS. """
    print("[PLAYBACK WORKER] Ready")
    p = pyaudio.PyAudio()
    
    try:
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, output=True)
    except Exception as e:
        print(f"[PLAYBACK INIT ERROR] {e}")
        return

    while True:
        item = audio_in_queue.get()
        if item is None: break

        data = item["audio_data"]
        
        # Calculate Latency BEFORE playing
        t_play_start = time.perf_counter()
        total_latency_ms = (t_play_start - item["first_recognition_time"]) * 1000
        
        # Send feedback to TTS worker
        feedback_out_queue.put({"total_latency": total_latency_ms})

        print(f"   📣 Playing ({total_latency_ms:.0f}ms lag)")
        
        stream.write(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

# -------------------------------------------------------------------
# MAIN THREAD
# -------------------------------------------------------------------

def load_silero_model():
    print("[INIT] Loading Silero punctuation model...")
    model, _, _, _, apply_te = torch.hub.load(
        repo_or_dir='snakers4/silero-models',
        model='silero_te',
        trust_repo=True,
        verbose=False
    )
    return apply_te

def clean_text_for_silero(text):
    return text.replace(".", "").replace("?", "").replace("!", "").replace(",", "").lower().strip()

def count_words(text):
    return len(text.split())

def main():
    # 1. Queues
    mt_queue = mp.Queue()
    tts_queue = mp.Queue()
    audio_queue = mp.Queue()
    feedback_queue = mp.Queue() 

    # 2. Workers
    p_mt = mp.Process(target=mt_worker, args=(mt_queue, tts_queue, TARGET_LANGUAGE, TRANSLATOR_KEY, TRANSLATOR_REGION))
    p_tts = mp.Process(target=tts_worker, args=(tts_queue, audio_queue, feedback_queue, SPEECH_KEY, SPEECH_REGION, TTS_VOICE, TARGET_LANGUAGE))
    p_play = mp.Process(target=playback_worker, args=(audio_queue, feedback_queue))

    p_mt.start()
    p_tts.start()
    p_play.start()

    # 3. Load Silero
    apply_te = load_silero_model()

    # 4. State Management
    state = {
        "word_index": 0,
        "last_context_words": [],  
        "first_recognition_time": None
    }

    # 5. Smart Flush Logic
    def smart_flush_check(full_raw_text, is_final=False):
        if state["first_recognition_time"] is None:
            state["first_recognition_time"] = time.perf_counter()

        all_words = full_raw_text.split()
        total_len = len(all_words)
        start_idx = state["word_index"]

        if start_idx >= total_len: return

        new_words = all_words[start_idx:]
        
        context_words = state["last_context_words"]
        input_words = context_words + new_words
        input_str = " ".join(input_words)
        
        if len(new_words) < MIN_WORDS_FOR_SILERO and not is_final:
            return 
            
        clean_input = clean_text_for_silero(input_str)
        try:
            punctuated_full = apply_te(clean_input, lan='en')
        except Exception:
            punctuated_full = input_str

        if TARGET_LANGUAGE in ["ja-JP", "ko-KR", "zh-CN"]:
            punc_regex = r'[.?!](?=\s|$)' 
        else:
            punc_regex = r'[,.?!](?=\s|$)' 

        match_iter = re.finditer(punc_regex, punctuated_full)
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
            
            current_end_idx = state["word_index"]
            new_context_start = max(0, current_end_idx - 4)
            state["last_context_words"] = all_words[new_context_start:current_end_idx]

            print(f"[FLUSH:{reason}] '{flush_text_punct}'")
            
            mt_queue.put({
                "chunk_id": str(uuid.uuid4())[:6],
                "text": flush_text_punct,
                "first_recognition_time": state["first_recognition_time"]
            })

    # 6. Azure ASR
    speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    speech_config.speech_recognition_language = SOURCE_LANGUAGE
    speech_config.set_property(speechsdk.PropertyId.SpeechServiceResponse_StablePartialResultThreshold, "2")
    
    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True) if USE_MIC else speechsdk.audio.AudioConfig(filename=AUDIO_FILE)
    recognizer = speechsdk.SpeechRecognizer(speech_config, audio_config)

    recognizer.recognizing.connect(lambda e: smart_flush_check(e.result.text, is_final=False))
    recognizer.recognized.connect(lambda e: [smart_flush_check(e.result.text, is_final=True), state.update({"word_index":0, "last_context_words":[], "first_recognition_time":None})])

    print(f"🚀 System Online ({TARGET_LANGUAGE})")
    print(f"   Feedback Loop: Enabled (Switch if 3 consecutive segments > {SAFE_LATENCY_MS}ms)")
    
    recognizer.start_continuous_recognition_async().get()

    try:
        while True: time.sleep(0.5)
    except KeyboardInterrupt:
        print("Stopping...")
        recognizer.stop_continuous_recognition_async().get()
        p_mt.terminate()
        p_tts.terminate()
        p_play.terminate()
        p_mt.join()
        p_tts.join()
        p_play.join()

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()