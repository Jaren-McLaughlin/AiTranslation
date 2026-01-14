# import os
# import time
# import uuid
# import re
# import multiprocessing as mp
# import requests
# import azure.cognitiveservices.speech as speechsdk
# import torch
# import json

# # -------------------------------------------------------------------
# # CONFIG
# # -------------------------------------------------------------------

# secrets_dict = json.load(open("secrets.json", "r"))

# SPEECH_KEY = secrets_dict.get("speech_key") or os.getenv("SPEECHKEY")
# SPEECH_REGION = secrets_dict.get("SERVICE_LOCATION") or os.getenv("REGION") or "eastus"

# # TRANSLATOR_KEY = os.getenv("TRANSLATEKEY")
# TRANSLATOR_KEY = secrets_dict.get("translate_key") or os.getenv("TRANSLATEKEY")
# TRANSLATOR_REGION = secrets_dict.get("translate_region") or os.getenv("TRANSLATEREGION") or "global"

# SOURCE_LANGUAGE = "en-US"
# # TARGET_LANGUAGE = "pt-BR"
# TARGET_LANGUAGE = "de-DE"
# # TTS_VOICE = "pt-BR-FranciscaNeural"
# TTS_VOICE = "de-DE-KatjaNeural"

# USE_MIC = True
# AUDIO_FILE = "locked.wav"

# # Silero Configuration
# MIN_WORDS_FOR_SILERO = 4        # Don't bother running model on < 4 words
# MAX_SEGMENT_SILENCE = 2.5       # Flush anyway if silence > 2.5s
# TTS_RATE = "1.3"

# LAG_BUFFER_WORDS = 7

# # -------------------------------------------------------------------
# # MODEL LOADER (Main Process)
# # -------------------------------------------------------------------

# def load_silero_model():
#     print("[INIT] Loading Silero punctuation model...")
#     model, example_texts, languages, punct, apply_te = torch.hub.load(
#         repo_or_dir='snakers4/silero-models',
#         model='silero_te',
#         trust_repo=True,
#         verbose=False
#     )
#     return apply_te

# # -------------------------------------------------------------------
# # WORKERS (MT + TTS)
# # -------------------------------------------------------------------

# def translate_text(text: str, to_lang: str = TARGET_LANGUAGE) -> str:
#     # (Same translation logic as before)
#     text = text.strip()
#     if not text: return ""
    
#     endpoint = "https://api.cognitive.microsofttranslator.com/translate"
#     params = {"api-version": "3.0", "to": to_lang}
#     headers = {
#         "Ocp-Apim-Subscription-Key": TRANSLATOR_KEY,
#         "Ocp-Apim-Subscription-Region": TRANSLATOR_REGION,
#         "Content-Type": "application/json"
#     }
#     try:
#         resp = requests.post(endpoint, params=params, headers=headers, json=[{"text": text}], timeout=5)
#         resp.raise_for_status()
#         return resp.json()[0]["translations"][0]["text"]
#     except Exception as e:
#         print(f"[MT ERROR] {e}")
#         return ""

# def create_tts_synthesizer():
#     speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
#     speech_config.speech_synthesis_voice_name = TTS_VOICE
#     audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
#     return speechsdk.SpeechSynthesizer(speech_config, audio_config)

# def mt_worker(mt_in_queue, tts_out_queue):
#     print("[MT WORKER] Ready")
#     while True:
#         item = mt_in_queue.get()
#         if item is None: break

#         text = item["text"]
#         print(f"[MT] Translating: '{text}'")
#         translated = translate_text(text)
        
#         if translated:
#             tts_out_queue.put({**item, "translated_text": translated})

# # def tts_worker(tts_in_queue):
# #     print("[TTS WORKER] Ready")
# #     synthesizer = create_tts_synthesizer()
# #     while True:
# #         item = tts_in_queue.get()
# #         if item is None: break

# #         ssml = f"""<speak version="1.0" xml:lang="{TARGET_LANGUAGE}">
# #                 <voice name="{TTS_VOICE}">
# #                     <prosody rate="{TTS_RATE}">{item['translated_text']}</prosody>
# #                 </voice></speak>"""
# #         try:
# #             synthesizer.speak_ssml_async(ssml).get()
# #         except Exception as e:
# #             print(f"[TTS ERROR] {e}")
# def tts_worker(tts_in_queue):
#     print("[TTS WORKER] Ready")
#     synthesizer = create_tts_synthesizer()
    
#     # NEW: Connect to the synthesis_started event for precise playback timing
#     # This fires when the audio stream actually begins
#     def synthesis_started_cb(evt):
#         pass # We can handle logging here if we wanted event-based logging
#     synthesizer.synthesis_started.connect(synthesis_started_cb)

#     while True:
#         item = tts_in_queue.get()
#         if item is None: break

#         # CHANGE: Calculate Latency
#         asr_end_time = item.get("start_time", 0)
#         current_time = time.perf_counter()
#         latency_ms = (current_time - asr_end_time) * 1000

#         print(f"[{item['chunk_id']}] Latency (ASR Flush -> TTS Start): {latency_ms:.0f}ms")

#         ssml = f"""<speak version="1.0" xml:lang="{TARGET_LANGUAGE}">
#                 <voice name="{TTS_VOICE}">
#                     <prosody rate="{TTS_RATE}">{item['translated_text']}</prosody>
#                 </voice></speak>"""
#         try:
#             # This line triggers the playback
#             synthesizer.speak_ssml_async(ssml).get()
#         except Exception as e:
#             print(f"[TTS ERROR] {e}")

# # -------------------------------------------------------------------
# # MAIN PROCESS STATE
# # -------------------------------------------------------------------

# # state = {
# #     "stable_prefix": "",      # The RAW text from Azure we have already processed
# #     "last_flush_time": time.perf_counter(),
# # }
# state = {
#     "word_index": 0,
#     "last_flush_time": time.perf_counter(),
#     "last_context_words": [] # NEW: Buffer to store the last 3 words
# }

# def clean_text_for_silero(text):
#     # Azure TrueText sometimes adds basic punctuation; strip it so Silero has a clean slate
#     return text.replace(".", "").replace("?", "").replace("!", "").replace(",", "").lower().strip()

# def count_words(text):
#     return len(text.split())

# # -------------------------------------------------------------------
# # MAIN ASR LOOP
# # -------------------------------------------------------------------

# def main():
#     if not SPEECH_KEY or not TRANSLATOR_KEY:
#         raise RuntimeError("Missing Keys")

#     # Load Silero (blocking load, but only once at startup)
#     apply_te = load_silero_model()

#     # Queues
#     mt_queue = mp.Queue()
#     tts_queue = mp.Queue()

#     # Workers
#     mp.Process(target=mt_worker, args=(mt_queue, tts_queue), daemon=True).start()
#     mp.Process(target=tts_worker, args=(tts_queue,), daemon=True).start()

#     # ASR Config
#     speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
#     speech_config.speech_recognition_language = SOURCE_LANGUAGE
#     # Disable TrueText if possible to give Silero raw input, but keeping it on is fine too
#     speech_config.set_property(speechsdk.PropertyId.SpeechServiceResponse_PostProcessingOption, "TrueText")
    
#     audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True) if USE_MIC else speechsdk.audio.AudioConfig(filename=AUDIO_FILE)
#     recognizer = speechsdk.SpeechRecognizer(speech_config, audio_config)

#     # ------------------------------------------------
#     # SMART FLUSH LOGIC
#     # ------------------------------------------------
#     def smart_flush_check(full_raw_text, is_final=False):
#         """
#         BUFFERED TRANSLATION LOGIC:
#         1. Always keeps 'LAG_BUFFER_WORDS' (e.g., 7) words in the chamber.
#         2. Only flushes punctuation if it is 'Old Enough' (outside the lag buffer).
#         3. Catches up instantly when is_final=True.
#         """
#         all_words = full_raw_text.split()
#         total_len = len(all_words)
#         start_idx = state["word_index"]

#         # Nothing new to process
#         if start_idx >= total_len: return

#         new_words = all_words[start_idx:]
        
#         # --- 1. CONTEXT PREPENDING ---
#         context_words = state["last_context_words"]
#         input_words = context_words + new_words
        
#         input_str = " ".join(input_words)
#         clean_input = clean_text_for_silero(input_str)
        
#         try:
#             punctuated_full = apply_te(clean_input, lan='en')
#         except Exception:
#             punctuated_full = input_str

#         # --- 2. FIND STABLE CUT POINT ---
#         match_iter = re.finditer(r'[.?!](?=\s|$)', punctuated_full)
#         matches = list(match_iter)

#         should_flush = False
#         flush_text_punct = ""
#         flush_num_new_words = 0
#         reason = ""

#         # We prefer the latest possible safe punctuation
#         safe_punct_idx = -1
        
#         for m in reversed(matches):
#             p_idx = m.end()
            
#             # Check what comes AFTER this punctuation
#             remainder = punctuated_full[p_idx:].strip()
#             words_after = count_words(remainder)
            
#             # --- THE LAG CHECK ---
#             # If we are STREAMING (not final), we enforce the Lag Buffer.
#             # We reject any punctuation that is inside the "New/Volatile" zone.
#             if not is_final:
#                 if words_after < LAG_BUFFER_WORDS:
#                     continue # Too new! Skip this punctuation.
            
#             # If we pass the check (or if is_final=True), this is valid.
#             safe_punct_idx = p_idx
#             break
        
#         if safe_punct_idx != -1:
#             # We found a stable sentence boundary outside the lag zone
#             full_sentence_punct = punctuated_full[:safe_punct_idx]
#             total_punct_words = full_sentence_punct.split()
#             num_words_context = len(context_words)
#             num_new_committed = len(total_punct_words) - num_words_context
            
#             if num_new_committed > 0:
#                 flush_text_punct = " ".join(total_punct_words[num_words_context:])
#                 flush_num_new_words = num_new_committed
#                 should_flush = True
#                 reason = "SENTENCE"

#         elif is_final:
#             # --- 3. FINAL CATCH-UP ---
#             # If Azure says "Sentence Finalized", we flush the ENTIRE buffer
#             # regardless of punctuation or lag.
#             flush_text_punct = " ".join(new_words) # Fallback to raw if no punctuation found
#             flush_num_new_words = len(new_words)
#             should_flush = True
#             reason = "FINAL_CATCHUP"
        
#         # Note: We REMOVED the "Timeout" flush here. 
#         # We rely on Azure's 'recognized' event (Final) to handle the catch-up.

#         # --- 4. EXECUTE FLUSH ---
#         if should_flush and flush_num_new_words > 0:
#             if start_idx + flush_num_new_words > total_len:
#                 flush_num_new_words = total_len - start_idx

#             state["word_index"] += flush_num_new_words
#             state["last_flush_time"] = time.perf_counter()
            
#             current_end_idx = state["word_index"]
#             new_context_start = max(0, current_end_idx - 4)
#             state["last_context_words"] = all_words[new_context_start:current_end_idx]

#             print(f"[FLUSH:{reason}] '{flush_text_punct}'")
            
#             mt_queue.put({
#                 "chunk_id": str(uuid.uuid4())[:6],
#                 "text": flush_text_punct,
#                 "reason": reason,
#                 "start_time": time.perf_counter()
#             })
#     # def smart_flush_check(full_raw_text):
#     #     """
#     #     1. Prepends past context so Silero doesn't fragment sentences.
#     #     2. Uses 'Lookahead' to ensure we don't flush a period until we see the next word.
#     #     """
#     #     all_words = full_raw_text.split()
#     #     total_len = len(all_words)
#     #     start_idx = state["word_index"]

#     #     if start_idx >= total_len: return

#     #     new_words = all_words[start_idx:]
        
#     #     # --- 1. CONTEXT PREPENDING ---
#     #     # We attach the last 3 words of the PREVIOUS flush to the front.
#     #     # This helps Silero realize "to act like humans" is a continuation.
#     #     context_words = state["last_context_words"]
#     #     input_words = context_words + new_words
        
#     #     # Don't run model if we only have the context + 1 or 2 new words (unless timeout)
#     #     time_since_flush = time.perf_counter() - state["last_flush_time"]
#     #     if len(new_words) < MIN_WORDS_FOR_SILERO and time_since_flush < MAX_SEGMENT_SILENCE:
#     #         return

#     #     # Prepare string for Silero
#     #     input_str = " ".join(input_words)
#     #     clean_input = clean_text_for_silero(input_str)
        
#     #     try:
#     #         punctuated_full = apply_te(clean_input, lan='en')
#     #     except Exception:
#     #         punctuated_full = input_str

#     #     # --- 2. FINDING THE CUT POINT ---
#     #     # We need to find punctuation that exists inside the NEW section.
#     #     # But wait! 'punctuated_full' includes our context words.
#     #     # We don't want to re-flush the context words.
        
#     #     # Strategy: We look for punctuation, but we only accept it if:
#     #     # A) It is 'deep' enough in the string (covering the context).
#     #     # B) There is text AFTER the punctuation (Lookahead).
        
#     #     match_iter = re.finditer(r'[.?!](?=\s|$)', punctuated_full)
#     #     matches = list(match_iter)

#     #     should_flush = False
#     #     flush_text_punct = ""
#     #     flush_num_new_words = 0
#     #     reason = ""

#     #     if matches:
#     #         # Check the LAST detected punctuation mark
#     #         last_match = matches[-1]
#     #         punct_idx = last_match.end()
            
#     #         # LOOKAHEAD CHECK:
#     #         # Is the punctuation at the very end of the string?
#     #         is_at_end = (punct_idx >= len(punctuated_full))
            
#     #         # If it's at the end, we normally WAIT for the next word to confirm it.
#     #         # UNLESS it's been a long time (Timeout).
#     #         if is_at_end and time_since_flush < MAX_SEGMENT_SILENCE:
#     #             return # Wait for more context!

#     #         # If valid, slice the string up to the punctuation
#     #         full_sentence_punct = punctuated_full[:punct_idx]
            
#     #         # Now we must separate the "Context" from the "New Content"
#     #         # We do this by word counting the punctuated output.
#     #         # (Approximation: We assume Silero didn't insert/delete words, only chars)
#     #         total_punct_words = full_sentence_punct.split()
            
#     #         # The number of new words we are committing to = Total - Context
#     #         num_words_total = len(total_punct_words)
#     #         num_words_context = len(context_words)
            
#     #         num_new_committed = num_words_total - num_words_context
            
#     #         if num_new_committed > 0:
#     #             # We have a valid new sentence!
#     #             # Extract just the new part for translation
#     #             # (We rejoin the words to be safe about spacing)
#     #             flush_text_punct = " ".join(total_punct_words[num_words_context:])
#     #             flush_num_new_words = num_new_committed
#     #             should_flush = True
#     #             reason = "SENTENCE"

#     #     elif time_since_flush > MAX_SEGMENT_SILENCE:
#     #         # Timeout: Force flush everything new we have
#     #         # (We don't rely on Silero punctuation here, just raw flush to keep moving)
#     #         flush_text_punct = " ".join(new_words)
#     #         flush_num_new_words = len(new_words)
#     #         should_flush = True
#     #         reason = "TIMEOUT"

#     #     # --- 3. EXECUTE FLUSH ---
#     #     if should_flush and flush_num_new_words > 0:
#     #         # Safety clamp
#     #         if start_idx + flush_num_new_words > total_len:
#     #             flush_num_new_words = total_len - start_idx

#     #         # Update State
#     #         state["word_index"] += flush_num_new_words
#     #         state["last_flush_time"] = time.perf_counter()
            
#     #         # Update Context Buffer (Keep last 3 words of THIS flush)
#     #         # We go back to raw 'all_words' to ensure clean context for next loop
#     #         current_end_idx = state["word_index"]
#     #         # Take up to 3 words ending at current_end_idx
#     #         new_context_start = max(0, current_end_idx - 3)
#     #         state["last_context_words"] = all_words[new_context_start:current_end_idx]

#     #         print(f"[FLUSH:{reason}] '{flush_text_punct}'")
            
#     #         mt_queue.put({
#     #             "chunk_id": str(uuid.uuid4())[:6],
#     #             "text": flush_text_punct,
#     #             "reason": reason
#     #         })

#     # ------------------------------------------------
#     # HANDLERS
#     # ------------------------------------------------
#     # def recognizing_handler(evt):
#     #     # Continuous partial stream
#     #     smart_flush_check(evt.result.text)

#     # # def recognized_handler(evt):
#     # #     # Final sentence from Azure
#     # #     # Azure resets its buffer here, so we must reset ours
#     # #     print(f"[ASR FINAL] {evt.result.text}")
#     # #     state["stable_prefix"] = ""
#     # #     # Optional: One final check if anything was left over? 
#     # #     # Usually recognized contains the same text as the last recognizing.
#     # def recognized_handler(evt):
#     #     print(f"[ASR FINAL] {evt.result.text}")
#     #     state["word_index"] = 0
#     #     state["last_context_words"] = [] # Reset context on final
#     def recognizing_handler(evt):
#         # Continuous stream: Use Lag
#         smart_flush_check(evt.result.text, is_final=False)

#     def recognized_handler(evt):
#         # Final event: Force Catch-up (Flush everything remaining)
#         print(f"[ASR FINAL] {evt.result.text}")
#         smart_flush_check(evt.result.text, is_final=True)
        
#         # Reset state for the next utterance
#         state["word_index"] = 0
#         state["last_context_words"] = []

#     recognizer.recognizing.connect(recognizing_handler)
#     recognizer.recognized.connect(recognized_handler)

#     print("ASR + Silero Punctuation Pipeline Started...")
#     recognizer.start_continuous_recognition_async().get()

#     try:
#         while True: time.sleep(0.5)
#     except KeyboardInterrupt:
#         recognizer.stop_continuous_recognition_async().get()
#         mt_queue.put(None)
#         tts_queue.put(None)

# if __name__ == "__main__":
#     mp.set_start_method("spawn", force=True)
#     main()

import os
import time
import uuid
import re
import multiprocessing as mp
import requests
import azure.cognitiveservices.speech as speechsdk
import torch
import json

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

secrets_dict = json.load(open("secrets.json", "r"))

SPEECH_KEY = secrets_dict.get("speech_key") or os.getenv("SPEECHKEY")
SPEECH_REGION = secrets_dict.get("SERVICE_LOCATION") or os.getenv("REGION") or "eastus"

TRANSLATOR_KEY = secrets_dict.get("translate_key") or os.getenv("TRANSLATEKEY")
TRANSLATOR_REGION = secrets_dict.get("translate_region") or os.getenv("TRANSLATEREGION") or "global"

SOURCE_LANGUAGE = "en-US"
TARGET_LANGUAGE = "de-DE"
TTS_VOICE = "de-DE-KatjaNeural"

USE_MIC = True
AUDIO_FILE = "locked.wav"

# Silero Configuration
MIN_WORDS_FOR_SILERO = 4        
MAX_SEGMENT_SILENCE = 2.5       
TTS_RATE = "1.3"

LAG_BUFFER_WORDS = 3

# -------------------------------------------------------------------
# MODEL LOADER (Main Process)
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
# WORKERS (MT + TTS)
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

def create_tts_synthesizer():
    speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    speech_config.speech_synthesis_voice_name = TTS_VOICE
    audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
    return speechsdk.SpeechSynthesizer(speech_config, audio_config)

def mt_worker(mt_in_queue, tts_out_queue):
    print("[MT WORKER] Ready")
    while True:
        item = mt_in_queue.get()
        if item is None: break

        text = item["text"]
        translated = translate_text(text)
        
        if translated:
            # We pass the timestamps through unchanged
            tts_out_queue.put({**item, "translated_text": translated})

def tts_worker(tts_in_queue):
    print("[TTS WORKER] Ready")
    synthesizer = create_tts_synthesizer()
    while True:
        item = tts_in_queue.get()
        if item is None: break

        # --- TIMING CALCULATION ---
        current_time = time.perf_counter()
        
        # 1. When did we start capturing this sentence?
        t0_recognition = item.get("first_recognition_time", current_time)
        # 2. When did we decide to flush it to MT?
        t1_flush = item.get("asr_flush_time", current_time)
        
        # Calculate Durations
        accumulation_time = (t1_flush - t0_recognition) * 1000
        processing_time = (current_time - t1_flush) * 1000
        total_latency = (current_time - t0_recognition) * 1000

        print(f"[{item['chunk_id']}] ⏱️ Total: {total_latency:.0f}ms (Accumulate: {accumulation_time:.0f}ms + Process: {processing_time:.0f}ms)")
        # --------------------------

        ssml = f"""<speak version="1.0" xml:lang="{TARGET_LANGUAGE}">
                <voice name="{TTS_VOICE}">
                    <prosody rate="{TTS_RATE}">{item['translated_text']}</prosody>
                </voice></speak>"""
        try:
            synthesizer.speak_ssml_async(ssml).get()
        except Exception as e:
            print(f"[TTS ERROR] {e}")

# -------------------------------------------------------------------
# MAIN PROCESS STATE
# -------------------------------------------------------------------

state = {
    "word_index": 0,
    "last_flush_time": time.perf_counter(),
    "last_context_words": [],
    "first_recognition_time": None # NEW: Tracks start of current utterance
}

def clean_text_for_silero(text):
    return text.replace(".", "").replace("?", "").replace("!", "").replace(",", "").lower().strip()

def count_words(text):
    return len(text.split())

# -------------------------------------------------------------------
# MAIN ASR LOOP
# -------------------------------------------------------------------

def main():
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
    
    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True) if USE_MIC else speechsdk.audio.AudioConfig(filename=AUDIO_FILE)
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
        
        # Context Prepending
        context_words = state["last_context_words"]
        input_words = context_words + new_words
        
        input_str = " ".join(input_words)
        clean_input = clean_text_for_silero(input_str)
        
        try:
            punctuated_full = apply_te(clean_input, lan='en')
        except Exception:
            punctuated_full = input_str

        # Find Stable Cut Point
        match_iter = re.finditer(r'[.?!](?=\s|$)', punctuated_full)
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
        
        # Execute Flush
        if should_flush and flush_num_new_words > 0:
            if start_idx + flush_num_new_words > total_len:
                flush_num_new_words = total_len - start_idx

            state["word_index"] += flush_num_new_words
            state["last_flush_time"] = time.perf_counter()
            
            current_end_idx = state["word_index"]
            new_context_start = max(0, current_end_idx - 4)
            state["last_context_words"] = all_words[new_context_start:current_end_idx]

            print(f"[FLUSH:{reason}] '{flush_text_punct}'")
            
            # --- SEND TIMESTAMPS ---
            # If for some reason we missed the start time (edge case), default to now
            t0 = state["first_recognition_time"] or time.perf_counter()

            mt_queue.put({
                "chunk_id": str(uuid.uuid4())[:6],
                "text": flush_text_punct,
                "reason": reason,
                "asr_flush_time": time.perf_counter(), # End of Recognition Event
                "first_recognition_time": t0           # Start of Recognition Event
            })

    def recognizing_handler(evt):
        # Capture the VERY first moment we hear something for this stream
        if state["first_recognition_time"] is None:
            state["first_recognition_time"] = time.perf_counter()
            
        smart_flush_check(evt.result.text, is_final=False)

    def recognized_handler(evt):
        print(f"[ASR FINAL] {evt.result.text}")
        smart_flush_check(evt.result.text, is_final=True)
        
        # Reset Logic for next utterance
        state["word_index"] = 0
        state["last_context_words"] = []
        state["first_recognition_time"] = None # Reset clock for next sentence

    recognizer.recognizing.connect(recognizing_handler)
    recognizer.recognized.connect(recognized_handler)

    print("ASR + Silero Punctuation Pipeline Started...")
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