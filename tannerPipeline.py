import os
import time
import uuid
import re
import string
import multiprocessing as mp

import requests
import azure.cognitiveservices.speech as speechsdk

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

import json
secrets_dict = json.load(open("secrets.json", "r"))

# SPEECH_KEY = os.getenv("ASRKEY")
SPEECH_KEY = secrets_dict.get("speech_key") or os.getenv("SPEECHKEY")
# SPEECH_REGION = os.getenv("REGION") or "eastus"
SPEECH_REGION = secrets_dict.get("SERVICE_LOCATION") or os.getenv("REGION") or "eastus"

# TRANSLATOR_KEY = os.getenv("TRANSLATEKEY")
TRANSLATOR_KEY = secrets_dict.get("translate_key")
# TRANSLATOR_REGION = "global"
TRANSLATOR_REGION = secrets_dict.get("translate_region")

SOURCE_LANGUAGE = "en-US"
TARGET_LANGUAGE = "es-MX"
TTS_VOICE = "es-MX-DaliaNeural"

USE_MIC = True
# USE_MIC = False
AUDIO_FILE = "locked.wav"

# Segmentation knobs
USE_PARTIAL_SEGMENTATION = True   # True = incremental; False = FINAL-only
WORDS_PER_CHUNK = 20               # N words per chunk
PUNCT_FLUSH_CHARS = ".?!,"
MAX_SEGMENT_SILENCE = 3.0         # seconds since last flush

# TTS speed (SSML prosody rate)
TTS_RATE = "1.3"                  # 1.0 = normal, 1.3 = 30% faster


# -------------------------------------------------------------------
# MT + TTS HELPERS (used inside worker processes)
# -------------------------------------------------------------------

def translate_text(text: str, to_lang: str = TARGET_LANGUAGE) -> str:
    text = text.strip()
    if not text:
        return ""

    endpoint = "https://api.cognitive.microsofttranslator.com/translate"
    params = {"api-version": "3.0", "to": to_lang}

    headers = {
        "Ocp-Apim-Subscription-Key": TRANSLATOR_KEY,
        "Ocp-Apim-Subscription-Region": TRANSLATOR_REGION,
        "Content-Type": "application/json"
    }

    try:
        resp = requests.post(
            endpoint,
            params=params,
            headers=headers,
            json=[{"text": text}],
            timeout=8
        )
        resp.raise_for_status()
        return resp.json()[0]["translations"][0]["text"]
    except Exception as e:
        print(f"[MT ERROR] {e} | text='{text[:80]}...'")
        return ""


def create_tts_synthesizer():
    speech_config = speechsdk.SpeechConfig(
        subscription=SPEECH_KEY,
        region=SPEECH_REGION
    )
    speech_config.speech_synthesis_voice_name = TTS_VOICE
    audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
    synthesizer = speechsdk.SpeechSynthesizer(speech_config, audio_config)
    print("[TTS WORKER] synthesizer =", synthesizer)
    return synthesizer


# -------------------------------------------------------------------
# WORKERS
# -------------------------------------------------------------------

def mt_worker(mt_in_queue: mp.Queue, tts_out_queue: mp.Queue):
    """
    MT worker: receives chunks from ASR/segmentation,
    runs translation, pushes to TTS queue.
    """
    print("[MT WORKER] started")
    while True:
        item = mt_in_queue.get()
        if item is None:
            print("[MT WORKER] received shutdown signal")
            break

        chunk_id = item["chunk_id"]
        text = item["text"]
        flush_reason = item["reason"]
        created_time = item["created_time"]

        mt_start_time = time.perf_counter()
        translated = translate_text(text)
        mt_end_time = time.perf_counter()

        if not translated:
            print(f"[MT WORKER] chunk={chunk_id} empty translation")
            continue

        mt_duration_ms = (mt_end_time - mt_start_time) * 1000
        print(
            f"[MT WORKER] chunk={chunk_id} reason={flush_reason} "
            f"mt={mt_duration_ms:.1f}ms src='{text}' → '{translated}'"
        )

        tts_out_queue.put({
            "chunk_id": chunk_id,
            "src_text": text,
            "translated_text": translated,
            "reason": flush_reason,
            "created_time": created_time,
            "mt_start_time": mt_start_time,
            "mt_end_time": mt_end_time,
        })

    print("[MT WORKER] exiting")


def tts_worker(tts_in_queue: mp.Queue):
    """
    TTS worker: receives translated segments, speaks them,
    logs latency from ASR flush to TTS end.
    """
    print("[TTS WORKER] started")
    synthesizer = create_tts_synthesizer()

    while True:
        item = tts_in_queue.get()
        if item is None:
            print("[TTS WORKER] received shutdown signal")
            break

        chunk_id = item["chunk_id"]
        translated = item["translated_text"]
        created_time = item["created_time"]
        mt_start_time = item["mt_start_time"]
        mt_end_time = item["mt_end_time"]
        flush_reason = item["reason"]

        # Build SSML with speedup
        ssml = f"""<speak version="1.0" xml:lang="{TARGET_LANGUAGE}">
            <voice name="{TTS_VOICE}">
                <prosody rate="{TTS_RATE}">
                {translated}
                </prosody>
            </voice>
            </speak>
            """

        tts_start_time = time.perf_counter()
        try:
            synthesizer.speak_ssml_async(ssml).get()
        except Exception as e:
            print("[TTS SSML Request ERROR]", e)
        tts_end_time = time.perf_counter()

        # Latency metrics
        asr_to_mt_start_ms = (mt_start_time - created_time) * 1000
        mt_duration_ms = (mt_end_time - mt_start_time) * 1000
        asr_to_tts_start_ms = (tts_start_time - created_time) * 1000
        tts_duration_ms = (tts_end_time - tts_start_time) * 1000
        asr_to_tts_end_ms = (tts_end_time - created_time) * 1000

        print(
            f"[LATENCY] chunk={chunk_id} reason={flush_reason} "
            f"asr→mt_start={asr_to_mt_start_ms:.1f}ms, "
            f"mt={mt_duration_ms:.1f}ms, "
            f"asr→tts_start={asr_to_tts_start_ms:.1f}ms, "
            f"tts={tts_duration_ms:.1f}ms, "
            f"asr→tts_end={asr_to_tts_end_ms:.1f}ms"
        )

    print("[TTS WORKER] exiting")


# -------------------------------------------------------------------
# ASR CREATION (MAIN PROCESS)
# -------------------------------------------------------------------

def create_recognizer():
    speech_config = speechsdk.SpeechConfig(
        subscription=SPEECH_KEY,
        region=SPEECH_REGION
    )
    speech_config.speech_recognition_language = SOURCE_LANGUAGE

    # TrueText
    speech_config.set_property(
        speechsdk.PropertyId.SpeechServiceResponse_PostProcessingOption,
        "TrueText"
    )

    # Don’t end the session on silence
    speech_config.set_property(
        speechsdk.PropertyId.SpeechServiceConnection_EndSilenceTimeoutMs,
        "0"
    )

    if USE_MIC:
        audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
        print("[ASR] Using microphone input")
    else:
        audio_config = speechsdk.audio.AudioConfig(filename=AUDIO_FILE)
        print(f"[ASR] Using audio file: {AUDIO_FILE}")

    recognizer = speechsdk.SpeechRecognizer(speech_config, audio_config)
    return recognizer


# -------------------------------------------------------------------
# MONOTONIC SEGMENTATION STATE (MAIN PROCESS)
# -------------------------------------------------------------------

state = {
    "stable_prefix": "",
    "last_partial": "",
    "last_flush_time": time.perf_counter(),
}


def normalize_partial(partial: str) -> str:
    """
    Very simple monotonicity: ignore shrinking partials.
    For mic chaos, you can later add fuzzy prefix logic.
    """
    prev = state["last_partial"]
    if len(partial) < len(prev):
        return prev
    state["last_partial"] = partial
    return partial


def flush_segment(reason: str, full_text: str, mt_queue: mp.Queue, is_final=False):
    full_text = full_text.strip()
    if not full_text:
        return

    stable = state["stable_prefix"]

    # Compute suffix = new content after stable_prefix (simple version)
    if full_text.startswith(stable):
        suffix = full_text[len(stable):].strip()
    else:
        # fallback: treat entire text as new if prefix changed too much
        suffix = full_text

    if not suffix:
        return

    state["stable_prefix"] = full_text
    state["last_flush_time"] = time.perf_counter()

    chunk_id = str(uuid.uuid4())[:8]
    created_time = time.perf_counter()

    print(f"[ASR FLUSH:{reason}] chunk={chunk_id} segment='{suffix}'")

    # Push to MT worker
    mt_queue.put({
        "chunk_id": chunk_id,
        "text": suffix,
        "reason": reason,
        "created_time": created_time,
    })

    # if is_final:
    #     # Reset for next utterance
    #     state["stable_prefix"] = ""
    #     state["last_partial"] = ""
    #     state["last_flush_time"] = time.perf_counter()


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------

def main():
    if not SPEECH_KEY or not TRANSLATOR_KEY:
        raise RuntimeError("Missing SPEECH_KEY or TRANSLATOR_KEY")

    # Queues for MT and TTS
    mt_in_queue = mp.Queue()
    tts_in_queue = mp.Queue()

    # Start worker processes
    mt_proc = mp.Process(target=mt_worker, args=(mt_in_queue, tts_in_queue), daemon=True)
    tts_proc = mp.Process(target=tts_worker, args=(tts_in_queue,), daemon=True)

    mt_proc.start()
    tts_proc.start()

    recognizer = create_recognizer()

    # -----------------------------
    # ASR HANDLERS (MAIN PROCESS)
    # -----------------------------

    def recognizing_handler(evt):
        if not USE_PARTIAL_SEGMENTATION:
            return

        partial = evt.result.text
        if not partial:
            return

        partial = normalize_partial(partial)
        now = time.perf_counter()
        # print(f"[ASR PARTIAL] '{partial}'")

        stable = state["stable_prefix"]
        if partial.startswith(stable):
            suffix_text = partial[len(stable):].strip()
        else:
            suffix_text = partial

        suffix_words = suffix_text.split()

        # 1. PRIMARY: flush every N new words
        if len(suffix_words) >= WORDS_PER_CHUNK:
            flush_segment("WORD", partial, mt_in_queue)
            return

        # 2. Punctuation-based segmentation
        if partial and partial[-1] in PUNCT_FLUSH_CHARS:
            flush_segment("PUNCT", partial, mt_in_queue)
            return

        # 3. TIME-based fallback since last flush
        if now - state["last_flush_time"] > MAX_SEGMENT_SILENCE:
            flush_segment("TIME", partial, mt_in_queue)
            return

    def recognized_handler(evt):
        final = evt.result.text
        if not final:
            return

        # print(f"[ASR FINAL] '{final}'")

        # Do NOT flush FINAL content.
        # FINAL is only used to reset state for the next utterance.
        state["stable_prefix"] = ""
        state["last_partial"] = ""
        state["last_flush_time"] = time.perf_counter()

    recognizer.recognizing.connect(recognizing_handler)
    recognizer.recognized.connect(recognized_handler)

    # -----------------------------
    # SESSION HANDLERS (auto-restart)
    # -----------------------------

    def restart_recognizer_async(reason: str):
        import threading

        def _restart():
            print(f"[ASR RESTART] restarting due to {reason}")
            try:
                recognizer.stop_continuous_recognition_async().get()
            except Exception as e:
                print("[ASR RESTART] stop error:", e)
            try:
                recognizer.start_continuous_recognition_async().get()
                print("[ASR RESTART] restart complete")
            except Exception as e:
                print("[ASR RESTART] start error:", e)

        threading.Thread(target=_restart, daemon=True).start()

    def session_started(evt):
        print("[ASR EVENT] session_started")

    def session_stopped(evt):
        print("[ASR EVENT] session_stopped")
        if USE_MIC:
            restart_recognizer_async("session_stopped")

    def canceled(evt):
        print("[ASR EVENT] canceled:", evt.reason, evt.error_details)
        if USE_MIC:
            restart_recognizer_async("canceled")

    recognizer.session_started.connect(session_started)
    recognizer.session_stopped.connect(session_stopped)
    recognizer.canceled.connect(canceled)

    # -----------------------------
    # START ASR
    # -----------------------------

    print("Multiprocess 6-word interpreter running.")
    recognizer.start_continuous_recognition_async().get()
    print("[ASR] Recognizer started")

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Stopping...")
        try:
            recognizer.stop_continuous_recognition_async().get()
        except Exception as e:
            print("[ASR STOP] error:", e)

        # Signal workers to shut down
        mt_in_queue.put(None)
        tts_in_queue.put(None)

        mt_proc.join(timeout=5.0)
        tts_proc.join(timeout=5.0)
        print("Stopped.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()