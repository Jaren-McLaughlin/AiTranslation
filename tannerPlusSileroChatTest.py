import os
import time
import uuid
import re
import json
import requests
import multiprocessing as mp
import azure.cognitiveservices.speech as speechsdk
import torch

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

secrets = json.load(open("secrets.json", "r"))

SPEECH_KEY = secrets["speech_key"]
SPEECH_REGION = secrets.get("SERVICE_LOCATION", "eastus")

TRANSLATOR_KEY = secrets["translate_key"]
TRANSLATOR_REGION = secrets.get("translate_region", "global")

SOURCE_LANGUAGE = "en-US"
TARGET_LANGUAGE = "de-DE"
TTS_VOICE = "de-DE-KatjaNeural"

USE_MIC = False                 # ← set False for repeatable testing
AUDIO_FILE = r"C:\Research\TranslateCapstone\test.wav"

MIN_WORDS_FOR_SILERO = 4
LAG_BUFFER_WORDS = 2             # lowered
TTS_RATE = "1.4"

# -------------------------------------------------------------------
# SILERO (loaded once)
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
# MT + TTS WORKERS
# -------------------------------------------------------------------

mt_session = requests.Session()

def translate_text(text):
    if not text.strip():
        return ""

    r = mt_session.post(
        "https://api.cognitive.microsofttranslator.com/translate",
        params={"api-version": "3.0", "to": TARGET_LANGUAGE},
        headers={
            "Ocp-Apim-Subscription-Key": TRANSLATOR_KEY,
            "Ocp-Apim-Subscription-Region": TRANSLATOR_REGION,
            "Content-Type": "application/json"
        },
        json=[{"text": text}],
        timeout=3
    )
    r.raise_for_status()
    return r.json()[0]["translations"][0]["text"]

def create_tts():
    cfg = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    cfg.speech_synthesis_voice_name = TTS_VOICE
    return speechsdk.SpeechSynthesizer(
        cfg,
        speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
    )

def mt_worker(mt_q, tts_q):
    while True:
        item = mt_q.get()
        if item is None:
            break
        item["translated"] = translate_text(item["text"])
        tts_q.put(item)

def tts_worker(tts_q):
    synth = create_tts()
    while True:
        item = tts_q.get()
        if item is None:
            break

        now = time.perf_counter()
        t0 = item["t0"]
        t1 = item["t1"]

        print(
            f"[{item['id']}] ⏱️ {((now-t0)*1000):.0f}ms "
            f"(acc {(t1-t0)*1000:.0f} + proc {(now-t1)*1000:.0f})"
        )

        txt = item["translated"]
        if len(txt.split()) <= 6:
            synth.speak_text_async(txt).get()
        else:
            ssml = f"""
            <speak version="1.0" xml:lang="{TARGET_LANGUAGE}">
              <voice name="{TTS_VOICE}">
                <prosody rate="{TTS_RATE}">{txt}</prosody>
              </voice>
            </speak>
            """
            synth.speak_ssml_async(ssml).get()

# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------

def main():
    apply_te = load_silero()

    mt_q = mp.SimpleQueue()
    tts_q = mp.SimpleQueue()

    mp.Process(target=mt_worker, args=(mt_q, tts_q), daemon=True).start()
    mp.Process(target=tts_worker, args=(tts_q,), daemon=True).start()

    # warm-up
    translate_text("hello")
    create_tts().speak_text_async(" ").get()

    speech_cfg = speechsdk.SpeechConfig(
        subscription=SPEECH_KEY,
        region=SPEECH_REGION
    )
    speech_cfg.speech_recognition_language = SOURCE_LANGUAGE
    speech_cfg.set_property(
        speechsdk.PropertyId.SpeechServiceResponse_StablePartialResultThreshold,
        "1"
    )

    audio_cfg = (
        speechsdk.audio.AudioConfig(use_default_microphone=True)
        if USE_MIC else
        speechsdk.audio.AudioConfig(filename=AUDIO_FILE)
    )

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
        t1 = time.perf_counter()
        mt_q.put({
            "id": str(uuid.uuid4())[:6],
            "text": text,
            "t0": state["t0"],
            "t1": t1,
            "reason": reason
        })

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

        for m in re.finditer(r"[,.?!]", punct):
            tail = punct[m.end():].split()
            if not final and len(tail) >= LAG_BUFFER_WORDS:
                continue

            segment = punct[:m.end()].strip()
            wc = len(segment.split())
            state["idx"] += wc
            flush(segment, "EARLY")
            return

        if final:
            flush(combined, "FINAL")
            state["idx"] = 0
            state["t0"] = None

    recognizer.recognizing.connect(lambda e: process(e.result.text))
    recognizer.recognized.connect(lambda e: process(e.result.text, True))

    print("🚀 Live translation started")
    recognizer.start_continuous_recognition_async().get()

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        recognizer.stop_continuous_recognition_async().get()
        mt_q.put(None)
        tts_q.put(None)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
