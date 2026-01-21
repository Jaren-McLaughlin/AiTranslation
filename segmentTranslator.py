import os
import re
import json
import requests
import sacrebleu
import sys
import time

# Load secrets
try:
    secrets_dict = json.load(open("secrets.json", "r"))
except FileNotFoundError:
    print("Error: 'secrets.json' not found.")
    sys.exit(1)

TRANSLATOR_KEY = secrets_dict.get("translate_key") or os.getenv("TRANSLATEKEY")
TRANSLATOR_REGION = secrets_dict.get("translate_region") or os.getenv("TRANSLATEREGION") or "global"

SOURCE_LANGUAGE = "en-US"
TARGET_LANGUAGE = "zh-CN"  

mt_session = requests.Session()

def translate_batch(texts: list, to_lang: str = TARGET_LANGUAGE) -> list:
    """Translates a batch of sentences."""
    if not texts: return []
    
    endpoint = "https://api.cognitive.microsofttranslator.com/translate"
    params = {"api-version": "3.0", "to": to_lang}
    headers = {
        "Ocp-Apim-Subscription-Key": TRANSLATOR_KEY,
        "Ocp-Apim-Subscription-Region": TRANSLATOR_REGION,
        "Content-Type": "application/json"
    }
    body = [{"text": t} for t in texts]

    try:
        resp = mt_session.post(endpoint, params=params, headers=headers, json=body, timeout=10)
        resp.raise_for_status()
        results = resp.json()
        return [item["translations"][0]["text"] for item in results]
    except Exception as e:
        print(f"[MT ERROR] Batch failed: {e}")
        return [""] * len(texts)

def segment_text(text: str):
    """Simple segmentation by sentence endings."""
    if not text: return []
    # Split on . ? !
    segments = [s.strip() for s in re.split(r'(?<=[.?!])\s+', text) if s.strip()]
    comma_segments = [s.strip() for s in re.split(r'(?<=[,.?!])\s+', text) if s.strip()]
    return segments, comma_segments

def main():
    if not os.path.exists("browningOriginal.txt") or not os.path.exists("browningASR.txt"):
        print("Error: Input text files not found.")
        return

    with open("browningOriginal.txt", "r", encoding="utf-8") as f_orig:
        with open("browningASR.txt", "r", encoding="utf-8") as f_asr:
            orig_text = f_orig.read().strip()
            asr_text = f_asr.read().strip()
            
            # --- CRITICAL CHECK ---
            if not orig_text:
                print("❌ Error: 'browningOriginal.txt' is empty!")
                return
            if not asr_text:
                print("❌ Error: 'browningASR.txt' is empty!")
                return

            # 1. SEGMENT (Only for the API's sake)
            orig_segments, orig_comma_segments = segment_text(orig_text)
            asr_segments, asr_comma_segments = segment_text(asr_text)
            
            print(f"Original: {len(orig_segments)} segments")
            print(f"ASR:      {len(asr_segments)} segments")
            print(f"Original (Comma): {len(orig_comma_segments)} segments")
            print(f"ASR (Comma):      {len(asr_comma_segments)} segments")

            print(orig_segments[:3])
            print(asr_segments[:3])
            print(orig_comma_segments[:3])
            print(asr_comma_segments[:3])

            # return

            # 2. BATCH TRANSLATE
            BATCH_SIZE = 25 
            
            translated_asr_list = []
            print(f"Translating ASR...")
            for i in range(0, len(asr_segments), BATCH_SIZE):
                batch = asr_segments[i : i + BATCH_SIZE]
                translated_asr_list.extend(translate_batch(batch))
                time.sleep(0.5) 

            translated_asr_comma_list = []
            print(f"Translating ASR...")
            for i in range(0, len(asr_comma_segments), BATCH_SIZE):
                batch = asr_comma_segments[i : i + BATCH_SIZE]
                translated_asr_comma_list.extend(translate_batch(batch))
                time.sleep(0.5) 
            
            translated_orig_list = []
            print(f"Translating Original...")
            for i in range(0, len(orig_segments), BATCH_SIZE):
                batch = orig_segments[i : i + BATCH_SIZE]
                translated_orig_list.extend(translate_batch(batch))
                time.sleep(0.5)

            # 3. JOIN (The Fix)
            # We reconstruct the full document so misalignment doesn't matter
            full_translation_orig = " ".join(translated_orig_list)
            full_translation_asr = " ".join(translated_asr_list)
            full_translation_asr_comma = " ".join(translated_asr_comma_list)

            # 4. SCORE DOCUMENT-LEVEL
            # SacreBLEU expects lists, so we wrap our giant strings in a list
            # Hypothesis: The ASR-based translation
            # Reference: The Original-based translation

            # tokenizer_opt = 'zh' if TARGET_LANGUAGE == "zh-CN" else '13a'
            if TARGET_LANGUAGE == "ja-JP":
                tokenizer_opt = 'ja-mecab'
            elif TARGET_LANGUAGE == "ko-KR":
                tokenizer_opt = 'char'
            elif TARGET_LANGUAGE == "zh-CN":
                tokenizer_opt = 'zh'
            else:
                tokenizer_opt = '13a'

            reference_translation = ""
            with open("browningZhs.txt", "r", encoding="utf-8") as f_ref:
                reference_translation = f_ref.read().strip()

            orig_bleu = sacrebleu.corpus_bleu([full_translation_orig], [[reference_translation]], tokenize=tokenizer_opt)
            asr_bleu = sacrebleu.corpus_bleu([full_translation_asr], [[reference_translation]], tokenize=tokenizer_opt)
            # bleu = sacrebleu.corpus_bleu([full_translation_asr], [[full_translation_orig]], tokenize=tokenizer_opt)
            # comma_bleu = sacrebleu.corpus_bleu([full_translation_asr_comma], [[full_translation_orig]], tokenize=tokenizer_opt)
            comma_bleu = sacrebleu.corpus_bleu([full_translation_asr_comma], [[reference_translation]], tokenize=tokenizer_opt)
            
            print("\n" + "="*40)
            print(f"Orig sentence-based BLEU Score: {orig_bleu.score:.2f}")
            print(f"ASR sentence-based BLEU Score: {asr_bleu.score:.2f}")
            print(f"BLEU Score (Comma splitting): {comma_bleu.score:.2f}")
            print("="*40)

if __name__ == "__main__":
    main()