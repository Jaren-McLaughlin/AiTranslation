import os
import re
import json
import requests
import sacrebleu
import sys
import time
from comet import download_model, load_from_checkpoint

COMET_AVAILABLE = True

# Load secrets
try:
    secrets_dict = json.load(open("secrets.json", "r"))
except FileNotFoundError:
    print("Error: 'secrets.json' not found.")
    sys.exit(1)

TRANSLATOR_KEY = secrets_dict.get("translate_key") or os.getenv("TRANSLATEKEY")
TRANSLATOR_REGION = secrets_dict.get("translate_region") or os.getenv("TRANSLATEREGION") or "global"

SOURCE_LANGUAGE = "en-US"
# Change this to test other languages: "fr-FR", "es-ES", "ja-JP", "ko-KR", "pt-BR", "zh-CN"
TARGET_LANGUAGE = "fr-FR" 

mt_session = requests.Session()

# --- GLOBAL COMET MODEL (Load once) ---
comet_model = None

def load_comet():
    """Loads the heavy COMET model only if needed."""
    global comet_model
    if not COMET_AVAILABLE: return None
    if comet_model is not None: return comet_model
    
    print("[INIT] Loading COMET model (Unbabel/wmt22-comet-da)...")
    try:
        # Downloads model to local cache (~2GB)
        model_path = download_model("Unbabel/wmt22-comet-da")
        comet_model = load_from_checkpoint(model_path)
        return comet_model
    except Exception as e:
        print(f"❌ Failed to load COMET: {e}")
        return None

def calculate_comet(sources, hypotheses, references):
    """
    Calculates COMET score.
    COMET needs: Source (Src), Translation (Mt), and Reference (Ref).
    """
    model = load_comet()
    if not model: return 0.0

    print(f"   Computing COMET score for {len(sources)} segments...")
    
    # Prepare data for COMET: List of dicts
    data = [
        {"src": s, "mt": h, "ref": r} 
        for s, h, r in zip(sources, hypotheses, references)
    ]
    
    # Run prediction (batch_size=8 is safe for CPU)
    model_output = model.predict(data, batch_size=8, gpus=0)
    
    # system_score is the average of all segment scores
    return model_output.system_score * 100 # Scale to 0-100 to match BLEU

def translate_batch(texts: list, to_lang: str = TARGET_LANGUAGE) -> list:
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
    if not text: return [], []
    # Segments for API (Full stop)
    segments = [s.strip() for s in re.split(r'(?<=[.?!])\s+', text) if s.strip()]
    # Segments for Comma Test
    comma_segments = [s.strip() for s in re.split(r'(?<=[,.?!])\s+', text) if s.strip()]
    return segments, comma_segments

def main():
    if not os.path.exists("browningOriginal.txt") or not os.path.exists("browningASR.txt") or not os.path.exists(f"browning{TARGET_LANGUAGE}.txt"):
        print(os.path.exists("browningOriginal.txt"))
        print(os.path.exists("browningASR.txt"))
        print(os.path.exists(f"browning{TARGET_LANGUAGE}.txt"))
        print("Error: Input text files (Original, ASR, or Reference) not found.")
        return

    with open("browningOriginal.txt", "r", encoding="utf-8") as f_orig:
        with open("browningASR.txt", "r", encoding="utf-8") as f_asr:
             with open(f"browning{TARGET_LANGUAGE}.txt", "r", encoding="utf-8") as f_ref: # Reference file
                orig_text = f_orig.read().strip()
                asr_text = f_asr.read().strip()
                ref_text = f_ref.read().strip() # This should be the human translation of the target language
                
                if not orig_text or not asr_text or not ref_text:
                    print("❌ Error: One of the input files is empty!")
                    return

                # 1. SEGMENT
                orig_segments, _ = segment_text(orig_text)
                asr_segments, asr_comma_segments = segment_text(asr_text)
                
                print(f"Original Segments: {len(orig_segments)}")
                print(f"ASR Segments:      {len(asr_segments)}")
                print(f"ASR Comma Segs:    {len(asr_comma_segments)}")

                # 2. BATCH TRANSLATE
                BATCH_SIZE = 25 
                
                translated_asr_list = []
                print(f"Translating ASR (Sentence Split)...")
                for i in range(0, len(asr_segments), BATCH_SIZE):
                    batch = asr_segments[i : i + BATCH_SIZE]
                    translated_asr_list.extend(translate_batch(batch))
                    time.sleep(0.5) 

                translated_asr_comma_list = []
                print(f"Translating ASR (Comma Split)...")
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

                # 3. RECONSTRUCT DOCUMENTS
                full_translation_orig = " ".join(translated_orig_list)
                full_translation_asr = " ".join(translated_asr_list)
                full_translation_asr_comma = " ".join(translated_asr_comma_list)

                # 4. BLEU SCORING (Baseline)
                if TARGET_LANGUAGE == "zh-CN": tokenizer_opt = 'zh'
                elif TARGET_LANGUAGE == "ja-JP": tokenizer_opt = 'char' # Fallback if mecab missing
                elif TARGET_LANGUAGE == "ko-KR": tokenizer_opt = 'char'
                else: tokenizer_opt = '13a'

                bleu_orig = sacrebleu.corpus_bleu([full_translation_orig], [[ref_text]], tokenize=tokenizer_opt)
                bleu_asr = sacrebleu.corpus_bleu([full_translation_asr], [[ref_text]], tokenize=tokenizer_opt)
                bleu_comma = sacrebleu.corpus_bleu([full_translation_asr_comma], [[ref_text]], tokenize=tokenizer_opt)

                print("\n" + "="*40)
                print(f"METRICS FOR {TARGET_LANGUAGE}")
                print("="*40)
                print(f"BLEU (Orig):  {bleu_orig.score:.2f}")
                print(f"BLEU (ASR):   {bleu_asr.score:.2f}")
                print(f"BLEU (Comma): {bleu_comma.score:.2f}")
                print("-" * 20)

                # 5. COMET SCORING (Advanced)
                # COMET needs lists of segments, not one giant string.
                # However, since our reference is one giant string (ref_text), 
                # we must treat the entire document as 1 segment to be fair, 
                # OR split the reference into sentences if you had a sentence-aligned reference.
                # Assuming 'ref_text' is just a raw block of text, we treat the whole doc as 1 sample.
                
                comet_orig = calculate_comet([orig_text], [full_translation_orig], [ref_text])
                comet_asr = calculate_comet([orig_text], [full_translation_asr], [ref_text])
                comet_comma = calculate_comet([orig_text], [full_translation_asr_comma], [ref_text])

                print(f"COMET (Orig):  {comet_orig:.2f}")
                print(f"COMET (ASR):   {comet_asr:.2f}")
                print(f"COMET (Comma): {comet_comma:.2f}")
                print("="*40)

if __name__ == "__main__":
    main()