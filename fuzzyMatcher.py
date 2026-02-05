from rapidfuzz import process, fuzz

class TranslationMemoryMatcher:
    def __init__(self, filepath):
        """
        Loads the candidate segments into memory once on initialization.
        This is critical for low latency.
        """
        self.candidates = self._load_candidates(filepath)
        print(f"Loaded {len(self.candidates)} phrases into memory.")

    def _load_candidates(self, filepath):
        """Helper to read valid lines from the file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            # unique candidates only, stripped of whitespace
            return list(set(line.strip() for line in f if line.strip()))

    def find_best_match(self, asr_input, score_threshold=85):
        if not self.candidates or not asr_input:
            return None

        # 1. Run the fuzzy match
        result = process.extractOne(
            asr_input, 
            self.candidates, 
            scorer=fuzz.token_set_ratio, 
            score_cutoff=score_threshold
        )

        if result:
            match_string, score, index = result
            
            # --- THE FIX: LENGTH GUARDRAIL ---
            # Calculate the length difference ratio
            len_input = len(asr_input)
            len_match = len(match_string)
            
            # If the match is more than 20% longer than the input, 
            # it's likely predicting the rest of the sentence. REJECT IT.
            # We allow it to be shorter (ASR might add fluff), but not much longer.
            if len_match > len_input * 1.2:
                # Optional: specific check for very short inputs to avoid "I" matching "I am..."
                return None
                
            return match_string, score
            
        return None

# --- Usage Example ---

# 1. Initialize once (Simulate loading your document)
# Assume 'tours_phrases.txt' contains: 
# "Please keep your hands inside the vehicle"
# "To your left is the historic library"
# matcher = TranslationMemoryMatcher('tours_phrases.txt')

# # 2. Simulate imperfect ASR input
# asr_output = "uh please keep hands inside vehicle"
# # Hook up to system like this  \/

# # 3. Run the match
# match = matcher.find_best_match(asr_output)

# if match:
#     print(f"Found Match: '{match[0]}' (Confidence: {match[1]})")
#     # Output: Found Match: 'Please keep your hands inside the vehicle' (Confidence: 100.0)
# else:
#     print("No sufficient match found, sending raw ASR to translator.")