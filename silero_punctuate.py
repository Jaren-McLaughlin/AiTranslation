import torch
import yaml
import time

model, example_texts, languages, punct, apply_te = torch.hub.load(
    repo_or_dir='snakers4/silero-models',
    model='silero_te',
    trust_repo=True
)

# print(model)

# It usually takes raw text and outputs punctuated text instantly

with open("tourTranscriptTestNoPunc.txt") as f:
    input_text = f.read()

    time_start = time.time()

    output = apply_te(input_text, lan='en')
    time_end = time.time()

with open("tourTranscriptTestPunc.txt", "w") as f:
    f.write(output)
    print(f"Processed in {time_end - time_start} seconds")