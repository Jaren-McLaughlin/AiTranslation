# README

### List of files and their purpose
- silero_punctuate - for silero model testing. This is what predicts where punctuation should go in a sentence. It's fast and light but also not as accurate as a transformer. I've found it only messes up a couple sentences in an entire talk, and those still work kinda well unless we're talking about an asian SOV language.
- tannerPiepeline - Tanner's code that does segmentation based solely on time
- tannerPlusSilero - Tanner's code I threw together with some gemini code to segment based on silero's predicted punctuation as opposed to the timing-based splitting.
- tannerSileroSimulate - you can feed in mp4s into this since it gets tiring and awkward just speaking into a mic. Lets you see a little of how the tour might go by comparing text to where you are in the audio. It generates the csv files
- CSV log files - some more detailed information about the latency. Accumulation is the ASR delay, processing includes sending the recognized text to get translated, and latency stops as soon as the new translation is uttered. Total latency combines the two. Gap is the time between translated utterances.
- dataSummary.ipynb - an easy way to repeatably look at data. You'll need to make sure you have jupyter installed.
- requirements.txt - includes the pertinent libraries installed via pip necessary for the code to run. You might run into a few things installed outside of pip like ffmpeg but chat or gemini can walk you through those. This set of libraries might have a few libraries that are decently large which most of these files don't require to run, so there's something to just manually fetching the right libraries and then regenerating the requirements.txt

- tannerPlusSileroChatTest.py - testing out some optimizations suggested by chat, looks pretty promising with portuguese for now, keeping latency around 4-7 seconds and almost always below 10s. It's actually mostly under 10 for chinese as well.