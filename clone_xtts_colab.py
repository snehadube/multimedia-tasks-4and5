# Run this in Google Colab with a T4 GPU.
# Upload your own voice sample as sample.wav first.
# Install first: pip install coqui-tts

import os
os.environ["COQUI_TOS_AGREED"] = "1"

import torch
from TTS.api import TTS

device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

tts.tts_to_file(
    text="Hello, this is my cloned voice. This project demonstrates voice cloning using the XTTS version two model.",
    speaker_wav="sample.wav",
    language="en",
    file_path="cloned_output.wav",
)
print("Done")