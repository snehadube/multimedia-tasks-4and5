# Task 4: Voice Cloning

## Objective
Clone a voice from a short audio sample and generate new speech in that voice.

## Tools
Python, Google Colab (T4 GPU), Coqui TTS with the XTTS-v2 model
(open-source, free alternative to ElevenLabs).

## Approach
1. Recorded a short, clean sample of my own voice (samples/sample.wav).
2. Loaded the XTTS-v2 multilingual model in Google Colab.
3. Generated new speech from text, using the sample as the speaker reference.

## How to run
Open clone_xtts_colab.ipynb in Google Colab, select a GPU runtime,
upload your own sample.wav and run all cells.

## Results
- Input sample: samples/sample.wav
- Cloned output: outputs/cloned_output.wav

## Limitations
- Quality depends on the length and clarity of the sample.
- The cloned voice is similar but not identical to the original.
- ElevenLabs voice cloning needs a paid plan, so a free open-source
  model was used instead.

## Ethical note
Only my own voice was cloned, with my own consent. Voice cloning must
not be used to impersonate other people.