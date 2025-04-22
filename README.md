# Wav2Vec2 Arabic Speech Recognition API

This project is a FastAPI-based service that uses a pretrained Wav2Vec2 model to transcribe Arabic speech from uploaded audio files.

## Endpoints

- `POST /transcribe/`: Upload an audio file (WAV) and get the transcription.

## Requirements

- Python 3.8+
- FastAPI
- Transformers
- Torch
- Torchaudio

## Run locally

```bash
uvicorn app:app --reload
