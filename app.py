from fastapi import FastAPI, UploadFile, File
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import torchaudio
from io import BytesIO

app = FastAPI()

# تحميل الموديل والمعالج من Hugging Face
processor = Wav2Vec2Processor.from_pretrained("myab/wav2vec2-ar-model")
model = Wav2Vec2ForCTC.from_pretrained("myab/wav2vec2-ar-model")

@app.post("/transcribe/")
async def transcribe(audio: UploadFile = File(...)):
    audio_data = await audio.read()
    with BytesIO(audio_data) as f:
        waveform, sample_rate = torchaudio.load(f)

    inputs = processor(waveform.squeeze().numpy(), sampling_rate=sample_rate, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]

    return {"transcription": transcription}
