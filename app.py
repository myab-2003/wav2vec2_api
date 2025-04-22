from fastapi import FastAPI, UploadFile, File
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import soundfile as sf
from io import BytesIO

app = FastAPI()

# تحميل الموديل والمعالج من Hugging Face
processor = Wav2Vec2Processor.from_pretrained("myab/wav2vec2-ar-model")
model = Wav2Vec2ForCTC.from_pretrained("myab/wav2vec2-ar-model")

@app.post("/transcribe/")
async def transcribe(audio: UploadFile = File(...)):
    # قراءة ملف الصوت
    audio_data = await audio.read()
    audio_input, sample_rate = sf.read(BytesIO(audio_data))

    # تجهيز البيانات
    inputs = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]

    return {"transcription": transcription}
