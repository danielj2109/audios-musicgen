import torch
from scipy.io.wavfile import write
from transformers import AutoProcessor, MusicgenForConditionalGeneration

MODEL_ID = "facebook/musicgen-small"
DEVICE = "cpu"

PROMPT = "An upbeat modern salsa song with a bright piano montuno, powerful brass section with trumpets and trombones, energetic congas, timbales and bongos, a walking bass line, and cowbell keeping the clave rhythm. Fast tempo around 180 BPM, joyful and festive mood, Caribbean party atmosphere, rich horn melodies, exciting percussion breaks, and a warm, polished studio mix perfect for dancing."

print("Cargando Musicgen...")

processor = AutoProcessor.from_pretrained(MODEL_ID)

model = MusicgenForConditionalGeneration.from_pretrained(MODEL_ID)

model = model.to(DEVICE)
model.eval()

print("Preparando prompt...")

inputs = processor(text=[PROMPT], padding=True, return_tensors="pt").to(DEVICE)

print("Generando musica...")

with torch.inference_mode():
    audio = model.generate(**inputs, do_sample=True, guidance_scale=3.0, max_new_tokens=1500)

sample_rate = model.config.audio_encoder.sampling_rate
waveform = audio[0, 0].cpu().float().numpy()

write("musica.wav", sample_rate, waveform)
print("Musica generada")