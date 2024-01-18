import os
import time
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "distil-whisper/distil-small.en"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    torch_dtype=torch_dtype,
    device=device,
)

if os.path.exists("results.txt"):
    os.remove("results.txt")
f = open("results.txt", "w")

# write to f
for i in range(1):
    dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    sample = dataset[i]["audio"]

    start = time.time()
    result = pipe(sample)
    end = time.time()

    orig_len = sample['array']
    orig_len = len(orig_len)/sample['sampling_rate']

    f.write(f"======= Audio {i} =======\n")
    f.write(result["text"] + "\n")
    f.write(f"audio length: {orig_len}\n")
    f.write(f"Time elapsed: {end - start}\n")

f.close()
