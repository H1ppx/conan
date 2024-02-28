import os
import time
import nemo.collections.asr as nemo_asr
from datasets import load_dataset

asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-ctc-0.6b")

dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    
if os.path.exists("results.txt"):
    os.remove("results.txt")
f = open("results.txt", "w")


for sample in dataset:
    audio = sample["audio"]

    orig_len = audio["array"]
    orig_len = len(orig_len)/audio["sampling_rate"]

    start = time.time()
    result = asr_model.transcribe([audio])
    end = time.time()

    f.write(f"======= Audio ID: {sample['id']} =======\n") 
    f.write(result[0] + "\n")
    f.write(f"audio length: {orig_len}\n")
    f.write(f"Time elapsed: {end - start}\n")
    f.write(f"Real time factor: {(end - start)/orig_len}\n\n")

f.close()
    