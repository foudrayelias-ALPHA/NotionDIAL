"""Precondition screen for the 7B rung: tokenizer gates + day-arithmetic baselines."""
import json
import subprocess
import sys

name = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                      capture_output=True, text=True).stdout
print("GPU:", name.strip(), flush=True)
if "P100" in name:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
                    "torch==2.4.1", "torchvision==0.19.1", "torchaudio==2.4.1",
                    "--index-url", "https://download.pytorch.org/whl/cu121"])

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
FEWSHOT = ("Let's do some day of the week math. Two days after Monday is "
           "Wednesday. Two days after Friday is Sunday. Two days after {} is")
FORMS = [("succ", "If today is {}, then tomorrow is", 1),
         ("after1", "The day after {} is", 1),
         ("after2_zero", "Two days after {} is", 2),
         ("after2_few", FEWSHOT, 2)]

results = {}
for model_name in ["mistralai/Mistral-7B-v0.1", "Qwen/Qwen2.5-7B"]:
    tok = AutoTokenizer.from_pretrained(model_name)
    gate = {}
    for cname, words in [("days", DAYS), ("months", MONTHS)]:
        ids = []
        ok = True
        for w in words:
            t = tok.encode(" " + w, add_special_tokens=False)
            if len(t) != 1:
                ok = False
                break
            ids.append(t[0])
        gate[cname] = ok
    res = {"single_token": gate}
    if not gate["days"]:
        results[model_name] = res
        print(model_name, json.dumps(res), flush=True)
        continue
    ids = [tok.encode(" " + w, add_special_tokens=False)[0] for w in DAYS]
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto").eval()
    for tag, tpl, shift in FORMS:
        correct, margins = 0, []
        with torch.no_grad():
            for i, w in enumerate(DAYS):
                enc = tok(tpl.format(w), return_tensors="pt").input_ids.to(model.device)
                dl = model(input_ids=enc).logits[0, -1][ids].float()
                want = (i + shift) % 7
                margins.append(float(dl[want] - dl[torch.arange(7) != want].max()))
                correct += int(dl.argmax()) == want
        res[tag] = {"acc": f"{correct}/7", "margin": round(sum(margins) / 7, 2)}
        print(model_name, tag, res[tag], flush=True)
    results[model_name] = res
    del model
    torch.cuda.empty_cache()

json.dump(results, open("/kaggle/working/screen.json", "w"), indent=2)
print(json.dumps(results, indent=2), flush=True)
