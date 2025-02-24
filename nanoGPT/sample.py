"""
Sample from a trained model
"""

import json
import os
import pickle
import subprocess
from contextlib import nullcontext

import tiktoken
import torch
from model import GPT, GPTConfig

text_type = "abc"
# -----------------------------------------------------------------------------
init_from = "resume"  # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = "out"  # ignored if init_from is not 'resume'
start = "$ "  # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 100  # number of samples to draw
max_new_tokens = 500  # number of tokens generated in each sample
temperature = 0.8  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200  # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = (
    "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
)  # 'float32' or 'bfloat16' or 'float16'
compile = False  # use PyTorch 2.0 to compile the model to be faster
exec(open("configurator.py").read())  # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == "resume":
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint["model_args"])
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith("gpt2"):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model)  # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if (
    init_from == "resume" and "config" in checkpoint and "dataset" in checkpoint["config"]
):  # older checkpoints might not have these...
    meta_path = os.path.join("data", checkpoint["config"]["dataset"], "meta.pkl")
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta["stoi"], meta["itos"]
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
if start.startswith("FILE:"):
    with open(start[5:], "r", encoding="utf-8") as f:
        start = f.read()
start_ids = encode(start)
x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

# run generation
with torch.no_grad():
    with ctx:
        wav_paths = []
        output_dir = os.path.join(out_dir, "samples")
        os.makedirs(output_dir, exist_ok=True)
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            res = decode(y[0].tolist())
            print(res)
            print("---------------")
            file_name = os.path.join(output_dir, f"sample_{k}.abc")
            midi_name = file_name.rstrip("abc") + "mid"
            wav_name = file_name.rstrip("abc") + "wav"
            normalized_res = f"X:{k}\n" + res.split("$")[1].strip()
            with open(file_name, "w") as f:
                f.write(normalized_res)
            if text_type == "abc":
                # Execute a command and capture its output
                result = subprocess.run(
                    [
                        "abc2midi",
                        file_name,
                        "-o",
                        midi_name,
                    ],
                    capture_output=True,
                    text=True,
                )
                # Print the output from the command
                print(result.stdout)
                result = subprocess.run(
                    [
                        "timidity",
                        midi_name,
                        "-Ow",
                        "-o",
                        wav_name,
                    ],
                    capture_output=True,
                    text=True,
                )
                # Print the output from the command
                print(result.stdout)
                wav_paths.append(os.path.abspath(wav_name))

        aesthetics_folder = os.path.join(out_dir, "audiobox_aesthetics")
        os.makedirs(aesthetics_folder, exist_ok=True)
        input_jsonl_filename = os.path.join(aesthetics_folder, "wav_paths.jsonl")
        output_jsonl_filename = os.path.join(aesthetics_folder, "aesthetics.jsonl")
        # Write the collected WAV file paths to the JSONL file
        with open(input_jsonl_filename, "w") as out_file:
            for path in wav_paths:
                json_line = json.dumps({"path": path})
                out_file.write(json_line + "\n")

        print(f"\nWAV file paths saved to {input_jsonl_filename}")
        with open(output_jsonl_filename, "w") as outfile:
            subprocess.run(["audio-aes", input_jsonl_filename, "--batch-size", "10"], stdout=outfile)
