"""
Sample from a trained model
"""

import json
import os
import pickle
import re
from contextlib import nullcontext
from typing import Generator, Literal

import tiktoken
import torch
from model import GPT, GPTConfig
from tqdm import tqdm

text_type = "abc"
use_validation_prefixes = False
dataset = "irishman"
# dataset = "bach"
tokens_format: Literal["char", "midi"] = "midi"
validation_path = "TODO"
# validation_path = "../data/02_preprocessed/irishman/validation_leadsheet.json"
# validation_path = "../data/03_converted/music21_bach/validation/midi_texts"
n_conditional_measures = 4
# -----------------------------------------------------------------------------
init_from = "resume"  # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = "out"  # ignored if init_from is not 'resume'
start = "$"  # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 100  # number of samples to draw
max_new_tokens = 500  # number of tokens generated in each sample
temperature = 0.8  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200  # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = (
    "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
)  # 'float32' or 'bfloat16' or 'float16'
compile = True  # use PyTorch 2.0 to compile the model to be faster
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
    if tokens_format == "char":
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: "".join([itos[i] for i in l])
    else:
        encode = lambda s: [stoi[c] for c in s.split()]
        decode = lambda l: " ".join([itos[i] for i in l])
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


generator: Generator[tuple[str, str], None, None] | Generator[tuple[int, str], None, None] = (
    (i, start) for i in range(num_samples)
)
if use_validation_prefixes:
    if dataset == "irishman":
        if tokens_format == "char":
            # Load JSON data
            with open(validation_path, "r") as f:
                leadsheets = json.load(f)
            regex = re.compile(r"(:\||::|\s\||\|\])")
            prefixes: list[tuple[int, str]] = []
            for sheet in leadsheets:
                id = sheet.get("id")
                abc_notation = sheet.get("abc notation")
                splitted_notation = regex.split(abc_notation)
                prefixes.append((id, start + "".join(splitted_notation[: n_conditional_measures * 2])))
            generator = (e for e in prefixes)
        elif tokens_format == "midi":
            raise NotImplementedError()
    if dataset == "bach":
        if tokens_format == "midi":
            file_contents: list[tuple[str, str]] = []
            for fname in os.listdir(validation_path):
                if fname.endswith(".txt"):
                    full_path = os.path.join(validation_path, fname)
                    with open(full_path, "r") as f:
                        file_contents.append((fname[:-4], f.read()))
            bach_prefixes: list[tuple[str, str]] = []
            for bach_choral_name, bach_choral_midi_text in file_contents:
                prefix = (
                    start + " " + "|".join(bach_choral_midi_text.split("|")[:n_conditional_measures]).strip() + " |"
                )
                bach_prefixes.append((bach_choral_name, prefix))
            generator = (e for e in bach_prefixes)
        else:
            NotImplementedError()


# run generation
with torch.no_grad():
    with ctx:
        output_dir = os.path.join(out_dir, "samples")
        os.makedirs(output_dir, exist_ok=True)
        for k, prefix in tqdm(generator):
            start_ids = encode(prefix)
            x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            res = decode(y[0].tolist())
            print(res)
            print("-" * 50)
            if tokens_format == "char":
                file_name = os.path.join(output_dir, f"sample_{k}.abc")
            else:
                file_name = os.path.join(output_dir, f"sample_{k}.txt")
            if dataset == "irishman":
                normalized_res = f"X:{k}\n" + res.split("$")[1].strip()
            else:
                normalized_res = res.split("$")[1].strip()
                if not normalized_res.endswith("|"):
                    normalized_res = "|".join(res.split("|")[:-1]).strip() + " |"
            with open(file_name, "w") as f:
                f.write(normalized_res)
