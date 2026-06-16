#!/usr/bin/env python3
"""Assemble a warm DiLoCo server master from a DDP-pretrained output dir.

The warm-start pretrain (``warm_pretrain.yaml``, plain 4xDDP) writes the model
definition (config.json + model .py + tokenizer) to ``<out>/`` and the trained
weights to ``<out>/checkpoints/checkpoint-N/`` (safetensors when
``save_safetensors`` is on, else ``pytorch_model*.bin``). A DiLoCo server master,
by contrast, wants the trained weights as safetensors at the dir *root* (the same
layout as the pristine master). This script assembles that:

  <warm_master>/  =  <out>/{config.json, *.py, tokenizer*}  (model definition)
                  +  the latest checkpoint's weights as root safetensors

The result loads directly via ``forgather diloco server -o <warm_master>`` and is
what ``experiment.sh``'s ``WARM_MASTER`` / ``PRISTINE_WARM`` knob copies per arm.

    python make_warm_master.py <trained_out_dir> <warm_master_out>
"""

import json
import os
import re
import shutil
import sys

import torch
from safetensors.torch import save_file

_DEF_SUFFIXES = (".py", ".json", ".model")  # model-definition files to carry over
_TOKENIZER_HINT = "tokenizer"


def latest_checkpoint(out_dir):
    ck = os.path.join(out_dir, "checkpoints")
    if not os.path.isdir(ck):
        raise SystemExit(f"no checkpoints/ under {out_dir} — did the pretrain save?")
    cks = [
        (int(m.group(1)), os.path.join(ck, d))
        for d in os.listdir(ck)
        if (m := re.fullmatch(r"checkpoint-(\d+)", d))
    ]
    if not cks:
        raise SystemExit(f"no checkpoint-N dirs under {ck}")
    return max(cks)[1]


def copy_definition(out_dir, dst):
    """Copy config + model .py + tokenizer from the output-dir root (not weights)."""
    for f in os.listdir(out_dir):
        p = os.path.join(out_dir, f)
        if not os.path.isfile(p):
            continue
        # Skip any root-level weight/index files; we write the trained weights below.
        if f.endswith((".safetensors", ".bin")) or "index" in f:
            continue
        if f.endswith(_DEF_SUFFIXES) or _TOKENIZER_HINT in f:
            shutil.copy2(p, os.path.join(dst, f))


def write_root_safetensors(ckpt_dir, dst):
    """Place the trained weights as a single root safetensors shard + index."""
    st = [f for f in os.listdir(ckpt_dir) if f.endswith(".safetensors")]
    if st:
        # Already safetensors (save_safetensors on): copy shard(s) + index verbatim.
        for f in os.listdir(ckpt_dir):
            if f.endswith(".safetensors") or f == "model.safetensors.index.json":
                shutil.copy2(os.path.join(ckpt_dir, f), os.path.join(dst, f))
        return "copied safetensors"
    # Fall back: convert pytorch_model*.bin -> one safetensors shard + index.
    bins = sorted(
        f for f in os.listdir(ckpt_dir) if re.fullmatch(r"pytorch_model.*\.bin", f)
    )
    if not bins:
        raise SystemExit(f"no safetensors or pytorch_model*.bin in {ckpt_dir}")
    sd = {}
    for b in bins:
        sd.update(
            torch.load(os.path.join(ckpt_dir, b), map_location="cpu", weights_only=True)
        )
    sd = {k: v.contiguous() for k, v in sd.items()}
    shard = "model-00001-of-00001.safetensors"
    save_file(sd, os.path.join(dst, shard), metadata={"format": "pt"})
    total = sum(v.numel() * v.element_size() for v in sd.values())
    index = {"metadata": {"total_size": total}, "weight_map": {k: shard for k in sd}}
    with open(os.path.join(dst, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)
    return f"converted {len(bins)} .bin -> {shard} ({len(sd)} tensors)"


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    out_dir, dst = sys.argv[1], sys.argv[2]
    ckpt = latest_checkpoint(out_dir)
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.makedirs(dst)
    copy_definition(out_dir, dst)
    how = write_root_safetensors(ckpt, dst)
    # sanity: must have config.json + tokenizer + a safetensors shard + .py
    need = ["config.json", "model.safetensors.index.json"]
    missing = [n for n in need if not os.path.isfile(os.path.join(dst, n))]
    pys = [f for f in os.listdir(dst) if f.endswith(".py")]
    if missing or not pys:
        raise SystemExit(
            f"assembled master incomplete: missing {missing}, .py={len(pys)}"
        )
    print(f"warm master -> {dst}")
    print(f"  source checkpoint: {ckpt}")
    print(f"  weights: {how}")
    print(f"  files: {sorted(os.listdir(dst))}")


if __name__ == "__main__":
    main()
