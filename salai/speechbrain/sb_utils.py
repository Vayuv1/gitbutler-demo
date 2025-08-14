import os
import json
from typing import Dict, Tuple

import torch
import torch.nn as nn
import speechbrain as sb


class LoRALinear(nn.Module):
    """A simple LoRA wrapper for Linear layers.
    W(x) + scale * B(A(x)), where A is down rank r and B is up rank r.
    Base weight W is frozen when LoRA is enabled.
    """
    def __init__(self, base: nn.Linear, r: int = 32, alpha: int = 64, dropout: float = 0.05):
        super().__init__()
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.A = nn.Linear(base.in_features, r, bias=False)
        self.B = nn.Linear(r, base.out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        # Freeze base
        for p in self.base.parameters():
            p.requires_grad = False
        # Init A,B small
        nn.init.kaiming_uniform_(self.A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        return self.base(x) + self.dropout(self.B(self.A(x))) * self.scaling


def build_model_from_hparams(hparams: Dict, label_encoder=None) -> Tuple[Dict[str, nn.Module], Dict]:
    # Build modules defined in YAML-like dict hparams
    compute_features = hparams["compute_features"]
    normalize = hparams["normalize"]
    specaug = hparams.get("specaug")
    encoder = hparams["encoder"]
    ctc_lin = hparams["ctc_lin"]

    # If label encoder is known, adjust output size of ctc_lin
    if label_encoder is not None and isinstance(ctc_lin, nn.Linear):
        vocab_size = len(label_encoder.ind2lab)
        ctc_lin = nn.Linear(ctc_lin.in_features, vocab_size)

    modules = {
        "compute_features": compute_features,
        "normalize": normalize,
        "specaug": specaug,
        "encoder": encoder,
        "ctc_lin": ctc_lin,
    }
    # Put modules into CUDA if available happens in Brain run_opts
    return modules, hparams


def inject_lora_adapters(modules: Dict[str, nn.Module], r: int, alpha: int, dropout: float):
    def wrap_linear(module: nn.Module):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                setattr(module, name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
            else:
                wrap_linear(child)
    # Target attention and feedforward blocks in encoder
    wrap_linear(modules["encoder"])  # broad approach, safe and simple


def save_model_card(run_dir: str, args, hparams: Dict):
    card = {
        "engine": "speechbrain",
        "model": args.model_name,
        "tokenizer_type": args.tokenizer_type,
        "peft": args.peft,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "hparams_file": args.hparams_file,
    }
    with open(os.path.join(run_dir, "model_card.json"), "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2)