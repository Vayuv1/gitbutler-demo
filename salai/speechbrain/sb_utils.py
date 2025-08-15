# salai/asr_finetune/speechbrain/sb_utils.py
"""
Utility functions for the SpeechBrain ASR pipeline, including model
creation, tokenizer setup, and LoRA injection.
"""

import json
import torch
import logging
from pathlib import Path
from speechbrain.dataio.encoder import CTCTextEncoder

logger = logging.getLogger(__name__)


def build_asr_model(hparams):
    """Builds the ASR model from hyperparameters."""
    model = hparams["model"]
    return model


def save_model_card(output_dir: Path, hparams: dict):
    """Saves a simple model card to the output directory."""
    card_content = {
        "model_name": hparams.get("model_name", "speechbrain-conformer-ctc"),
        "speechbrain_version": ">=1.0.0",
        "base_model": "ConformerEncoder",
        "dataset": hparams.get("dataset_part"),
        "language": hparams.get("language", "en"),
        "task": hparams.get("task", "asr"),
        "training_params": {
            "epochs": hparams.get("epochs"),
            "learning_rate": hparams.get("lr"),
            "batch_size": hparams.get("batch_size"),
            "grad_accum": hparams.get("grad_accum"),
        },
    }
    card_path = output_dir / "model_card.json"
    with open(card_path, "w") as f:
        json.dump(card_content, f, indent=4)
    logger.info(f"Model card saved to {card_path}")


def create_char_tokenizer():
    """
    Creates a CTCTextEncoder for character-level tokenization.
    """
    # Define the character set as per the project brief
    char_set = [" "] + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789./")

    tokenizer = CTCTextEncoder(symbols=char_set)

    # Add an <unk> token to handle any character that might slip through
    # normalization. This is good practice.
    tokenizer.add_unk()

    return tokenizer


class LoRALinear(torch.nn.Module):
    """ LoRA-adapted Linear layer. """

    def __init__(self, linear_layer, r, alpha, dropout):
        super().__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features

        self.lora_A = torch.nn.Linear(self.in_features, r, bias=False)
        self.lora_B = torch.nn.Linear(r, self.out_features, bias=False)
        self.scaling = alpha / r
        self.dropout = torch.nn.Dropout(p=dropout)

        # Freeze the original layer
        self.original_layer = linear_layer
        self.original_layer.weight.requires_grad = False
        if self.original_layer.bias is not None:
            self.original_layer.bias.requires_grad = False

        # Initialize LoRA weights
        torch.nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        torch.nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        original_output = self.original_layer(x)
        lora_output = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        return original_output + lora_output


def inject_lora(model, r, alpha, dropout):
    """
    Injects LoRA layers into the specified layers of the Conformer model.
    This function specifically targets the Linear layers within the attention
    and feed-forward modules of each Conformer block.
    """
    logger.info(
        f"Injecting LoRA layers with r={r}, alpha={alpha}, dropout={dropout}")

    for layer in model.modules():
        if isinstance(layer, torch.nn.ModuleList):
            for conformer_block in layer:
                # Target Linear layers in MultiheadAttention and PositionwiseFeedForward
                if hasattr(conformer_block, 'mha') and hasattr(
                        conformer_block.mha, 'W_q'):
                    conformer_block.mha.W_q = LoRALinear(
                        conformer_block.mha.W_q, r, alpha, dropout)
                    conformer_block.mha.W_k = LoRALinear(
                        conformer_block.mha.W_k, r, alpha, dropout)
                    conformer_block.mha.W_v = LoRALinear(
                        conformer_block.mha.W_v, r, alpha, dropout)

                if hasattr(conformer_block, 'ffn') and hasattr(
                        conformer_block.ffn, 'ffn'):
                    # The FFN is a Sequential module, we need to find the Linear layers inside
                    for i, ffn_layer in enumerate(conformer_block.ffn.ffn):
                        if isinstance(ffn_layer, torch.nn.Linear):
                            conformer_block.ffn.ffn[i] = LoRALinear(ffn_layer,
                                                                    r, alpha,
                                                                    dropout)

    # Unfreeze only LoRA parameters
    for name, param in model.named_parameters():
        if 'lora_' in name:
            param.requires_grad = True

    logger.info("LoRA injection complete.")
    return model
