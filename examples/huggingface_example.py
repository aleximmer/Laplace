import logging
import warnings
from collections import UserDict
from collections.abc import MutableMapping

import numpy
import torch
import torch.utils.data as data_utils
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import (
    DataCollatorWithPadding,
    GPT2Config,
    GPT2ForSequenceClassification,
    GPT2Tokenizer,
    PreTrainedTokenizer,
)

from laplace import Laplace

logging.basicConfig(level="ERROR")
warnings.filterwarnings("ignore")

# make deterministic
torch.manual_seed(0)
numpy.random.seed(0)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token_id = tokenizer.eos_token_id

data = [
    {"text": "Today is hot, but I will manage!!!!", "label": 1},
    {"text": "Tomorrow is cold", "label": 0},
    {"text": "Carpe diem", "label": 1},
    {"text": "Tempus fugit", "label": 1},
]
dataset = Dataset.from_list(data)


def tokenize(row):
    return tokenizer(row["text"])


dataset = dataset.map(tokenize, remove_columns=["text"])
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
dataloader = data_utils.DataLoader(
    dataset, batch_size=100, collate_fn=DataCollatorWithPadding(tokenizer)
)

data = next(iter(dataloader))
print(
    f"Huggingface data defaults to UserDict, which is a MutableMapping? {isinstance(data, UserDict)}"
)
for k, v in data.items():
    print(k, v.shape)


class MyGPT2(nn.Module):
    """
    Huggingface LLM wrapper.

    Args:
        tokenizer: The tokenizer used for preprocessing the text data. Needed
            since the model needs to know the padding token id.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer) -> None:
        super().__init__()
        config = GPT2Config.from_pretrained("gpt2")
        config.pad_token_id = tokenizer.pad_token_id
        config.num_labels = 2
        self.hf_model = GPT2ForSequenceClassification.from_pretrained(
            "gpt2", config=config
        )

    def forward(self, data: MutableMapping) -> torch.Tensor:
        """
        Custom forward function. Handles things like moving the
        input tensor to the correct device inside.

        Args:
            data: A dict-like data structure with `input_ids` inside.
                This is the default data structure assumed by Huggingface
                dataloaders.

        Returns:
            logits: An `(batch_size, n_classes)`-sized tensor of logits.
        """
        device = next(self.parameters()).device
        input_ids = data["input_ids"].to(device)
        attn_mask = data["attention_mask"].to(device)
        output_dict = self.hf_model(input_ids=input_ids, attention_mask=attn_mask)
        return output_dict.logits


# Last-layer Laplace
# ------------------
model = MyGPT2(tokenizer)
model.eval()

la = Laplace(
    model,
    likelihood="classification",
    subset_of_weights="last_layer",
    hessian_structure="full",
    # This must reflect faithfully the reduction technique used in the model
    # Otherwise, correctness is not guaranteed
    feature_reduction="pick_last",
)
la.fit(dataloader)
la.optimize_prior_precision()

X_test = next(iter(dataloader))
print(f"[Last-layer Laplace] The predictive tensor is of shape: {la(X_test).shape}.")

del model
del la


# Laplace on a subset of parameters by disabling gradients
# --------------------------------------------------------
model = MyGPT2(tokenizer)
model.eval()

# Enable grad only for the last layer
for p in model.hf_model.parameters():
    p.requires_grad = False

for p in model.hf_model.score.parameters():
    p.requires_grad = True

la = Laplace(
    model,
    likelihood="classification",
    # Will only hit the last-layer since it's the only one that is grad-enabled
    subset_of_weights="all",
    hessian_structure="diag",
)
la.fit(dataloader)
la.optimize_prior_precision()

X_test = next(iter(dataloader))
print(f"[Subnetwork Laplace] The predictive tensor is of shape: {la(X_test).shape}.")

del model
del la


# Laplace on the LoRA-attached LLM
# --------------------------------


def get_lora_model():
    model = MyGPT2(tokenizer)  # Note we don't disable grad
    config = LoraConfig(
        r=4,
        lora_alpha=16,
        target_modules=["c_attn"],  # LoRA on the attention weights
        lora_dropout=0.1,
        bias="none",
    )
    lora_model = get_peft_model(model, config)
    return lora_model


lora_model = get_lora_model()
# Train it as usual

lora_model.eval()

lora_la = Laplace(
    lora_model,
    likelihood="classification",
    subset_of_weights="all",
    hessian_structure="kron",
)
# lora_la.fit(dataloader)

X_test = next(iter(dataloader))
print(f"[LoRA-LLM] The predictive tensor is of shape: {lora_la(X_test).shape}.")
