# %% [markdown]
# # Huggingface LLMs
#
# In this example, we will see how to apply Laplace on a GPT2 Huggingface (HF) model.
# Laplace only has lightweight requirements for this: the model's `forward` method
# must take a single input. That can be a tensor or, as in this example, a single
# dict-like object (`dict`, `UserDict`, or in general, `collections.abc.MutableMapping`).
# The dict-like input fits HF well, since HF's data loaders are assumed to emit an
# object derived from `UserDict`. However, you need to ensure this yourself --- you need
# to wrap the standard HF model so that it takes the whole batch object. Also, you need
# to move the tensors to the device, e.g. with `tensor.to(device)`, _inside_ the said
# `forward` method.
#
# Let's start with as usual with importing stuff.

# %%
import argparse
import logging
import warnings
from collections import UserDict
from collections.abc import MutableMapping

import numpy
import torch
import torch.utils.data as data_utils
from torch import nn

from laplace import CategoricalLikelihood, Laplace

logging.basicConfig(level="ERROR")
warnings.filterwarnings("ignore")

from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    DataCollatorWithPadding,
    GPT2Config,
    GPT2ForSequenceClassification,
    GPT2Tokenizer,
    PreTrainedTokenizer,
)

# make deterministic

torch.manual_seed(0)
numpy.random.seed(0)

# %% [markdown]
# Next, we create a toy dataset. You can use any HF datasets or your own, of course.

# %%
# Any GPT-2 checkpoint works, e.g. a tiny random one for a quick test run
parser = argparse.ArgumentParser()
parser.add_argument("--model", default="gpt2", help="a base GPT-2 checkpoint")
args, _ = parser.parse_known_args()

tokenizer = GPT2Tokenizer.from_pretrained(args.model)
tokenizer.pad_token_id = tokenizer.eos_token_id
# Pad on the left, so that the last token of every sequence is a real one. Laplace's
# `feature_reduction="pick_last"` below takes the last position.
tokenizer.padding_side = "left"

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

# %% [markdown]
# This is the output:
#
# ```
# Huggingface data defaults to UserDict, which is a MutableMapping? True
# input_ids torch.Size([4, 9])
# attention_mask torch.Size([4, 9])
# labels torch.Size([4])
# ```
#
# ## Laplace on a subset of an LLM's weights
#
# Now, let's do the main "meat" of this example: Wrapping the HF model into a model that is
# compatible with Laplace. Notice that this wrapper just wraps the HF model and nothing else.
# Notice also we do `inputs.to(device)` inside `self.forward()`.


# %%
class MyGPT2(nn.Module):
    """
    Huggingface LLM wrapper.

    Args:
        tokenizer: The tokenizer used for preprocessing the text data. Needed
            since the model needs to know the padding token id.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer) -> None:
        super().__init__()
        config = GPT2Config.from_pretrained(args.model)
        config.pad_token_id = tokenizer.pad_token_id
        config.num_labels = 2
        self.hf_model = GPT2ForSequenceClassification.from_pretrained(
            args.model, config=config
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

        # With left padding, count positions from the first real token
        position_ids = (attn_mask.cumsum(-1) - 1).clamp(min=0)
        output_dict = self.hf_model(
            input_ids=input_ids, attention_mask=attn_mask, position_ids=position_ids
        )
        return output_dict.logits


# %% [markdown]
# Now, let's apply Laplace. Let's do a last-layer Laplace first.
# Notice that we add
# an argument `feature_reduction` there. This is because Huggingface models reduce the
# logits and [not the features](https://github.com/huggingface/transformers/blob/a98c41798cf6ed99e1ff17e3792d6e06a2ff2ff3/src/transformers/models/gpt2/modeling_gpt2.py#L1678-L1704).

# %%
model = MyGPT2(tokenizer)
model.eval()

la = Laplace(
    model,
    likelihood=CategoricalLikelihood(),
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

# %% [markdown]
# Here's the output:
#
# ```
# [Last-layer Laplace] The predictive tensor is of shape: torch.Size([4, 2]).
# ```
#
# ## Subnetwork Laplace
#
# Also, we can do the same thing by switching off the gradients of all layers except the
# top layer. Laplace will automatically only compute the Hessian (and Jacobians) of the
# parameters in which `requires_grad` is `True`.
#
# Notice that you can "mix-and-match" this gradient switching. You can do a subnetwork Laplace
# easily by doing so!

# %%
model.eval()

# Enable grad only for the last layer

for p in model.hf_model.parameters():
    p.requires_grad = False

for p in model.hf_model.score.parameters():
    p.requires_grad = True

la = Laplace(
    model,
    # Will only hit the last-layer since it's the only one that is grad-enabled
    likelihood=CategoricalLikelihood(),
    subset_of_weights="all",
    hessian_structure="diag",
)
la.fit(dataloader)
la.optimize_prior_precision()

X_test = next(iter(dataloader))
print(f"[Subnetwork Laplace] The predictive tensor is of shape: {la(X_test).shape}.")

# %% [markdown]
# Here are the outputs to validate that Laplace works:
#
# ```
# [Subnetwork Laplace] The predictive tensor is of shape: torch.Size([4, 2]).
# ```
#
# ## Full Laplace on LoRA parameters only
#
# Of course, you can also apply Laplace on the parameter-efficient fine tuning weights (like LoRA).
# To do this, simply extend your LLM with LoRA, using HF's `peft` library, and apply Laplace as
# usual. Note that `peft` automatically switches off the non-LoRA weights.


# %%
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
    likelihood=CategoricalLikelihood(),
    subset_of_weights="all",
    hessian_structure="kron",
)
lora_la.fit(dataloader)

X_test = next(iter(dataloader))
print(f"[LoRA-LLM] The predictive tensor is of shape: {lora_la(X_test).shape}.")

# %% [markdown]
# Here is the output, as expected:
#
# ```
# [LoRA-LLM] The predictive tensor is of shape: torch.Size([4, 2]).
# ```
#
# As a final note, the dict-like input requirement of Laplace is very flexible. It can essentially
# be applicable to any tasks and any models. You just need to wrap the said model and make sure
# that your data loaders emit dict-like objects, where the input tensors are the dicts' values.
#
# ## Caveats
#
# `CurvlinopsEF` with `hessian_structure="diag"`, and subnetwork Laplace with `CurvlinopsEF`,
# compute per-example gradients with `torch.func`'s `vmap`. `vmap` does not support
# data-dependent control flow, which Huggingface transformers use, e.g., when building the
# attention mask. So these combinations raise an error for Huggingface models, whether the
# model's `forward` takes a dict or a single tensor.
#
# Instead, choose one of the following backends:
#
# - `CurvlinopsGGN` for `hessian_structure = {"kron", "diag"}`
# - `CurvlinopsEF` for `hessian_structure = {"kron"}`
# - `AsdlGGN` for `hessian_structure = {"kron", "diag"}`
# - `AsdlEF` for `hessian_structure = {"kron", "diag"}`
#
# The ASDL backends need the asdl extra: `pip install "laplace-torch[asdl]"`.
