## Full Example: Applying Laplace on a Huggingface LLM model

In this example, we will see how to apply Laplace on a GPT2 Huggingface (HF) model.
Laplace only has lightweight requirements for this; namely that the model's `forward`
method must only take a single dict-like object (`dict`, `UserDict`, or in general,
`collections.abc.MutableMapping`). This is entirely compatible with HF since HF's
data loaders are assumed to emit an object derived from `UserDict`. However, you
need to ensure this yourself --- you need to wrap the standard HF model to conform
to that requirement. Also, you need to e.g. do `torch.to(device)` _inside_ the
said `forward` method.

Let's start with as usual with importing stuff.

```python
from collections.abc import MutableMapping
from collections import UserDict
import numpy
import torch
from torch import nn
import torch.utils.data as data_utils

from laplace import Laplace

import logging
import warnings

logging.basicConfig(level="ERROR")
warnings.filterwarnings("ignore")

from transformers import ( # noqa: E402
    GPT2Config,
    GPT2ForSequenceClassification,
    GPT2Tokenizer,
    DataCollatorWithPadding,
    PreTrainedTokenizer,
)
from peft import LoraConfig, get_peft_model # noqa: E402
from datasets import Dataset # noqa: E402

# make deterministic

torch.manual_seed(0)
numpy.random.seed(0)
```

Next, we create a toy dataset. You can use any HF datasets or your own, of course.

```python
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
```

This is the output:

```
Huggingface data defaults to UserDict, which is a MutableMapping? True
input_ids torch.Size([4, 9])
attention_mask torch.Size([4, 9])
labels torch.Size([4])
```

### Laplace on a subset of an LLM's weights

Now, let's do the main "meat" of this example: Wrapping the HF model into a model that is
compatible with Laplace. Notice that this wrapper just wraps the HF model and nothing else.
Notice also we do `inputs.to(device)` inside `self.forward()`.

```python
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

model = MyGPT2(tokenizer)
```

Now, let's apply Laplace. Let's do a last-layer Laplace first.
Notice that we add
an argument `feature_reduction` there. This is because Huggingface models reduce the
logits and [not the features](https://github.com/huggingface/transformers/blob/a98c41798cf6ed99e1ff17e3792d6e06a2ff2ff3/src/transformers/models/gpt2/modeling_gpt2.py#L1678-L1704).

```python
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
```

Here's the output:

```
[Last-layer Laplace] The predictive tensor is of shape: torch.Size([4, 2]).
```

## Subnetwork Laplace

Also, we can do the same thing by switching off the gradients of all layers except the
top layer. Laplace will automatically only compute the Hessian (and Jacobians) of the
parameters in which `requires_grad` is `True`.

Notice that you can "mix-and-match" this gradient switching. You can do a subnetwork Laplace
easily by doing so!

```python
model.eval()

# Enable grad only for the last layer

for p in model.hf_model.parameters():
    p.requires_grad = False

for p in model.hf_model.score.parameters():
    p.requires_grad = True

la = Laplace(
    model,
    # Will only hit the last-layer since it's the only one that is grad-enabled
    likelihood="classification",
    subset_of_weights="all",
    hessian_structure="diag",
)
la.fit(dataloader)
la.optimize_prior_precision()

X_test = next(iter(dataloader))
print(f"[Subnetwork Laplace] The predictive tensor is of shape: {la(X_test).shape}.")
```

Here are the outputs to validate that Laplace works:

```
[Subnetwork Laplace] The predictive tensor is of shape: torch.Size([4, 2]).
```

## Full Laplace on LoRA parameters only

Of course, you can also apply Laplace on the parameter-efficient fine tuning weights (like LoRA).
To do this, simply extend your LLM with LoRA, using HF's `peft` library, and apply Laplace as
usual. Note that `peft` automatically switches off the non-LoRA weights.

```python
def get_lora_model():
    model = MyGPT2(tokenizer) # Note we don't disable grad
    config = LoraConfig(
        r=4,
        lora_alpha=16,
        target_modules=["c_attn"], # LoRA on the attention weights
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
lora_la.fit(dataloader)

X_test = next(iter(dataloader))
print(f"[LoRA-LLM] The predictive tensor is of shape: {lora_la(X_test).shape}.")
```

Here is the output, as expected:

```
[LoRA-LLM] The predictive tensor is of shape: torch.Size([4, 2]).
```

As a final note, the dict-like input requirement of Laplace is very flexible. It can essentially
be applicable to any tasks and any models. You just need to wrap the said model and make sure
that your data loaders emit dict-like objects, where the input tensors are the dicts' values.

### Caveats

Currently, diagonal EF with the Curvlinops backend is unsupported for dict-based inputs.
This is because we use `torch.func`'s `vmap` to compute the diag-EF, and it only accepts
tensor input in the model's `forward`.
See [this issue](https://github.com/pytorch/functorch/issues/159).
So, if you can write down your Huggingface model's `forward` to accept only a single tensor,
this is much preferable.

For instance, in the case of causal LLM like GPTs, only `input_ids`
tensor is necessary.
Then, any backend and any hessian factorization can be used in this case.

Otherwise, if you must use dict-based inputs, choose the following backends:

- `CurvlinopsGGN` for `hessian_factorization = {"kron", "diag"}`
- `CurvlinopsEF` for `hessian_factorization = {"kron"}`
- `AsdlGGN` for `hessian_factorization = {"kron", "diag"}`
- `AsdlEF` for `hessian_factorization = {"kron", "diag"}`
