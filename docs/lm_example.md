In this example, we apply Laplace to sequence models, where the inputs have 3 axes
(batch size, sequence length, and input dimensionality) and so do the outputs (batch
size, sequence length, and output dimensionality). We start with a model that outputs a
vector per token, trained with a Gaussian likelihood. Then we turn to
[next-token prediction](#next-token-prediction), i.e., language modeling, which is
classification with one set of logits per token.

Let's start with defining a toy model.
Notice that `laplace-torch` requires the model to take a single input, which can be a
dictionary.
See the following [explanation](huggingface_example.md).

!!! important "Intended usage"

    The model is assumed to take batch-first inputs with one output per token, i.e., inputs are
    `(B, L, D)` and outputs are `(B, L, K)`. Each of the `B * L` tokens is an observation
    in the likelihood. The ASDL backends (`AsdlEF`, `AsdlGGN`) rely on this layout. For other
    layouts, e.g., outputs pooled over tokens `(B, K)` or sequence-first layers `(L, B, D)`,
    use the Curvlinops backends (`CurvlinopsEF`, `CurvlinopsGGN`) instead.

    The ASDL backends need the asdl extra: `pip install "laplace-torch[asdl]"`.

```python
# `tensordict` is only needed for this toy dataset: `pip install tensordict`
import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.utils.data import DataLoader

from laplace import Laplace
from laplace.curvature.asdl import AsdlEF, AsdlGGN
from laplace.utils.enums import LinkApprox, PredType

BATCH_SIZE = 4  # B
SEQ_LENGTH = 6  # L
EMBED_DIM = 8  # D
OUTPUT_SIZE = 2  # K
INPUT_KEY = "input"
OUTPUT_KEY = "output"


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        # Batch first to follow the intended usage, otherwise MultiheadAttention will take `(L, B, D)`
        self.attn = nn.MultiheadAttention(EMBED_DIM, num_heads=1, batch_first=True)
        self.final_layer = nn.Linear(EMBED_DIM, OUTPUT_SIZE)

    def forward(self, x):
        x = x[INPUT_KEY].view(-1, SEQ_LENGTH, EMBED_DIM)  # (B, L, D)
        out = self.attn(x, x, x, need_weights=False)[0]  # (B, L, D)
        return self.final_layer(out)  # (B, L, K)
```

Next, we create a toy dataset. You can use any HF datasets or your own, of course.

```python
ds = TensorDict(
    {
        INPUT_KEY: torch.randn((100, SEQ_LENGTH, EMBED_DIM)),
        OUTPUT_KEY: torch.randn((100, SEQ_LENGTH, OUTPUT_SIZE)),
    },
    batch_size=[100],
)  # simulates a dataset
dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: x)

model = Model()
```

Suppose we want to do a last-layer Laplace approximation.
Then the easiest way to do this is by switching off the gradients of all but the final
layer.
Of course we can also do a subnetwork/full Laplace by mix-and-match the gradient requirements.

```python
model = Model()

for mod_name, mod in model.named_modules():
    if mod_name == "final_layer":
        for p in mod.parameters():
            p.requires_grad = True
    else:
        for p in mod.parameters():
            p.requires_grad = False

# GLM
la = Laplace(
    model,
    "regression",
    hessian_structure="full",
    subset_of_weights="all",
    backend=AsdlEF,
    dict_key_x=INPUT_KEY,
    dict_key_y=OUTPUT_KEY,
    enable_backprop=False,  # True => functorch Jacobian, False => ASDL Jacobian
)
la.fit(dl)
```

!!! note

    When `enable_backprop = True`, the Jacobian in the GLM predictive is obtained through
    functorch (`torch.func`). There is currently some memory inefficiency with this approach.
    Also, currently, only the ASDL and Curvlinops backends are supported.

Let's inspect the predictive distributions of this model.
For MAP estimate, this yields a `(B, L, K)` tensor.

```python
data = next(iter(dl))
pred_map = model(data)
```

For the GLM predictive of Laplace:

```python
pred_la_mean, pred_la_var = la(data, pred_type=PredType.GLM)
print(pred_la_mean.shape, pred_la_var.shape)

pred_la_mean, pred_la_var = la(data, pred_type=PredType.GLM, diagonal_output=True)
print(pred_la_mean.shape, pred_la_var.shape)
```

we will get shapes `(B, L, K)` for the mean tensor and `(B, L, K, K)` for the variance
tensor by default.
When `diagonal_output=True`, the variance tensor will instead be `(B, L, K)` as expected.

For the NN/sampled predictive:

```python
# MC
la = Laplace(
    model,
    "regression",
    hessian_structure="diag",
    subset_of_weights="all",
    backend=AsdlGGN,
    dict_key_x=INPUT_KEY,
    dict_key_y=OUTPUT_KEY,
)
la.fit(dl)

pred_la_mean, pred_la_var = la(
    data, pred_type=PredType.NN, link_approx=LinkApprox.MC, n_samples=10
)
print(pred_la_mean.shape, pred_la_var.shape)
```

we get two `(B, L, K)` tensors.

## Next-token prediction

For language modeling, the model returns logits of shape `(B, L, V)`, where `V` is the
vocabulary size, and the targets are the token ids of shape `(B, L)`. This is
classification, and, as before, each of the `B * L` tokens is an observation in the
likelihood. The class axis must be the last one, as is usual for language models, and
the targets must be class indices. Class probabilities as targets are only supported
for logits of shape `(B, C)`.

Below is a toy causal language model. It takes the token ids under `"input_ids"` and
predicts the next token at every position.

```python
from laplace.curvature.curvlinops import CurvlinopsGGN

VOCAB_SIZE = 20  # V
IDS_KEY = "input_ids"
LABELS_KEY = "labels"


class CausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.attn = nn.MultiheadAttention(EMBED_DIM, num_heads=1, batch_first=True)
        self.lm_head = nn.Linear(EMBED_DIM, VOCAB_SIZE)

    def forward(self, x):
        x = self.embedding(x[IDS_KEY])  # (B, L, D)
        L = x.shape[1]
        causal_mask = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1)
        out = self.attn(x, x, x, attn_mask=causal_mask, need_weights=False)[0]
        return self.lm_head(out)  # (B, L, V)
```

The label of each token is the token that follows it:

```python
tokens = torch.randint(VOCAB_SIZE, (100, SEQ_LENGTH + 1))
ds = TensorDict(
    {IDS_KEY: tokens[:, :-1], LABELS_KEY: tokens[:, 1:]},
    batch_size=[100],
)
dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: x)
```

The LM head of a real model has `V * D` weights, so we use a Kronecker-factored Hessian
over the head only:

```python
model = CausalLM()

for mod_name, mod in model.named_modules():
    for p in mod.parameters(recurse=False):
        p.requires_grad = mod_name == "lm_head"

la = Laplace(
    model,
    "classification",
    hessian_structure="kron",
    subset_of_weights="all",
    backend=CurvlinopsGGN,
    dict_key_x=IDS_KEY,
    dict_key_y=LABELS_KEY,
)
la.fit(dl)
la.optimize_prior_precision()
```

Both predictives return the next-token probabilities at every position, i.e., a
`(B, L, V)` tensor:

```python
data = next(iter(dl))

probs = la(data, pred_type=PredType.GLM, link_approx=LinkApprox.PROBIT)
print(probs.shape)

probs = la(data, pred_type=PredType.NN, link_approx=LinkApprox.MC, n_samples=10)
print(probs.shape)
```

```
torch.Size([4, 6, 20])
torch.Size([4, 6, 20])
```

The GLM predictive needs the Jacobian of all `V` logits at every token. For a real
vocabulary, with tens of thousands of tokens, that does not fit in memory, so use the
sampling predictive (`pred_type="nn"`) there.

!!! caution

    Every token is treated as an independent observation in the likelihood, i.e., the
    log-likelihood sums over all `B * L` tokens, not only over the `B` sequences.
    Padding tokens are not masked yet. They count toward the number of observations
    and, for most backends, toward the Hessian.

    Last-layer Laplace (`subset_of_weights="last_layer"`), the BackPack backends, and
    `FunctionalLaplace`, `ELLA`, and `VaLLA` do not support outputs with more than 2 axes
    yet. For regression, the Kronecker-factored Hessian is only available with the
    Curvlinops backends; the ASDL backends raise an error.
