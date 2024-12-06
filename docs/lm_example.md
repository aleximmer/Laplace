In this example, we will see how to apply Laplace for language modeling where the
input tensors have 3 axes (batch size, sequence length, and input dimensionality)
and the output tensors also have 3 axes (batch size, sequence length, and output dimensionality).

Let's start with defining a toy model.
Notice that `laplace-torch` requires the model to take a single input, which can be a
dictionary.
See the following [explanation](huggingface_example.md).

```python
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
        self.attn = nn.MultiheadAttention(EMBED_DIM, num_heads=1)
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

!!! caution

    Currently, `hessian_factorization="kron"` is not supported for inputs/outputs with
    more than 2 axes with the GLM predictive.

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
