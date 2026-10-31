# Requires `tensordict` (not a dependency of laplace-torch): `pip install tensordict`
import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.utils.data import DataLoader

from laplace import Laplace
from laplace.curvature.asdl import AsdlEF, AsdlGGN
from laplace.curvature.curvlinops import CurvlinopsGGN
from laplace.utils.enums import LinkApprox, PredType

BATCH_SIZE = 4  # B
SEQ_LENGTH = 6  # L
EMBED_DIM = 8  # D
OUTPUT_SIZE = 2  # K
INPUT_KEY = "input"
OUTPUT_KEY = "output"


# Intended usage: the model is batch-first with one output per token, i.e., inputs are
# `(B, L, D)` and outputs are `(B, L, K)`. Each of the `B * L` tokens is an observation
# in the likelihood. The ASDL backends (`AsdlEF`, `AsdlGGN`) rely on this layout. For other
# layouts, e.g., outputs pooled over tokens `(B, K)` or sequence-first layers `(L, B, D)`,
# use the Curvlinops backends (`CurvlinopsEF`, `CurvlinopsGGN`) instead.
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(EMBED_DIM, num_heads=1, batch_first=True)
        self.final_layer = nn.Linear(EMBED_DIM, OUTPUT_SIZE)

    def forward(self, x):
        x = x[INPUT_KEY].view(-1, SEQ_LENGTH, EMBED_DIM)  # (B, L, D)
        out = self.attn(x, x, x, need_weights=False)[0]  # (B, L, D)
        return self.final_layer(out)  # (B, L, K)


ds = TensorDict(
    {
        INPUT_KEY: torch.randn((100, SEQ_LENGTH, EMBED_DIM)),
        OUTPUT_KEY: torch.randn((100, SEQ_LENGTH, OUTPUT_SIZE)),
    },
    batch_size=[100],
)  # simulates a dataset
dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: x)

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

data = next(iter(dl))  # data[INPUT_KEY].shape = (B, L, D)
pred_map = model(data)  # (B, L, K)

pred_la_mean, pred_la_var = la(data, pred_type=PredType.GLM)
# torch.Size([B, L, K]) torch.Size([B, L, K, K])
print(pred_la_mean.shape, pred_la_var.shape)

pred_la_mean, pred_la_var = la(data, pred_type=PredType.GLM, diagonal_output=True)
# torch.Size([B, L, K]) torch.Size([B, L, K])
print(pred_la_mean.shape, pred_la_var.shape)


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

# torch.Size([B, L, K]) torch.Size([B, L, K])
print(pred_la_mean.shape, pred_la_var.shape)


# Next-token prediction: classification with logits `(B, L, V)` and targets `(B, L)`,
# where `V` is the vocabulary size. As in regression, each of the `B * L` tokens is an
# observation in the likelihood.
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


# The label of each token is the next token
tokens = torch.randint(VOCAB_SIZE, (100, SEQ_LENGTH + 1))
ds = TensorDict(
    {IDS_KEY: tokens[:, :-1], LABELS_KEY: tokens[:, 1:]},
    batch_size=[100],
)
dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: x)

model = CausalLM()

# Laplace over the LM head only
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

data = next(iter(dl))  # data[IDS_KEY].shape = (B, L)

probs = la(data, pred_type=PredType.GLM, link_approx=LinkApprox.PROBIT)
# torch.Size([B, L, V])
print(probs.shape)

probs = la(data, pred_type=PredType.NN, link_approx=LinkApprox.MC, n_samples=10)
# torch.Size([B, L, V])
print(probs.shape)
