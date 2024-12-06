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

data = next(iter(dl))  # data[INPUT_KEY].shape = (B, L * D)
pred_map = model(data)  # (B, D)

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

data = next(iter(dl))  # data[INPUT_KEY].shape = (B, L * D)
pred_map = model(data)  # (B, D)
pred_la_mean, pred_la_var = la(
    data, pred_type=PredType.NN, link_approx=LinkApprox.MC, n_samples=10
)

# torch.Size([B, L, K]) torch.Size([B, L, K])
print(pred_la_mean.shape, pred_la_var.shape)
