import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.utils.data import DataLoader

from laplace import Laplace
from laplace.curvature.asdl import AsdlGGN
from laplace.utils.enums import PredType

BATCH_SIZE = 4  # B
SEQ_LENGTH = 6  # L
EMBED_DIM = 8  # D
INPUT_KEY = "input"
OUTPUT_KEY = "output"


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(EMBED_DIM, num_heads=1)
        self.final_layer = nn.Linear(EMBED_DIM, 1)

    def forward(self, x):
        x = x[INPUT_KEY].view(-1, SEQ_LENGTH, EMBED_DIM)  # (B, L, D)
        out = self.attn(x, x, x, need_weights=False)[0]  # (B, L, D)
        return self.final_layer(out)  # (B, L, 1)


ds = TensorDict(
    {
        INPUT_KEY: torch.randn((100, SEQ_LENGTH, EMBED_DIM)),
        OUTPUT_KEY: torch.randn((100, SEQ_LENGTH, 1)),
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

la = Laplace(
    model,
    "regression",
    hessian_structure="full",
    subset_of_weights="all",
    backend=AsdlGGN,
    dict_key_x=INPUT_KEY,
    dict_key_y=OUTPUT_KEY,
    enable_backprop=True,
)
la.fit(dl)

data = next(iter(dl))  # data[INPUT_KEY].shape = (B, L * D)
pred_map = model(data)  # (B, D)
# pred_la_mean, pred_la_var = la(
#     data, pred_type=PredType.NN, link_approx=LinkApprox.MC, n_samples=10
# )
pred_la_mean, pred_la_var = la(data, pred_type=PredType.GLM, joint=False)

# torch.Size([4, 6, 1]) torch.Size([4, 6, 1])
print(pred_la_mean.shape, pred_la_var.shape)
