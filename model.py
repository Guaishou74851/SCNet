import torch
from torch import nn
import torch.nn.functional as F
from utils import *

class Module(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(C, C, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(C, C, 3, padding=1),
        )

    def forward(self, x):
        z = x[:, :1]
        z = z - AT(A(z) - y)
        x = torch.cat([z, x[:, 1:]], dim=1)
        return x + self.body(x)

class Net(nn.Module):
    def __init__(self, K, C):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(2, C, 3, padding=1),
            *[Module(C) for _ in range(K)],
            nn.Conv2d(C, 1, 3, padding=1),
        )

    def forward(self, y_, A_, AT_, cs_ratio):
        global y, A, AT
        y, A, AT = y_, A_, AT_
        x = AT(y)
        cs_ratio_map = cs_ratio.view(x.shape[0], 1, 1, 1).expand_as(x)
        x = torch.cat([x, cs_ratio_map], dim=1)
        x = self.body(x)
        return x - AT(A(x) - y)

if __name__ == "__main__":
    device = "cuda"
    model = Net(20, 32).to(device)
    param_cnt = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("#Param.", param_cnt/1e6, "M")
    x = torch.rand(1, 1, 256, 256, device=device)
    cs_ratio = torch.tensor([0.1], device=device)
    U, S, V = torch.linalg.svd(torch.randn(1024, 1024, device=device))
    Phi = (U @ V).reshape(1024, 1, 32, 32)[:103]
    A = lambda z: F.conv2d(z, Phi, stride=32)
    AT = lambda z: F.conv_transpose2d(z, Phi, stride=32)
    y = A(x)
    x_out = model(y, A, AT, cs_ratio)
    print(x_out.shape)