
import torch 
from torch import nn 

#https://github.com/atong01/ot-icnn-minimal/blob/main/icnn/icnn.py

class ICNN(torch.nn.Module):
    """Input Convex Neural Network"""
    #https://github.com/atong01/ot-icnn-minimal/blob/main/icnn/icnn.py

    def __init__(self, dim=2, dimh=64, num_hidden_layers=4):
        super().__init__()

        Wzs = []
        Wzs.append(nn.Linear(dim, dimh))
        for _ in range(num_hidden_layers - 1):
            Wzs.append(torch.nn.Linear(dimh, dimh, bias=False))
        Wzs.append(torch.nn.Linear(dimh, 1, bias=False))
        self.Wzs = torch.nn.ModuleList(Wzs)

        Wxs = []
        for _ in range(num_hidden_layers - 1):
            Wxs.append(nn.Linear(dim, dimh))
        Wxs.append(nn.Linear(dim, 1, bias=False))
        self.Wxs = torch.nn.ModuleList(Wxs)
        self.act = nn.Softplus()

    def forward(self, x):
        z = self.act(self.Wzs[0](x))
        for Wz, Wx in zip(self.Wzs[1:-1], self.Wxs[:-1]):
            z = self.act(Wz(z) + Wx(x))
        return self.Wzs[-1](z) + self.Wxs[-1](x)

def compute_w2(f, g, x, y, return_loss=False):
    fx = f(x)
    gy = g(y)

    grad_gy = torch.autograd.grad(torch.sum(gy), y, retain_graph=True, create_graph=True)[0]

    f_grad_gy = f(grad_gy)
    y_dot_grad_gy = torch.sum(torch.multiply(y, grad_gy), axis=1, keepdim=True)

    x_squared = torch.sum(torch.pow(x, 2), axis=1, keepdim=True)
    y_squared = torch.sum(torch.pow(y, 2), axis=1, keepdim=True)

    w2 = torch.mean(f_grad_gy - fx - y_dot_grad_gy + 0.5 * x_squared + 0.5 * y_squared)
    if not return_loss:
        return w2
    g_loss = torch.mean(f_grad_gy - y_dot_grad_gy)
    f_loss = torch.mean(fx - f_grad_gy)
    return w2, f_loss, g_loss

def transport(model, x):
    x = x.detach().requires_grad_(True)
    return torch.autograd.grad(torch.sum(model(x)), x)[0]

def to_torch(arr, device="cpu"):
    return torch.tensor(arr, device=device, requires_grad=True)


def train(f, g, x, y, batchsize=1024, reg=0, nepochs=1000, lr=1e-4):

    optimizer_f = torch.optim.Adam(f.parameters(), lr=lr, betas=(0.5, 0.9))
    optimizer_g = torch.optim.Adam(g.parameters(), lr=lr, betas=(0.5, 0.9))

    for epoch in range(1, nepochs + 1):
        for _ in range(10):
            optimizer_g.zero_grad()
            x = x[torch.randperm(x.size(0))[:batchsize]].detach()
            y = y[torch.randperm(y.size(0))[:batchsize]].detach()
            x.requires_grad = True; y.requires_grad = True
            fx = f(x)
            gy = g(y)
            grad_gy = torch.autograd.grad(
                torch.sum(gy), y, retain_graph=True, create_graph=True
            )[0]
            f_grad_gy = f(grad_gy)
            y_dot_grad_gy = torch.sum(torch.mul(y, grad_gy), axis=1, keepdim=True)
            g_loss = torch.mean(f_grad_gy - y_dot_grad_gy)
            if reg > 0:
                g_loss += reg * torch.sum(
                    torch.stack([torch.sum(torch.nn.functional.relu(-w.weight) ** 2) / 2 for w in g.Wzs])
                )
            g_loss.backward()
            optimizer_g.step()

        optimizer_f.zero_grad()
        x = x[torch.randperm(x.size(0))[:batchsize]].detach()
        y = y[torch.randperm(y.size(0))[:batchsize]].detach()
        x.requires_grad = True; y.requires_grad = True
        fx = f(x)
        gy = g(y)
        grad_gy = torch.autograd.grad(torch.sum(gy), y, retain_graph=True, create_graph=True)[
            0
        ]
        f_grad_gy = f(grad_gy)
        f_loss = torch.mean(fx - f_grad_gy)
        if reg > 0:
            f_loss += reg * torch.sum(
                torch.stack([torch.sum(torch.nn.functional.relu(-w.weight) ** 2) / 2 for w in f.Wzs])
            )

        f_loss.backward()
        optimizer_f.step()

        print(f'Epoch {epoch}: F loss {f_loss.item():.4f}, G loss {g_loss.item():.4f}', end='\r')