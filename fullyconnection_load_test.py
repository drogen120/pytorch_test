import torch

N, D_in, H, D_out = 32, 1000, 100, 1
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
    )

model.load_state_dict(torch.load("./linear_model.pt"))
print(model.eval())