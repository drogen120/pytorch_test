import torch

N, D_in, H, D_out = 32, 1000, 100, 1

x = torch.randn(N, D_in)
y = x.sum(dim=-1, keepdim=True)

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
    )

loss_fn = torch.nn.MSELoss(size_average=True, reduce=True)

learning_rate = 1e-4

print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

for i in range(5000):
    y_predict = model(x)

    loss = loss_fn(y_predict, y)
    print(i, loss.item())

    model.zero_grad()

    loss.backward()

    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad

torch.save(model.state_dict(), "./linear_model.pt")