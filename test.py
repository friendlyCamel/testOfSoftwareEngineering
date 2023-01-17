from Fnet import Net
import torch
import torch.nn as nn


net = Net()
print(net)
input = torch.randn(1,1,32,32)
out = net(input)
print(out)
net.zero_grad()
target = torch.randn(10)
target = target.view(1,-1)
criterion = nn.MSELoss()

loss = criterion(out,target)
print(loss)

print(net.conv1.bias.grad)
loss.backward()
print(net.conv1.bias.grad)
