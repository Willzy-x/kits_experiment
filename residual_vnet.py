import Nets.ktsnet as kts
import torch as t
from apex import amp
import torch.optim as optim


model = kts.KTSNet()
model = model
# optimizer = optim.Adam(
#             model.parameters(), lr=1e-4, weight_decay=1e-5)
x = t.randn((2,1,64,64,64))
# model, optimizer = amp.initialize(model.parameters(), optimizer, opt_level="O1")
y = model(x)
print(y.shape)