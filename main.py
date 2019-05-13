import torch
torch.backends.cudnn.benchmark = True
import torch.nn as nn
from torchvision import transforms, datasets
import torchvision.utils

import numpy as np

import os
if not os.path.isdir("result"):
	os.makedirs("result")

distance = "W1" # W1 or W2 or hybrid: W1 better looking, W2 faster
if distance == "W1":
	print("Building C++ extension for W1 (requires PyTorch >= 1.0.0)...")
	from torch.utils.cpp_extension import load
	my_ops = load(name="my_ops", sources=["W1_extension/my_ops.cpp", "W1_extension/my_ops_kernel.cu"], verbose=False)
	import my_ops
	print("Building complete")

transform = transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize((0.5,), (0.5,))
			])
dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)
device = torch.device("cuda")

def get_data(dataset):
	full_dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset))
	for y_batch, l_batch in full_dataloader:
		return y_batch
y_t = get_data(dataset)
n = len(y_t)
y_t = y_t.view(n, -1).to(device)

nz = 100
d = 784

batch_size = 64
n_hidden = 512

# Empirical stopping criterion
memory_size = 4000
early_end = (200, 320)

G = nn.Sequential(
	nn.Linear(nz, n_hidden),
	nn.ReLU(True),
	nn.Linear(n_hidden, n_hidden),
	nn.ReLU(True),
	nn.Linear(n_hidden, n_hidden),
	nn.ReLU(True),
	nn.Linear(n_hidden, d),
	nn.Tanh()
).to(device)
def initialize_weights(net):
	for m in net.modules():
		if isinstance(m, nn.Linear):
			m.weight.data.normal_(0, 0.02)
		if hasattr(m, "bias") and m.bias is not None:
			m.bias.data.zero_()
initialize_weights(G)

opt_g = torch.optim.Adam(G.parameters(), lr=1e-4)

psi = torch.zeros(len(y_t), requires_grad=True, device=device)
opt_psi = torch.optim.Adam([psi], lr=1e-1)

z_output = torch.randn(batch_size, nz, device=device)
for it in range(1, 101):
	# OTS
	ot_loss = []
	w1_estimate = []
	memory_p = 0
	memory_z = torch.zeros(memory_size, batch_size, nz)
	memory_y = torch.zeros(memory_size, batch_size, dtype=torch.long)
	for ots_iter in range(1, 20001):
		opt_psi.zero_grad()
		
		z_batch = torch.randn(batch_size, nz, device=device)
		y_fake = G(z_batch)
		
		if distance == "W1":
			score = -my_ops.l1_t(y_fake, y_t) - psi
		elif distance == "W2" or distance == "Hybrid":
			score = torch.matmul(y_fake, y_t.t()) - psi
		phi, hit = torch.max(score, 1)
		
		loss = torch.mean(phi) + torch.mean(psi)
		if distance == "W1" or distance == "Hybrid":
			loss_primal = torch.mean(torch.abs(y_fake - y_t[hit])) * d
		elif distance == "W2":
			loss_primal = torch.mean((y_fake - y_t[hit]) ** 2) * d
		loss_back = -torch.mean(psi[hit]) # equivalent to loss
		loss_back.backward()
		
		opt_psi.step()

		ot_loss.append(loss.item())
		w1_estimate.append(loss_primal.item())

		memory_z[memory_p] = z_batch
		memory_y[memory_p] = hit
		memory_p = (memory_p + 1) % memory_size

		if ots_iter % 1000 == 0:
			# Empirical stopping criterion
			histo = np.histogram(memory_y.reshape(-1), bins=1000, range=(0, n - 1))[0]
			print(it, "OTS", ots_iter, "loss_dual:", "%.2f" % np.mean(ot_loss[-1000:]), "loss_primal:", "%.2f" % np.mean(w1_estimate[-1000:]),
				"histogram", histo.min(), histo.max())
			if histo.min() >= early_end[0] and histo.max() <= early_end[1]:
				break

	# FIT
	g_loss = []
	for fit_iter in range(memory_size):
		opt_g.zero_grad()

		z_batch = memory_z[fit_iter].to(device)
		y_fake = G(z_batch) # G_t(z)
		y0_hit = y_t[memory_y[fit_iter].to(device)] # T(G_{t-1}(z))
		
		if distance == "W1" or distance == "Hybrid":
			loss_g = torch.mean(torch.abs(y0_hit - y_fake)) * d
		elif distance == "W2":
			loss_g = torch.mean((y0_hit - y_fake) ** 2) * d
		
		loss_g.backward()            
		opt_g.step()
		
		g_loss.append(loss_g.item())
	print(it, "FIT loss:", np.mean(g_loss))
	y_output = G(z_output).view(-1, 1, 28, 28)
	y_output = y_output * 0.5 + 0.5
	torchvision.utils.save_image(y_output, "result/fake_%d.png" % it)
		
