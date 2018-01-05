from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import torch.nn.functional as F
import torchvision


'''
Define a network.
'''
class VAE(nn.Module):

	def __init__(self):
		super(VAE, self).__init__()
		self.fc1 = nn.Linear(784, 400)
		self.fc2 = nn.Linear(400, 20)
		self.fc3 = nn.Linear(400, 20)
		self.fc4 = nn.Linear(20, 400)
		self.fc5 = nn.Linear(400, 784)

	def encode(self, x):
		x = F.relu(self.fc1(x))
		mu_vec = self.fc2(x)
		std_vec = self.fc3(x)
		return mu_vec, std_vec

	def reparameter(self, mu, std):
		z = Variable(std.data.new(std.size()).normal_())
		z = z * std + mu
		return z

	def decode(self, z):
		z = F.relu(self.fc4(z))
		re = F.sigmoid(self.fc5(z))
		return re

	def forward(self, x):
		mu, std = self.encode(x)
		z = self.reparameter(mu, std)
		re = self.decode(z)
		return mu, std, re


def get_dataloader(label="train", use_cuda=False):
	if label == "train":
		dataset = torchvision.datasets.MNIST(
			"./data", download=True, transform=transforms.ToTensor()
			)
	elif label == "test":
		dataset = torchvision.datasets.MNIST(
			"./data", train=False, download=True, transform=transforms.ToTensor()
			)
	else:
		raise("Invalid label for dataloader")
	if use_cuda:
		return DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4,
						  pin_memory=True)
	else:
		return DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4,
						  pin_memory=False)


def loss_func(mu, std, reconstruct, target):
	mse = F.binary_cross_entropy(reconstruct, target)
	KL = 0.5 * torch.sum(
		std * std + mu * mu - torch.log(std * std) - 1.0, dim=1
		)
	KL = torch.mean(KL) / 784
	return mse + 10.0 * KL


def train(epoch_num, print_interval=1, use_cuda=False):
	net = VAE()
	net.train()
	if use_cuda:
		net = net.cuda()

	use_cuda = torch.cuda.is_available()
	dataloader = get_dataloader(use_cuda=use_cuda)

	optimizer = optim.Adam(net.parameters(), lr=1e-3)
	for epoch in range(epoch_num):
		avg_loss = 0
		total_loss = 0
		for i_batch, sampled_batch in enumerate(dataloader):
			optimizer.zero_grad()
			data, _ = sampled_batch
			if use_cuda:
				data = data.cuda()
			data = Variable(data)
			data = data.view(-1, 784)
			mu, std, re = net(data)
			loss = loss_func(mu, std, re, data)
			loss.backward()
			optimizer.step()
			total_loss += loss.data[0]
			avg_loss = total_loss / (i_batch + 1)

		if (epoch + 1) % print_interval == 0:
			print("Average loss of this epoch is {:.4f}".format(avg_loss))
	print("Train step finished!")
	return net


def generate(net, num, use_cuda=False):
	random_noise = Variable(torch.randn(num, 20))
	if use_cuda:
		random_noise = random_noise.cuda()
	re = net.decode(random_noise).cpu()
	re = re.view(num, 1, 28, 28).data
	torchvision.utils.save_image(re, './results/sample.png')


net = train(15, use_cuda=torch.cuda.is_available())
generate(net, 64, use_cuda=torch.cuda.is_available())