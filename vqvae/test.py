from model import VQ_VAE
from torch import optim, nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
import json
import torch


def get_params(path):
	with open(path, 'r') as f:
		params = json.load(f)
	f.close()
	return params

def get_dataloader(use_cuda=False):
	pin_memory = use_cuda
	dataset = CIFAR10('./data/', transform=ToTensor())
	dataloader = DataLoader(
		dataset, batch_size=128, shuffle=True, num_workers=4,
		pin_memory=pin_memory
		)
	return dataloader



def main(test_type="forward"):
	use_cuda = torch.cuda.is_available()
	vqvae_params = get_params("./params/vqvae_params.json")
	vqvae_params["encoder_params"] = vqvae_params["params"]
	vqvae_params["decoder_params"] = vqvae_params["params"]
	del vqvae_params["params"]
	net = VQ_VAE(**vqvae_params)
	if use_cuda:
			net = net.cuda()
	dataloader = get_dataloader(use_cuda)

	if test_type == 'forward':
		print(net)

		for i, sample in enumerate(dataloader):
			pic, _ = sample
			if use_cuda:
				pic = pic.cuda()
			pic = Variable(pic)
			re, ze,zq, ze_sg, zq_sg = net(pic)
			print("Reconstruction size is {}".format(re.size()))
			if i > 1:
				break
		print("Forward test finished!")


main()