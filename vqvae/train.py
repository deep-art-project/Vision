from data import get_dataloader
from logger import Logger
from model import VQ_VAE
from torch import nn, optim
from torch.autograd import Variable
import glob
import json
import os
import torch


def get_params(filepath):
	with open(filepath, 'r') as f:
		params = json.load(f)
	f.close()
	return params


def save_checkpoint(model, epoch_trained, replace=False):
	filepath = './save/model/ckpt' + str(epoch_trained) + '.pth.tar'
	torch.save({
		"model":model.state_dict(), "epoch_trained":epoch_trained})
	if replace:
		ckpts = glob.glob("./save/model/ckpt*")
		ckpt_nums = [int(x.split('/')[-1].split('.')[0][4:]) for x in ckpts]
		oldest = "./save/model/ckpt" + str(min(ckpt_nums)) + '.pth.tar'
		os.remove(os.path.join(oldest))


def load_checkpoint(net, ckpt_path):
	ckpt = torch.load(ckpt_path)
	net.load(ckpt["model"])
	epoch_trained = ckpt["epoch_trained"]
	return net, epoch_trained


def train():
	'''
	Whether use cuda.
	'''
	use_cuda = torch.cuda.is_available()
	if use_cuda:
		torch.backends.cudnn.benchmark = True

	'''
	Get network, dataloader logger, and optimizer.
	'''
	train_params = get_params("./params/train_params.json")
	data_params = get_params("./params/data_params.json")
	vqvae_params = get_params("./params/vqvae_params.json")
	vqvae_params["encoder_params"] = vqvae_params["params"]
	vqvae_params["decoder_params"] = vqvae_params["params"]
	del vqvae_params["params"]
	net = VQ_VAE(**vqvae_params)

	if train_params["restore_path"] is None:
		epoch_trained = 0
		if use_cuda:
			if train_params["device_ids"]:
				batch_size = data_params["batch_size"]
				num_gpu = len(train_params["device_ids"])
				assert batch_size % num_gpu == 0
				net = nn.DataParallel(
					net, device_ids=train_params["device_ids"]
					)
			net = net.cuda()
	else:
		net, epoch_trained = load_checkpoint(net, train_params["restore_path"])

	dataloader = get_dataloader(**data_params)
	optimizer = optim.Adam(net.parameters(), lr=train_params["lr"])
	logger = Logger(train_params["log_path"])

	'''
	Start training and writing log info.
	'''
	if train_params["loss_func"] == "l1":
		loss_func = nn.L1Loss()
	elif train_params["loss_func"] == "l2":
		loss_func = nn.MSELoss()
	else:
		raise ("Invalid loss function type!")
	if use_cuda:
		loss_func = loss_func.cuda()
	l2_dist = nn.MSELoss()
	beta = net.beta

	for epoch in range(train_params["num_epochs"]):
		avg_loss = 0.0
		total_loss = 0.0
		for i, sample in enumerate(dataloader):
			optimizer.zero_grad()
			pic, _  = sample
			if use_cuda:
				pic = pic.cuda()
			pic = Variable(pic)
			reconstruct, z_e, z_q, z_e_sg, z_q_sg = net(pic)
			reconstruction_loss = loss_func(reconstruct, pic)
			dict_loss = l2_dist(z_q, z_e_sg)
			encoder_loss = l2_dist(z_e, z_q_sg)
			loss = reconstruction_loss + dict_loss + beta * encoder_loss
			loss.backward(retain_graph=True)
			net.backward_encoder()
			optimizer.step()

			total_loss += loss.data[0]
			avg_loss = total_loss / (i + 1)
		logger.scalar_summary("avg_loss", avg_loss, epoch_trained + epoch + 1)
		for tag, value in net.named_parameters():
			logger.histo_summary(
				tag.replace('.', '/'), value.cpu().data.numpy(),
				epoch_trained + epoch + 1
				)
		if (epoch_trained + epoch + 1) % train_params["sample_interval"]:
			logger.image_summary(
				'reconstruct_images',
				reconstruct.cpu().data[:train_params["sample_size"], :].view(
					-1, 32, 32, 3
					).numpy(),
				epoch_trained + epoch + 1
				)
		if (epoch_trained + epoch + 1) % train_params["save_interval"]:
			num_ckpts = len(glob.glob("./save/model/ckpt*"))
			if num_ckpts == train_params["max_ckpts"]:
				save_checkpoint(net, epoch_trained + epoch + 1, True)
			else:
				save_checkpoint(net, epoch_trained + epoch + 1)


train()
