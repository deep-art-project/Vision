from torch import nn
import torch


def get_res_block(channels, res_kernel_size):
	res_block = []
	for i, kernel_size in enumerate(res_kernel_size):
		res_block.append(nn.ReLU())
		if kernel_size == 1:
			padding = 0
		else:
			padding = 1
		res_block.append(
			nn.Conv2d(channels, channels, kernel_size, padding=padding)
			)
	return nn.Sequential(*res_block)

class VQ_VAE(nn.Module):

	def __init__(self, k_dim, d_dim, beta, encoder_params, decoder_params):
		super(VQ_VAE, self).__init__()
		self.encoder_params = encoder_params
		self.decoder_params = decoder_params
		self.k_dim = k_dim
		self.d_dim = d_dim
		self.beta = beta
		self.dict = nn.Embedding(k_dim, d_dim)
		self.encoder = Encoder(**self.encoder_params)
		self.decoder = Decoder(**self.decoder_params)
		self.init_weights()

	def init_weights(self):
		interval = 1.0 / self.k_dim
		nn.init.uniform(self.dict.weight, -interval, interval)

	def forward(self, x):
		'''
		Encoding original picture x.
		'''
		z_e = self.encoder(x)
		batch_size, channel, height, width = z_e.size()

		'''
		Find the nearest embedding in dictionary.
		'''
		z_e = z_e.view(batch_size, 1, -1)
		W = self.dict.weight
		W_ = torch.unsqueeze(W, 0)
		L2_dist = torch.sum((z_e - W_)**2, dim=2)
		_, index = torch.min(L2_dist, dim=1)
		z_q = W[index]
		z_e_sg = z_e.detach()
		z_q_sg = z_q.detach()

		'''
		Decode z_q to reconstruct the picture.
		'''
		latent = z_q.view(batch_size, height, width, channel)
		latent = latent.permute(0, 3, 1, 2)

		def hook(grad):
			nonlocal z_e
			self.saved_grad = grad
			self.saved_z_e = z_e
		latent.register_hook(hook)
		reconstruct = self.decoder(latent)
		return reconstruct, z_e, z_q, z_e_sg, z_q_sg

	def backward_encoder(self):
		torch.autograd.backward(
			self.saved_z_e, self.saved_grad, retain_graph=True
			)

class Encoder(nn.Module):

	def __init__(self, hidden_channels, out_channels, stride, kernel_size,
				 res_kernel_size, num_convs, num_res_blocks):
		super(Encoder, self).__init__()
		self.hidden_channels = hidden_channels
		self.out_channels = out_channels
		self.stride = stride
		self.kernel_size = kernel_size
		self.res_kernel_size = res_kernel_size
		self.num_convs = num_convs
		self.num_res_blocks = num_res_blocks
		self._init_convs()
		self._init_res_blocks()
		self.out_conv = nn.Conv2d(self.hidden_channels, self.out_channels, 1)
		self.init_weights()

	def init_weights(self):
		pass

	def _init_convs(self):
		for i in range(self.num_convs):
			if i == 0:
				in_channels = 3
			else:
				in_channels = self.hidden_channels
			self.add_module(
				'conv' + str(i+1),
				nn.Conv2d(
					in_channels, self.hidden_channels, self.kernel_size,
					stride=self.stride, padding=1
					)
				)

	def _init_res_blocks(self):
		for i in range(self.num_res_blocks):
			self.add_module(
				'block' + str(i+1), get_res_block(
					self.hidden_channels, self.res_kernel_size
					)
				)

	def forward(self, x):
		for name, module in self.named_children():
			'''
			For convolution layers before residual block.
			'''
			if name[:4] == 'conv':
				x = module(x)
			elif name[:5] == 'block':
				x = module(x) + x
			else:
				out = module(x)
		return out

class Decoder(nn.Module):

	def __init__(self, hidden_channels, out_channels, stride, kernel_size,
				 res_kernel_size, num_convs, num_res_blocks):
		super(Decoder, self).__init__()
		self.hidden_channels = hidden_channels
		self.out_channels = out_channels
		self.stride = stride
		self.kernel_size = kernel_size
		self.res_kernel_size = res_kernel_size
		self.num_convs = num_convs
		self.num_res_blocks = num_res_blocks
		self.out_conv_transpose = nn.ConvTranspose2d(
			out_channels, hidden_channels, 1
			)
		self._init_res_blocks()
		self._init_conv_transpose()
		self.init_weights()

	def init_weights(self):
		pass

	def _init_res_blocks(self):
		for i in range(self.num_res_blocks):
			self.add_module(
				'block' + str(i+1), get_res_block(
					self.hidden_channels, self.res_kernel_size
					)
				)

	def _init_conv_transpose(self):
		for i in range(self.num_convs):
			if i == (self.num_convs - 1):
				out_channels = 3
			else:
				out_channels = self.hidden_channels
			self.add_module(
				'conv_transpose' + str(i+1),
				nn.ConvTranspose2d(
					self.hidden_channels, out_channels, self.kernel_size,
					stride=self.stride, padding=1)
				)

	def forward(self, latent):
		reconstruct = self.out_conv_transpose(latent)
		for name, module in self.named_children():
			if name[:5] == 'block':
				reconstruct = module(reconstruct) + reconstruct
			elif name[:4] == 'conv':
				reconstruct = module(reconstruct)
		return reconstruct