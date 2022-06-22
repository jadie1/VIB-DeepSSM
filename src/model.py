# Jadie Adams
import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
from collections import OrderedDict
from numbers import Number

'''
Input:
	loader_dir - used to infer dimensions
'''
class VIBDeepSSM(nn.Module):
	def __init__(self, loader_dir):
		super(VIBDeepSSM, self).__init__()
		if torch.cuda.is_available():
			self.device = 'cuda:0'
		else:
			self.device = 'cpu'
		loader = torch.load(loader_dir + "validation")
		self.img_dims = loader.dataset.input_images[0].shape[1:]
		self.num_points = loader.dataset.target_points[0].shape[0]
		self.latent_dim = loader.dataset.pcas[0].shape[0]
		self.encoder = StochasticEncoder(self.latent_dim, self.img_dims)
		self.decoder = NonLinearDecoder(self.latent_dim, self.num_points)

	def reparametrize_n(self, mu, std, n=1):
		# Reference: http://pytorch.org/docs/0.3.1/_modules/torch/distributions.html#Distribution.sample_n
		def expand(v):
			if isinstance(v, Number):
				return torch.Tensor([v]).expand(n, 1)
			else:
				return v.expand(n, *v.size())
		if n != 1 :
			mu = expand(mu)
			std = expand(std)
		eps = Variable(cuda(std.data.new(std.size()).normal_(), std.is_cuda))
		return mu + eps * std

	def forward(self, x, num_samples=1):
		z_mean, z_log_var = self.encoder(x)
		# Sample zs
		if num_samples == 0: # test mode
			zs = z_mean
		else:
			zs = self.reparametrize_n(z_mean, z_log_var, num_samples)
		y_mean = self.decoder(zs)
		y_log_var = torch.zeros(y_mean.size()) # placeholder
		if num_samples > 1:
			y_log_var = torch.log(y_mean.var(0))
			y_mean  = y_mean.mean(0)
		return z_mean, z_log_var, y_mean, y_log_var

class ConvolutionalBackbone(nn.Module):
	def __init__(self, img_dims):
		super(ConvolutionalBackbone, self).__init__()
		self.img_dims = img_dims
		# basically using the number of dims and the number of poolings to be used 
		# figure out the size of the last fc layer so that this network is general to 
		# any images
		self.out_fc_dim = np.copy(img_dims)
		padvals = [4, 8, 8]
		for i in range(3):
			self.out_fc_dim[0] = poolOutDim(self.out_fc_dim[0] - padvals[i], 2)
			self.out_fc_dim[1] = poolOutDim(self.out_fc_dim[1] - padvals[i], 2)
			self.out_fc_dim[2] = poolOutDim(self.out_fc_dim[2] - padvals[i], 2)
		
		self.conv = nn.Sequential(OrderedDict([
			('conv1', nn.Conv3d(1, 12, 5)),
			('bn1', nn.BatchNorm3d(12)),
			('relu1', nn.PReLU()),
			('mp1', nn.MaxPool3d(2)),

			('conv2', nn.Conv3d(12, 24, 5)),
			('bn2', nn.BatchNorm3d(24)),
			('relu2', nn.PReLU()),
			('conv3', nn.Conv3d(24, 48, 5)),
			('bn3', nn.BatchNorm3d(48)),
			('relu3', nn.PReLU()),
			('mp2', nn.MaxPool3d(2)),

			('conv4', nn.Conv3d(48, 96, 5)),
			('bn4', nn.BatchNorm3d(96)),
			('relu4', nn.PReLU()),
			('conv5', nn.Conv3d(96, 192, 5)),
			('bn5', nn.BatchNorm3d(192)),
			('relu5', nn.PReLU()),
			('mp3', nn.MaxPool3d(2)),
		]))

		self.fc = nn.Sequential(OrderedDict([
			('flatten', Flatten()),
			('fc1', nn.Linear(self.out_fc_dim[0]*self.out_fc_dim[1]*self.out_fc_dim[2]*192, 384)),
			('relu6', nn.PReLU()),
		]))

	def forward(self, x, offset=False):
		x_conv_features = self.conv(x)
		x_features = self.fc(x_conv_features)
		return x_features

class StochasticEncoder(nn.Module):
	def __init__(self, num_latent, img_dims):
		super(StochasticEncoder, self).__init__()
		self.num_latent = num_latent
		self.img_dims = img_dims
		self.ConvolutionalBackbone = ConvolutionalBackbone(self.img_dims)
		self.pred_z_mean = nn.Linear(384, self.num_latent)
		self.pred_z_log_var = nn.Linear(384, self.num_latent)
	def forward(self, x):
		features = self.ConvolutionalBackbone(x)
		z_mean = self.pred_z_mean(features)
		z_log_var = self.pred_z_log_var(features)
		return z_mean, z_log_var

class NonLinearDecoder(nn.Module):
	def __init__(self, num_latent, num_corr):
		super(NonLinearDecoder, self).__init__()
		self.num_latent = num_latent
		self.num_corr = num_corr
		self.features = nn.Sequential(OrderedDict([			
			('fc1', nn.Linear(self.num_latent, int(self.num_corr/3))),
			('relu1', nn.PReLU()),
			('fc2', nn.Linear(int(self.num_corr/3),self.num_corr)),
			('relu1', nn.PReLU()),
		]))
		self.pred_y_mean = nn.Linear(self.num_corr, self.num_corr*3)
	def forward(self, z):
		features = self.features(z)
		y_mean =  self.pred_y_mean(features)
		y_mean = y_mean.reshape(*y_mean.size()[:-1], self.num_corr, 3)
		return y_mean

def cuda(tensor, is_cuda):
    if is_cuda : return tensor.cuda()
    else : return tensor

class Flatten(nn.Module):
	def forward(self, x):
		return x.view(x.size(0), -1)

def poolOutDim(inDim, kernel_size, padding=0, stride=0, dilation=1):
	if stride == 0:
		stride = kernel_size
	num = inDim + 2*padding - dilation*(kernel_size - 1) - 1
	outDim = int(np.floor(num/stride + 1))
	return outDim