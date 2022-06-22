# Jadie Adams
import os
import torch
import time
import numpy as np
from src import model

eps = 1e-8
nll_loss_func = torch.nn.GaussianNLLLoss()

'''
VIB loss with deterministic burn in for faster training
'''
def vib_burnin_loss(z_mu, z_log_var, y_mu, y_log_var, target_y, params):
	current_epoch = params["current_epoch"]
	beta = params['beta']
	init = params['initiate_stochastic']
	comp =params['complete_stochastic']
	y_mse = torch.mean((y_mu - target_y)**2)
	# Deterministic phase
	if current_epoch <= init:
		loss = y_mse
	# Introduce stochastic
	else:
		y_nll = nll_loss_func(y_mu, target_y, torch.exp(y_log_var))
		z_kld = torch.mean(-0.5 * (1 + z_log_var - z_mu.pow(2) - (z_log_var + eps).exp()))
		alpha = min(1, ((current_epoch - init)/(comp - init)))
		loss = (1-alpha)*y_mse + alpha*y_nll + alpha*beta*z_kld
	return loss

'''
Train helper
	Initilaizes all weights using initialization function specified by initf
'''
def weight_init(module, initf):
	def foo(m):
		classname = m.__class__.__name__.lower()
		if isinstance(m, module):
			initf(m.weight)
	return foo

'''
Train helper
	prints and logs values during training
'''
def log_print(logger, values):
	# write to csv
	if not isinstance(values[0], str):
		values = ['%.5f' % i for i in values]
	string_values = [str(i) for i in values]
	log_string = ','.join(string_values)
	logger.write(log_string + '\n')
	# print
	for i in range(len(string_values)):
		while len(string_values[i]) < 15:
			string_values[i] += ' '
	print(' '.join(string_values))

'''
Model training
'''
def train(model_params):
	loss_params = model_params['loss_params']
	model_dir = model_params['model_dir']
	if not os.path.exists(model_dir):
		os.makedirs(model_dir)

	# Load the loaders
	print("Loading train and validation data loaders...")
	loader_dir = model_params['loader_dir']
	train_loader = torch.load(loader_dir + "train")
	val_loader = torch.load(loader_dir + "validation")
	
	# Define the model
	print("Defining model...")
	net = model.VIBDeepSSM(loader_dir)
	device = net.device
	net.to(device)
	# Intialize model weights with xavier
	net.apply(weight_init(module=torch.nn.Conv2d, initf=torch.nn.init.xavier_normal_))	
	net.apply(weight_init(module=torch.nn.Linear, initf=torch.nn.init.xavier_normal_))
	# Initialize z log_var weights to be very small
	torch.nn.init.normal_(net.encoder.pred_z_log_var.weight, mean=0.0, std=1e-6)

	# Define the optimizer
	opt = torch.optim.Adam(net.parameters(), model_params['learning_rate'])
	opt.zero_grad()

	# Initialize logger
	logger = open(model_dir + "train_log.csv", "w+")
	log_header = ["Epoch", "LR", "train_loss", "train_y_mse", "val_y_mse", "Sec"]

	# Intialize training variables
	t0 = time.time()
	best_val_error = np.Inf
	patience_count = 0

	### Train
	print("Beginning training on device = " + device)
	log_print(logger, log_header)
	net.train()
	for e in range(1, model_params['epochs'] + 1):
		torch.cuda.empty_cache()
		train_losses = []
		loss_params['current_epoch'] = e
		if e < loss_params['initiate_stochastic']:
			num_samples = 0
		else:
			num_samples = model_params['num_samples']
		for img, target, _, _ in train_loader:
			opt.zero_grad()
			img = img.to(device)
			target = target.to(device)
			z_mu, z_log_var, y_mu, y_log_var = net(img, num_samples=num_samples)
			loss = vib_burnin_loss(z_mu, z_log_var, y_mu, y_log_var, target, loss_params)
			loss.backward()
			torch.nn.utils.clip_grad_norm_(net.parameters(), 50)
			opt.step()
			train_losses.append(loss.item())
			if torch.isnan(loss):
				print(("Error:Loss is NAN"))
				patience_count = model_params['early_stop_patience']
				break
		train_loss = np.mean(train_losses)
		# Test
		train_corr_mse = test(net, train_loader)
		val_corr_mse = test(net, val_loader)
		log_print(logger, [e, opt.param_groups[0]['lr'], train_loss, train_corr_mse, val_corr_mse, time.time()-t0])

		if e > loss_params['complete_stochastic']+10:
			if val_corr_mse < best_val_error:
				best_val_error = val_corr_mse
				best_epoch = e
				torch.save(net.state_dict(), os.path.join(model_dir, 'best_model.torch'))
				print("Saving.")
				patience_count = 0
			# Check early stoppping criteria
			else:
				patience_count += 1
				if patience_count >= model_params['early_stop_patience']:
					break
		t0 = time.time()
	
	# Save final model
	logger.close()
	torch.save(net.state_dict(), os.path.join(model_dir, 'final_model.torch'))
	print("Training complete, model saved. Best model after epoch " + str(best_epoch) + '\n')

'''
Test on given loader 
'''
def test(net, loader):
	device = net.device
	net.eval()
	corr_mses = []
	for img, target, _, _ in loader:
		img = img.to(device)
		target = target.to(device)
		z_mu, z_log_var, y_mu, y_log_var = net(img, num_samples=0)
		corr_mses.append(torch.mean((y_mu - target)**2).item())
	corr_mse = np.mean(corr_mses)
	return corr_mse