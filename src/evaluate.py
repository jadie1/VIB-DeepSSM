# Jadie Adams
import os
import torch
import json
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt 
from src import model

eps = 1e-8
nll_loss_func = torch.nn.GaussianNLLLoss()

def predict(model_params, num_samples=1):
	model_dir = model_params['model_dir']
	pred_dir = model_dir + 'test_predictions/'
	make_dir(pred_dir)
	mean_dir = pred_dir + 'mean/'
	make_dir(mean_dir)
	var_dir = pred_dir + 'var/'
	make_dir(var_dir)
	target_dir = pred_dir + 'target/'
	make_dir(target_dir)

	# Load the loaders
	print("Loading test data loader...")
	loader_dir = model_params['loader_dir']
	test_loader = torch.load(loader_dir + "test")

	# Load the trained model
	model_path = model_dir + 'best_model.torch'
	print("Loading model " + model_path + "...")
	net = model.VIBDeepSSM(loader_dir)
	net.load_state_dict(torch.load(model_path))
	device = net.device
	net.to(device)
	net.eval()
	
	# Predict
	print("Beginning test prediction on device = " + device)
	MSES = []
	i = 1
	for img, target, _, name in test_loader:
		img = img.to(device)
		target = target.to(device)
		z_mu, z_log_var, y_mu, y_log_var = net(img, num_samples=num_samples)
		MSES.append(torch.mean((y_mu - target)**2).item())
		# Save files
		np.savetxt(mean_dir + name[0] + '.particles', y_mu.detach().cpu().numpy()[0])
		np.savetxt(var_dir + name[0] + '.particles', torch.exp(y_log_var).detach().cpu().numpy()[0])
		np.savetxt(target_dir + name[0] + '.particles', target.detach().cpu().numpy()[0])
		i += 1

	mean_MSE = round(np.mean(MSES),4)
	std_MSE  = round(np.std(MSES), 4)
	print("Mean test MSE = " +str(mean_MSE)+'+-'+str(std_MSE))
	return pred_dir

def make_dir(directory):
	if not os.path.exists(directory):
		os.makedirs(directory)

def analyze(pred_dir, loader_dir):
	out_dir = pred_dir + '../analysis/'
	make_dir(out_dir)
	mean_dir = pred_dir + 'mean/'
	var_dir = pred_dir + 'var/'
	target_dir = pred_dir + 'target/'
	# Get outlier degree dictionary
	with open(loader_dir + "image_outlier_degrees.json") as json_file: 
		out_deg_dict = json.load(json_file)
	out_degs, rrmse, particle_rrmse, particle_entropy, sample_entropy = [], [], [], [], []
	for file in sorted(os.listdir(target_dir)):
		out_degs.append(float(out_deg_dict[file.split(".")[0]]))
		target_particles = np.loadtxt(target_dir + file)
		pred_particles = np.loadtxt(mean_dir + file)
		rrmse.append(RRMSE(target_particles, pred_particles))
		particle_rrmse.append(RRMSEparticles(target_particles, pred_particles))
		particle_var = np.loadtxt(var_dir + file)
		particle_entropy.append(np.sum(particle_var, 1))
		sample_entropy.append(np.sum(particle_var))
	particleRRMSE = np.array(particle_rrmse)
	sampleRRMSE = np.array(rrmse)
	particle_entropy = np.array(particle_entropy)
	# Plots
	corr = scatter_plot(out_degs, "Outlier Degree", sampleRRMSE, "Sample-RRMSE", out_dir)
	corr = scatter_plot(sample_entropy, "Sample-Uncertainty", out_degs, "Outlier Degree", out_dir)
	corr = scatter_plot(sample_entropy, "Sample-Uncertainty", sampleRRMSE, "RRMSE", out_dir)
	corr = scatter_plot(particle_entropy.flatten(), "Particle-Entropy", particleRRMSE.flatten(), "Particle-RRMSE", out_dir)

def RRMSEparticles(true, pred):
	return np.sqrt(np.sum(np.square(true - pred), 1)/3)

def RRMSE(true, pred):
	return np.sqrt(np.sum(np.square(true - pred))/(128*3))

def scatter_plot(x_values, x_label, y_values, y_label, out_dir):
	try:
		corr, _ = pearsonr(x_values, y_values)
	except:
		corr = 0.0
	print("Correlation between "+x_label+" and "+y_label+": "+str(round(corr,5)))
	plt.scatter(x_values, y_values)
	plt.rcParams.update({'figure.figsize':(7,4), 'figure.dpi':100})
	plt.title('\nCorrelation coefficient: ' + str(corr))
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.savefig(out_dir+x_label.replace(" ", "_")+"__VS__"+y_label.replace(" ", "_")+".png")
	plt.clf()
	return corr
