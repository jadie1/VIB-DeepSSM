import os
import sys
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
sys.path.append('../')
from src import loaders
from src import train
from src import evaluate

def run():
	# # These functions created the provided outlier degrees and data loaders:
	# loaders.get_image_outlier_degrees('data/', 'loaders/')
	# loaders.get_loaders('data/', 'loaders/', eval_size=100, batch_size=6)

	# Set model parameter dictionary
	model_params ={
		"model_dir": 'output/',
		"loader_dir": 'loaders/',
		"learning_rate": 5e-5,
		"epochs": 1000,
		"num_samples": 30,
		"early_stop_patience": 30,
		"loss_params": {
			"initiate_stochastic": 10,
			"complete_stochastic": 20,
			"beta": 0.01
		}
	}

	# Train model
	train.train(model_params)

	# Predict on test set
	prediction_dir = evaluate.predict(model_params, num_samples=100)

	# Analyze results
	evaluate.analyze(prediction_dir, 'loaders/')

if __name__ == '__main__':
	run()