# Jadie Adams
import os
import nrrd
import json
import random
import numpy as np
from sklearn.decomposition import PCA
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

'''
Main function - creates train, validation, and test loaders
Input:
	data_dir - directory containing "points" and "images" folders
	loader_dir - directory to save the loaders
	eval_size - size for validation and test sets
	batch_size - size of training batches
'''
def get_loaders(data_dir, loader_dir="loaders", eval_size=100, batch_size=6):
	# Create loader directory
	if not os.path.exists(loader_dir):
		os.makedirs(loader_dir)
	# Split data
	point_files = os.listdir(data_dir + 'points/')
	random.shuffle(point_files)
	train_point_files = point_files[:-2*eval_size]
	val_point_files = point_files[-2*eval_size:-eval_size]
	test_point_files = point_files[-eval_size:]
	# Train set
	print("Creating training set:")
	img_dir = data_dir + 'images/'
	train_points = get_points(train_point_files, point_dir=data_dir + 'points/')
	train_images = get_images(train_point_files, img_dir, loader_dir)
	train_pca, train_embedder = get_pca(train_points, loader_dir)
	train_data = DeepSSMdataset(train_images, train_points, train_pca, train_point_files)
	train_loader = DataLoader(
			train_data,
			batch_size=batch_size,
			shuffle=True,
			num_workers=4,
			pin_memory=torch.cuda.is_available()
		)
	torch.save(train_loader, loader_dir + 'train')
	print("Saved train loader of size " + str(len(train_data)) + ".\n")
	# Val set
	print("Creating validation set:")
	val_points = get_points(val_point_files, point_dir=data_dir + 'points/')
	val_images = get_images(val_point_files, img_dir, loader_dir)
	val_pca, _ = get_pca(val_points, loader_dir, train_embedder)
	val_data = DeepSSMdataset(val_images, val_points, val_pca, val_point_files)
	val_loader = DataLoader(
			val_data,
			batch_size=1,
			shuffle=True,
			num_workers=4,
			pin_memory=torch.cuda.is_available()
		)
	torch.save(val_loader, loader_dir + 'validation')
	print("Saved validation loader of size " + str(len(val_data)) + ".\n")
	# Test set
	print("Creating test set:")
	test_points = get_points(test_point_files, point_dir=data_dir + 'points/')
	test_images = get_images(test_point_files, img_dir, loader_dir)
	test_pca, _ = get_pca(test_points, loader_dir, train_embedder)
	test_data = DeepSSMdataset(test_images, test_points, test_pca, test_point_files)
	test_loader = DataLoader(
			test_data,
			batch_size=1,
			shuffle=True,
			num_workers=4,
			pin_memory=torch.cuda.is_available()
		)
	torch.save(test_loader, loader_dir + 'test')
	print("Saved test loader of size " + str(len(test_data)) + ".\n")
	
'''
Reads .particle files and returns numpy array
'''
def get_points(point_files, point_dir=''):
	points = []
	for point_file in point_files:
		f = open(point_dir + point_file, "r")
		data = []
		for line in f.readlines():
			pts = line.replace(' \n','').split(" ")
			pts = [float(i) for i in pts]
			data.append(pts)
		points.append(data)
	return np.array(points)

'''
Reads .nrrd files and returns numpy array
Whitens/normalizes images
'''
def get_images(point_files, img_dir, loader_dir):
	# get all images
	all_images = []
	for point_file in point_files:
		image_file = img_dir + point_file.replace(".particles", ".nrrd")
		img, header = nrrd.read(image_file)
		all_images.append(img)
	all_images = np.array(all_images)
	# get mean and std
	mean_path = loader_dir + 'mean_img.npy'
	std_path = loader_dir + 'std_img.npy'
	if not os.path.exists(mean_path) or not os.path.exists(std_path):
		mean_image = np.mean(all_images)
		std_image = np.std(all_images)
		np.save(mean_path, mean_image)
		np.save(std_path, std_image)
	else:
		mean_image = np.load(mean_path)
		std_image = np.load(std_path)
	# normalize
	norm_images = []
	for image in all_images:
		norm_images.append([(image-mean_image)/std_image])
	return np.array(norm_images)

'''
Performs PCA on point matrix and returns embedded matrix
Embedding prexerves 99% of the variability
'''
def get_pca(point_matrix, loader_dir, point_embedder=None):
	if not point_embedder:
		point_embedder = PCA_Embbeder(point_matrix)
		embedded_matrix = point_embedder.run_PCA()
		point_embedder.write_PCA(loader_dir + "/PCA_Particle_Info/", "particles")
	else:
		embedded_matrix = point_embedder.getEmbeddedMatrix(point_matrix)
	reconstructed = point_embedder.project(embedded_matrix)
	print("Reconstruction MSE: " + str(np.mean((point_matrix-reconstructed)**2)))
	return embedded_matrix, point_embedder

'''
Class for PCA Embedding
'''
class PCA_Embbeder():
	# overriding abstract methods
	def __init__(self, data_matrix, num_dim=0, percent_variability=0.99):
		self.data_matrix = data_matrix
		self.num_dim=num_dim
		self.percent_variability = percent_variability
	# run PCA on data_matrix for PCA_Embedder
	def run_PCA(self):
		# get covariance matrix (uses compact trick)
		N = self.data_matrix.shape[0]
		data_matrix_2d = self.data_matrix.reshape(self.data_matrix.shape[0], -1).T # flatten data instances and transpose
		mean = np.mean(data_matrix_2d, axis=1)
		centered_data_matrix_2d = (data_matrix_2d.T - mean).T
		trick_cov_matrix  = np.dot(centered_data_matrix_2d.T,centered_data_matrix_2d) * 1.0/np.sqrt(N-1)
		# get eignevectors and eigenvalues
		eigen_values, eigen_vectors = np.linalg.eigh(trick_cov_matrix)
		eigen_vectors = np.dot(centered_data_matrix_2d, eigen_vectors)
		for i in range(N):
			eigen_vectors[:,i] = eigen_vectors[:,i]/np.linalg.norm(eigen_vectors[:,i])
		eigen_values = np.flip(eigen_values)
		eigen_vectors = np.flip(eigen_vectors, 1)
		if self.num_dim == 0:
			cumDst = np.cumsum(eigen_values) / np.sum(eigen_values)
			num_dim = np.where(cumDst > float(self.percent_variability))[0][0] + 1
		W = eigen_vectors[:, :num_dim]
		PCA_scores = np.matmul(centered_data_matrix_2d.T, W)
		print("The PCA modes of particles being retained : " + str(num_dim))
		print("Variablity preserved: " + str(float(cumDst[num_dim-1])))
		self.mean=mean
		self.num_dim = num_dim
		self.PCA_scores = PCA_scores
		self.cumDst = np.cumsum(eigen_values) / np.sum(eigen_values)
		self.eigen_vectors = eigen_vectors
		self.eigen_values = eigen_values
		return PCA_scores
	# write PCA info to files 
	def write_PCA(self, out_dir, suffix):
		if not os.path.exists(out_dir):
			os.makedirs(out_dir)
		np.save(out_dir +  'original_PCA_scores.npy', self.PCA_scores)
		mean = np.mean(self.data_matrix, axis=0)
		np.savetxt(out_dir + 'mean.' + suffix, mean)
		np.savetxt(out_dir + 'eigenvalues.txt', self.eigen_values)
		for i in range(self.data_matrix.shape[0]):
			nm = out_dir + 'pcamode' + str(i) + '.' + suffix
			data = self.eigen_vectors[:, i]
			data = data.reshape(self.data_matrix.shape[1:])
			np.savetxt(nm, data)
		np.savetxt(out_dir + 'cummulative_variance.txt', self.cumDst)
	# returns embedded form of dtat_matrix
	def getEmbeddedMatrix(self, data_matrix):
		N = data_matrix.shape[0]
		data_matrix_2d = data_matrix.reshape(data_matrix.shape[0], -1).T # flatten data instances and transpose
		centered_data_matrix_2d = (data_matrix_2d.T - self.mean).T
		W = self.eigen_vectors[:, :self.num_dim]
		PCA_scores = np.matmul(centered_data_matrix_2d.T, W)
		return PCA_scores
	# projects embbed array into data
	def project(self, PCA_instance):
		W = self.eigen_vectors[:, :self.num_dim].T
		mean = np.mean(self.data_matrix, axis=0)
		data_instance =  np.matmul(PCA_instance, W) + mean.reshape(-1)
		data_instance = data_instance.reshape((-1,self.data_matrix.shape[1], self.data_matrix.shape[2]))
		return data_instance

'''
Class for DeepSSM datasets that works with Pytorch DataLoader
'''
class DeepSSMdataset():
	def __init__(self, input_images, target_points, pcas, filenames):
		self.input_images = torch.FloatTensor(input_images)
		self.target_points = torch.FloatTensor(target_points)
		self.pcas = torch.FloatTensor(pcas)
		self.names = [filename.split(".")[0] for filename in filenames]
	def __getitem__(self, index):
		x = self.input_images[index]
		y = self.target_points[index]
		pca = self.pcas[index]
		name = self.names[index]
		return x, y, pca, name
	def __len__(self):
		return len(self.names)

'''
Creates a JSON files of image names and outlier degrees
'''
def get_image_outlier_degrees(data_dir, loader_dir):
	# Create loader directory
	if not os.path.exists(loader_dir):
		os.makedirs(loader_dir)
	out_file = loader_dir + 'image_outlier_degrees.json'
	img_dir = data_dir + 'images/'
	imgs = []
	print("Getting images...")
	names = []
	for image_file in sorted(os.listdir(img_dir)):
		img, header = nrrd.read(img_dir + image_file)
		imgs.append(img.flatten())
		names.append(image_file.replace(".nrrd", ""))
	imgs = np.array(imgs)
	print("Running PCA...")
	imgs_pca = PCA(0.95)
	imgs_pca.fit(imgs[:800])
	scores = imgs_pca.transform(imgs)
	reconstructed = imgs_pca.inverse_transform(scores)
	print("Getting values...")
	dists = []
	for index in range(imgs.shape[0]):
		dists.append(getMahalanobisDist(scores[index], scores))
	recons = ((imgs - reconstructed)**2).sum(axis=1)
	recons = (recons - np.mean(recons))/np.std(recons)
	dists = (dists - np.mean(dists))/np.std(dists)
	values = np.abs(recons) + np.abs(dists)
	values_dict={}
	for i in range(len(names)):
		values_dict[names[i]] = values[i]
	write_json(values_dict, out_file)
	print("Done.")

'''
Returns Mahalanobis distance
'''
def getMahalanobisDist(score, scores):
	nearest_neighbor_dists = []
	cov = np.cov(scores.T)
	temp = score - np.mean(scores)
	dist = np.dot(np.dot(temp, np.linalg.inv(cov)), temp.T) 
	return dist
'''
Write json file
'''
def write_json(data_dict, out_file):
	with open(out_file, "w") as outfile:
		json.dump(data_dict, outfile, indent=2) 