import cv2
import numpy as np
import random
from matplotlib import pyplot as plt
import csv
from scipy.spatial import distance
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D

def get_data():
	w_ind = []
	with open('w_inds.csv', 'r') as file:
		reader = csv.reader(file)
		for row in reader:
			w_ind.append(int(float(row[0])))

	b_ind = []
	with open('b_inds.csv', 'r') as file:
		reader = csv.reader(file)
		index = 0
		for row in reader:
			b_ind.append(int(float(row[0])))

	hog_descriptors = []
	with open('hog_descriptors.csv', 'r') as file:
		reader = csv.reader(file)
		index = 0
		for row in reader:
			if index == 300:
				break
			# Disregard first element
			hog_descriptors.append(row[1:])
			index += 1

	haar_features = []
	with open('X_haar.csv', 'r') as file:
		reader = csv.reader(file)
		index = 0
		for row in reader:
			if index == 300:
				break
			haar_features.append(row)
			index += 1

	return np.array(w_ind), np.array(b_ind), np.array(hog_descriptors), np.array(haar_features)

def main():
	w_ind, b_ind, hog_descriptors, haar_features = get_data()

	hog_descriptors_w = hog_descriptors[w_ind].astype('float')
	hog_descriptors_b = hog_descriptors[b_ind].astype('float')

	haar_features_w = haar_features[w_ind].astype('float')
	haar_features_b = haar_features[b_ind].astype('float')

	dim = 4
	hog_w_pca = PCA(hog_descriptors_w, dim)
	hog_b_pca = PCA(hog_descriptors_b, dim)

	haar_w_pca =  PCA(haar_features_w, dim)
	haar_b_pca = PCA(haar_features_b, dim)

	print(get_intra(hog_w_pca[:, [2, 3]]))
	print(get_intra(hog_b_pca[:, [2, 3]]))
	print(get_inter(hog_w_pca[:, [2, 3]], hog_b_pca[:, [2, 3]]))

	print(get_intra(haar_w_pca[:, [2, 3]]))
	print(get_intra(haar_b_pca[:, [2, 3]]))
	print(get_inter(haar_w_pca[:, [2, 3]], haar_b_pca[:, [2, 3]]))

	# [:, [0, 2]]
	# 0.053994386069405605
	# 0.05477181851392262
	# 0.05454052824354563
	#
	# 0.04492811020996204
	# 0.048028726401705336
	# 0.05209840893057147

	# [:, [0, 1]]
	# 0.0541424599471698
	# 0.05582401558637018
	# 0.055104058473479
	#
	# 0.05019013726757001
	# 0.04878298974562132
	# 0.05045063344082904

	# [:, [1, 2]]
	# 0.054416574918539164
	# 0.05536440479669524
	# 0.05518597485414481
	#
	# 0.05375486206668006
	# 0.05643552762903527
	# 0.059794914475024954

	# [:, [2, 3]]
	# 0.05541684026538006
	# 0.05495999288684993
	# 0.05549557196756704
	#
	# 0.052649645767060424
	# 0.05305391840795438
	# 0.05834372212644106


	plot_pca_2_dimen(hog_w_pca, hog_b_pca, haar_w_pca, haar_b_pca)
	plot_pca_3_dimen(hog_w_pca, hog_b_pca, haar_w_pca, haar_b_pca)

def plot_pca_3_dimen(hog_w_pca, hog_b_pca, haar_w_pca, haar_b_pca):
	fig = plt.figure(figsize=plt.figaspect(0.5))
	ax1 = fig.add_subplot(1, 2, 1, projection='3d')
	ax2 = fig.add_subplot(1, 2, 2, projection='3d')

	x_sequence_w = []
	for i in hog_w_pca[:, 0]:
		x_sequence_w.append(i.real)
	x_sequence_b = []
	for i in hog_b_pca[:, 0]:
		x_sequence_b.append(i.real)
	y_sequence_w = []
	for i in hog_w_pca[:, 1]:
		y_sequence_w.append(i.real)
	y_sequence_b = []	
	for i in hog_b_pca[:, 1]:
		y_sequence_b.append(i.real)
	z_sequence_w = []
	for i in hog_w_pca[:, 2]:
		z_sequence_w.append(i.real)
	z_sequence_b = []
	for i in hog_w_pca[:, 2]:
		z_sequence_b.append(i.real)
	ax1.scatter(x_sequence_w, y_sequence_w, z_sequence_w, s=1, color='r');
	ax1.scatter(x_sequence_b, y_sequence_b, z_sequence_b, s=1, color='b');
	ax1.set_title('Hog Descriptors')
	ax1.set_xlabel('PC 1')
	ax1.set_ylabel('PC 2')
	ax1.set_zlabel('PC 3')

	x_sequence_w = []
	for i in haar_w_pca[:, 0]:
		x_sequence_w.append(i.real)
	x_sequence_b = []
	for i in haar_b_pca[:, 0]:
		x_sequence_b.append(i.real)
	y_sequence_w = []
	for i in haar_w_pca[:, 1]:
		y_sequence_w.append(i.real)
	y_sequence_b = []	
	for i in haar_b_pca[:, 1]:
		y_sequence_b.append(i.real)
	z_sequence_w = []
	for i in haar_w_pca[:, 2]:
		z_sequence_w.append(i.real)
	z_sequence_b = []
	for i in haar_w_pca[:, 2]:
		z_sequence_b.append(i.real)

	ax2.scatter(x_sequence_w, y_sequence_w, z_sequence_w, s=1, color='r');
	ax2.scatter(x_sequence_b, y_sequence_b, z_sequence_b, s=1, color='b');
	ax2.set_title('Haar Features')
	ax2.set_xlabel('PC 1')
	ax2.set_ylabel('PC 2')
	ax2.set_zlabel('PC 3')
	plt.legend(['Light-Tonned Skin', 'Dark-Tonned Skin'], loc='lower left')
	plt.show()



def plot_pca_2_dimen(hog_w_pca, hog_b_pca, haar_w_pca, haar_b_pca):
	plt.subplot(1, 2, 1)
	plt.scatter(hog_w_pca[:, 0], hog_w_pca[:, 1], s=0.2, color='r')
	plt.scatter(hog_b_pca[:, 0], hog_b_pca[:, 1], s=0.2, color='b')
	plt.title('Hog Descriptors')
	plt.xlabel('PC 1')
	plt.ylabel('PC 2')
	plt.legend(['Light-Tonned Skin', 'Dark-Tonned Skin'], loc='lower left')

	plt.subplot(1, 2, 2)
	plt.scatter(haar_w_pca[:, 0], haar_w_pca[:, 1], s=0.2, color='r')
	plt.scatter(haar_b_pca[:, 0], haar_b_pca[:, 1], s=0.2, color='b')
	plt.title('Haar Features')
	plt.xlabel('PC 1')
	plt.ylabel('PC 2')
	plt.show()

def get_intra(descriptors):
	ret = []
	num_of_pairs = 0
	for i in range(descriptors.shape[0]):
		for j in range(i + 1, descriptors.shape[0]):
			ret.append(euclidean(descriptors[i], descriptors[j]))
			num_of_pairs += 1
	return np.sum(np.array(ret)) / num_of_pairs

def get_inter(descriptors1, descriptors2):
	ret = []
	num_of_pairs = 0
	for d1 in descriptors1:
		for d2 in descriptors2:
			ret.append(euclidean(d1, d2))
			num_of_pairs += 1
	return np.sum(np.array(ret)) / num_of_pairs

# From COS 360 Assignment 4 Notebook 3
def PCA(X, dims):
    """Perform PCA (principal components analysis)."""
    U, V = np.linalg.eig(np.cov(X.T))
    idx = np.argsort(U)[::-1]
    V = V[:, idx[:dims]]

    return V

# Euclidean distance betwwen two vectors, a and b
def euclidean(a, b):
	return np.linalg.norm(a-b)

if __name__ == '__main__':
    main()












