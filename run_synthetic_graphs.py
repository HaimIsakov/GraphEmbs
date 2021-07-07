import argparse
import os
import numpy as np
from scipy.spatial import distance_matrix
from Nets import *
from multi_scale import mssne_implem
from scipy.cluster import hierarchy
from sklearn import metrics
import re


def map_id_to_airline(airlines_file):
	dict_id_to_airline = {}
	with open(airlines_file, "r") as airlines_file_reader:
		for line in airlines_file_reader:
			line = line.replace("\n", "")
			line = re.split('-|  ', line)
			id = int(line[0])
			airline = line[-1]
			dict_id_to_airline[id - 1] =airline
	return dict_id_to_airline


def tic():
	# Homemade version of matlab tic and toc functions
	import time
	global startTime_for_tictoc
	startTime_for_tictoc = time.time()


def toc():
	import time
	if 'startTime_for_tictoc' in globals():
		print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
	else:
		print("Toc: start time not set")


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-f', dest='f', help='Name of network')	
	parser.add_argument('-opc', dest='opc', help='Type of experiment')	
	return parser.parse_args()


def llf(id):
	return dict_id_to_airline[id]


# run run.py -f ER -opc 0  

args = parse_args()
name = args.f
opc = int(args.opc)

# -------------------------------------------------------------------------------------------------
dict_id_to_airline = map_id_to_airline(os.path.join("airport_data", "nodes.txt"))
if opc == 0:

	# Generating synthetic data
	data = Nets(name)
	data.autoencoder._train()

	# Visualizing graph embeddings
	print("Visualizing graph embeddings...")
	data.visualize_mssne()
	print("Finish graph embeddings and visualizing")
	#data.visualize_tsne()

elif opc==1:

	# Clustering graph embeddings in the embedding space
	print("Clustering graph embeddings...")
	nmi_list = []
	for i in range(0, 10):

		# Generate networks with different node permutations
		data = Nets(name, seed=None)
		data.autoencoder._train()
		nmi_list.append(data.clustering())

	print(str(round(np.mean(nmi_list),2))+" +/- "+str(round(np.std(nmi_list),2)))


elif opc == 2:
	print("Hiererhial clustring of ATN")
	data = Nets(name, 0)
	data.autoencoder._train()
	embds = data.autoencoder.embs
	# sim_mat = data.autoencoder.sim_mat
	# ytdist = np.exp(-1 * metrics.pairwise.euclidean_distances(embds))
	ytdist = metrics.pairwise.euclidean_distances(embds, squared=False)
	# dist = metrics.pairwise.euclidean_distances(embds)
	# dist_sc = (dist - dist.min()) / (dist.max() - dist.min())
	# ytdist = np.square(sim_mat)

	Z = hierarchy.linkage(ytdist, 'single')
	plt.figure()
	dn = hierarchy.dendrogram(Z, orientation='right', leaf_label_func=llf)
	plt.title("single")
	plt.tight_layout()
	plt.show()

	plt.clf()
	Z = hierarchy.linkage(ytdist, 'complete')
	plt.figure()
	dn = hierarchy.dendrogram(Z, orientation='right', leaf_label_func=llf)
	plt.title("complete")
	plt.tight_layout()
	plt.show()

	plt.clf()
	Z = hierarchy.linkage(ytdist, 'average')
	plt.figure()
	dn = hierarchy.dendrogram(Z, orientation='right', leaf_label_func=llf)
	plt.title("average")
	plt.tight_layout()
	plt.show()

	plt.clf()
	Z = hierarchy.linkage(ytdist, 'weighted')
	plt.figure()
	dn = hierarchy.dendrogram(Z, orientation='right', leaf_label_func=llf)
	plt.title("weighted")
	plt.tight_layout()
	plt.show()

	plt.clf()
	Z = hierarchy.linkage(ytdist, 'centroid')
	plt.figure()
	dn = hierarchy.dendrogram(Z, orientation='right', leaf_label_func=llf)
	plt.title("centroid")
	plt.tight_layout()
	plt.show()
