# libraries
import numpy as np
from bio_embeddings.embed import ProtTransT5BFDEmbedder
import pandas as pd 

embedder = ProtTransT5BFDEmbedder()

ds = pd.read_csv('ds_final_remake_domconds_annots_ss.csv')

sequences_Example = list(ds["pdb-seq"])
num_seq = len(sequences_Example)

i = 0
length = 500
while i < num_seq:
	print("Doing", i, num_seq)
	start = i 
	end = i + length

	sequences = sequences_Example[start:end]

	embeddings = []
	for seq in sequences:
		embeddings.append(np.asarray(embedder.embed(seq)))

	s_no = start / length
	filename = 'embeddings/' + 'T5_' + str(s_no) + '.npz'
	embeddings = np.asarray(embeddings)
	# print(embeddings.shape)
	np.savez_compressed(filename, embeddings)
	i += length

