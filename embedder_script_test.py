from utils.seq_evol_embedder import embedder, getSeqEvoLmEmbeddings

if __name__ == '__main__':
	##Input
	batch = ["PROTEIN","SEQWENCE"] #Takes a list of strings/sequences or just one string/sequence
	#batch = "PROTEIN" 

	##Generate embeddings
	#Returns a list of pytorch tensors embedding the sequences of length (3,L,512)
	embeddings = getSeqEvoLmEmbeddings(embedder,batch) 

	print(len(embeddings))
	print(embeddings[0].shape)