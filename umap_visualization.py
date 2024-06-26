import umap.umap_ as umap
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec

# Assuming loaded_w2v_model is your trained Word2Vec model
loaded_w2v_model = Word2Vec.load('word2vec/w2v_model.model')

# Get vocabulary and embeddings
words = list(loaded_w2v_model.wv.key_to_index.keys())
embeddings = loaded_w2v_model.wv[words]

# UMAP embedding
umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
umap_embeddings = umap_model.fit_transform(embeddings)

# Visualization
plt.figure(figsize=(10, 8))
plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], marker='.', color='b', alpha=0.7)
plt.title('UMAP Visualization of Word2Vec Embeddings')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')

for i, word in enumerate(words):
    plt.annotate(word, xy=(umap_embeddings[i, 0], umap_embeddings[i, 1]), fontsize=8)

plt.show()
