import numpy as np
from pymongo import MongoClient
from sklearn.metrics.pairwise import cosine_similarity
import nomalizor
import json
from gensim.models import Word2Vec
import multiprocessing
from tqdm import tqdm
import os
import logging
import time
import matplotlib.pyplot as plt

class EpochLogger:
    def __init__(self):
        self.epoch = 0
        self.start_time = time.time()
        self.previous_loss = None
        self.loss_log = []
        self.word_count = 0
        
    def on_train_begin(self, model):
        print("Training started")

    def on_epoch_begin(self, model):
        print(f"Epoch {self.epoch + 1} started")
        self.start_time = time.time()

    def on_epoch_end(self, model):
        self.epoch += 1
        elapsed_time = time.time() - self.start_time
        current_loss = model.get_latest_training_loss()
        
        if self.previous_loss is None:
            epoch_loss = current_loss
        else:
            epoch_loss = current_loss - self.previous_loss
        
        self.previous_loss = current_loss
        self.word_count = model.corpus_count
        avg_loss = epoch_loss / self.word_count if self.word_count > 0 else 0
        self.loss_log.append(avg_loss)
        
        print(f"Epoch {self.epoch} finished. Time elapsed: {elapsed_time:.2f} seconds. Average Loss: {avg_loss:.6f}")
    
    def on_train_end(self, model):
        print("Training ended")

    def plot_loss(self):
        plt.plot(range(1, len(self.loss_log) + 1), self.loss_log)
        plt.xlabel('Epoch')
        plt.ylabel('Average Loss')
        plt.title('Training Loss over Epochs')
        plt.show()

class Word2VecManager:
    def __init__(self, mongo_uri='mongodb://localhost:27017/', db_name='MusicBuddyVue', collection_name='tracks'):
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        self.w2v_model = None
        self.song_vectors = None
        self.top_similarities_json = None
        self.doc_id_to_index_map = None

    def preprocess_documents(self, documents):
        processed_documents = []
        for doc in documents:
            tokens = doc.split()
            processed_documents.append(tokens)
        return processed_documents

    def load_mongo_and_train(self, N=20, output_dir="word2vec"):
        try:
            lyrics_documents = list(self.collection.find())
            if not lyrics_documents:
                print("No documents found in MongoDB collection.")
                return
            print(f"Found {len(lyrics_documents)} documents in MongoDB collection.")
        except Exception as e:
            print(f"Error fetching documents from MongoDB: {e}")
            return

        lyrics = [doc.get('lyric', '') for doc in lyrics_documents if isinstance(doc.get('lyric', None), str)]
        processed_lyrics = list(tqdm(nomalizor.preprocess_lyrics(lyrics), desc="Preprocessing Lyrics"))
        processed_lyrics = self.preprocess_documents(processed_lyrics)
        
        epoch_logger = EpochLogger()
        
        self.w2v_model = Word2Vec(processed_lyrics, 
                         vector_size=300, 
                         window=10, 
                         min_count=5, 
                         workers=multiprocessing.cpu_count(),
                         sg=1, 
                         hs=1,
                         negative=8,
                         alpha=0.025,
                         min_alpha=0.0001,
                         epochs=250,
                         callbacks=[epoch_logger],
                         compute_loss=True)
        
        doc_ids = [str(doc['_id']) for doc in lyrics_documents]
        self.song_vectors = [self.get_song_vector(lyrics) for lyrics in tqdm(processed_lyrics, desc="Generating Song Vectors")]
        self.doc_id_to_index_map = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
        
        w2v_similarity_matrix = cosine_similarity(self.song_vectors)
        top_n_similarities = np.argsort(w2v_similarity_matrix, axis=1)[:, -N-1:-1]
        
        self.top_similarities_json = []
        for i, doc in enumerate(tqdm(lyrics_documents, desc="Preparing JSON for Top Similarities")):
            top_similar_docs = [
                {
                    "track": {"$oid": str(lyrics_documents[idx]['_id'])},
                    "value": float(w2v_similarity_matrix[i, idx])
                }
                for idx in top_n_similarities[i]
            ]
            self.top_similarities_json.append({
                "track": {"$oid": str(doc['_id'])},
                "topsimilar": top_similar_docs
            })
        
        os.makedirs(output_dir, exist_ok=True)
        
        with open(os.path.join(output_dir, 'doc_id_to_index_map.json'), 'w') as f:
            json.dump(self.doc_id_to_index_map, f, indent=2)
        
        with open(os.path.join(output_dir, 'top_similarities.json'), 'w') as f:
            json.dump(self.top_similarities_json, f, indent=2)

        np.save(os.path.join(output_dir, 'song_vectors.npy'), self.song_vectors)
        self.w2v_model.save(os.path.join(output_dir, 'w2v_model.model'))

    def get_song_vector(self, lyrics):
        vectors = [self.w2v_model.wv[word] for word in lyrics if word in self.w2v_model.wv]
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(self.w2v_model.vector_size)

    def load_saved_data(self, input_dir="word2vec"):
        self.song_vectors = np.load(os.path.join(input_dir, 'song_vectors.npy'))
        self.w2v_model = Word2Vec.load(os.path.join(input_dir, 'w2v_model.model'))
        with open(os.path.join(input_dir, 'top_similarities.json'), 'r') as f:
            self.top_similarities_json = json.load(f)
        with open(os.path.join(input_dir, 'doc_id_to_index_map.json'), 'r') as f:
            self.doc_id_to_index_map = json.load(f)

    def find_most_similar_words(self, word, topn=10):
        if self.w2v_model:
            try:
                similar_words = self.w2v_model.wv.most_similar(word, topn=topn)
                return similar_words
            except KeyError:
                print(f"Word '{word}' not in vocabulary.")
                return []
        else:
            print("Word2Vec model is not loaded.")
            return []
    
    def get_vector_by_doc_id(self, doc_id):
        index = self.doc_id_to_index_map.get(doc_id)
        if index is None:
            print(f"Document ID {doc_id} not found in the mapping.")
            return None
        return self.song_vectors[index]

if __name__ == "__main__":
    w2v_manager = Word2VecManager()
    input_dir = 'word2vec'

    if all(os.path.exists(os.path.join(input_dir, f)) for f in ['song_vectors.npy', 'w2v_model.model', 'top_similarities.json', 'doc_id_to_index_map.json']):
        print("Loading word2Vec results from files...")
        w2v_manager.load_saved_data()
    else:
        print("Loading data from MongoDB and training Word2Vec model...")
        w2v_manager.load_mongo_and_train()

    if w2v_manager.w2v_model is not None:
        song_vectors_matrix = np.array(w2v_manager.song_vectors)
        print("song_vectors_matrix shape:", song_vectors_matrix.shape)
        
        doc_id = '6678efa85e93215877cdfce9'
        doc_vector = w2v_manager.get_vector_by_doc_id(doc_id)
        if doc_vector is not None:
            print(f"Vector representation for document ID {doc_id}: {doc_vector}")
        
        words_to_find = ['love', 'university', 'cat', 'car', 'night', 'apple', 'bed']
        for word in words_to_find:
            similar_words = w2v_manager.find_most_similar_words(word)
            print(f"Words most similar to '{word}': {similar_words}")
            print()