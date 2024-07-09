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
from gensim.models.callbacks import CallbackAny2Vec
import time

class EpochLogger(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0
        self.start_time = time.time()
        self.previous_loss = None
        self.loss_log = []
        self.word_count = 0  # 初始化词计数

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
        
        # 使用 model.corpus_count 作为总词数
        self.word_count = model.corpus_count
        
        # 计算平均损失
        avg_loss = epoch_loss / self.word_count if self.word_count > 0 else 0
        
        self.loss_log.append(avg_loss)
        
        print(f"Epoch {self.epoch} finished. Time elapsed: {elapsed_time:.2f} seconds. Average Loss: {avg_loss:.6f}")

    def plot_loss(self):
        import matplotlib.pyplot as plt
        plt.plot(range(1, len(self.loss_log) + 1), self.loss_log)
        plt.xlabel('Epoch')
        plt.ylabel('Average Loss')
        plt.title('Training Loss over Epochs')
        plt.show()

        
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def preprocess_documents(documents):
    processed_documents = []
    for doc in documents:
        # Split the document into words
        tokens = doc.split()  # Basic split by whitespace
        processed_documents.append(tokens)
    return processed_documents

def load_mongo_and_train(N=20, output_dir="word2vec"):
    # Connect to MongoDB
    client = MongoClient('mongodb://localhost:27017/')
    db = client['MusicBuddyVue']
    collection = db['tracks']
    
    try:
        # Fetch lyric documents from MongoDB
        lyrics_documents = list(collection.find())
        if not lyrics_documents:
            print("No documents found in MongoDB collection.")
            return None, None, None, None
        else:
            print(f"Found {len(lyrics_documents)} documents in MongoDB collection.")
    except Exception as e:
        print(f"Error fetching documents from MongoDB: {e}")
        return None, None, None, None
    
    # Extract lyrics and preprocess them
    lyrics = [doc.get('lyric', '') for doc in lyrics_documents if isinstance(doc.get('lyric', None), str)]
    processed_lyrics = list(tqdm(nomalizor.preprocess_lyrics(lyrics), desc="Preprocessing Lyrics"))
    
    # Ensure processed_lyrics is a list of word lists
    processed_lyrics = preprocess_documents(processed_lyrics)
    
    epoch_logger = EpochLogger()
    
    # Train Word2Vec model
    w2v_model = Word2Vec(processed_lyrics, 
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
    
    # Convert _id to string for document indexing
    doc_ids = [str(doc['_id']) for doc in lyrics_documents]
    song_vectors = [get_song_vector(lyrics, w2v_model) for lyrics in tqdm(processed_lyrics, desc="Generating Song Vectors")]
    # Create a mapping from document ID to vector index
    doc_id_to_index_map = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
    
    # Compute similarity matrix
    w2v_similarity_matrix = cosine_similarity(song_vectors)
    
    # Sort indices to get top N similarities
    top_n_similarities = np.argsort(w2v_similarity_matrix, axis=1)[:, -N-1:-1]
    print(top_n_similarities[0])
    
    # Prepare top similarities in JSON format for MongoDB
    top_similarities_json = []
    for i, doc in enumerate(tqdm(lyrics_documents, desc="Preparing JSON for Top Similarities")):
        top_similar_docs = [
            {
                "track": {"$oid": str(lyrics_documents[idx]['_id'])},
                "value": float(w2v_similarity_matrix[i, idx])  # 获取相似度值
            }
            for idx in top_n_similarities[i]
        ]
        top_similarities_json.append({
            "track": {"$oid": str(doc['_id'])},
            "topsimilar": top_similar_docs
        })
        
        
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
        
    # Save doc_id_to_index_map to JSON file
    with open(os.path.join(output_dir, 'doc_id_to_index_map.json'), 'w') as f:
        json.dump(doc_id_to_index_map, f, indent=2)
    
    # Save to JSON
    with open(os.path.join(output_dir, 'top_similarities.json'), 'w') as f:
        json.dump(top_similarities_json, f, indent=2)

    # Save song_vectors and w2v_similarity_matrix
    np.save(os.path.join(output_dir, 'song_vectors.npy'), song_vectors)

    # Save w2v_model
    w2v_model.save(os.path.join(output_dir, 'w2v_model.model'))

    return song_vectors, w2v_model, top_similarities_json, doc_id_to_index_map

# Generate song vectors
def get_song_vector(lyrics, model):
    vectors = [model.wv[word] for word in lyrics if word in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

def load_saved_data(input_dir="word2vec"):
    # 读取 song_vectors 和 w2v_similarity_matrix
    loaded_song_vectors = np.load(os.path.join(input_dir, 'song_vectors.npy'))

    # 读取 w2v_model
    loaded_w2v_model = Word2Vec.load(os.path.join(input_dir, 'w2v_model.model'))

    # 读取 top_similarities_json
    with open(os.path.join(input_dir, 'top_similarities.json'), 'r') as f:
        loaded_top_similarities_json = json.load(f)
        
    with open(os.path.join(input_dir, 'doc_id_to_index_map.json'), 'r') as f:
        doc_id_to_index_map = json.load(f)

    return loaded_song_vectors, loaded_w2v_model, loaded_top_similarities_json, doc_id_to_index_map

def find_most_similar_words(word, loaded_w2v_model, topn=10):
    if loaded_w2v_model:
        try:
            similar_words = loaded_w2v_model.wv.most_similar(word, topn=topn)
            return similar_words
        except KeyError:
            print(f"Word '{word}' not in vocabulary.")
            return []
    else:
        print("Word2Vec model is not loaded.")
        return []
    
def get_vector_by_doc_id(doc_id, doc_id_to_index_map, song_vectors):
    index = doc_id_to_index_map.get(doc_id)
    if index is None:
        print(f"Document ID {doc_id} not found in the mapping.")
        return None
    return song_vectors[index]

if __name__ == "__main__":
    input_dir = 'word2vec'  # 或者你保存文件的其他目录
    if (os.path.exists(os.path.join(input_dir, 'song_vectors.npy')) and
        os.path.exists(os.path.join(input_dir, 'w2v_model.model')) and
        os.path.exists(os.path.join(input_dir, 'top_similarities.json')) and
        os.path.exists(os.path.join(input_dir, 'doc_id_to_index_map.json'))):
            print("Loading word2Vec results from files...")
            song_vectors, w2v_model, top_similarities_json,doc_id_to_index_map = load_saved_data()
    else:
        print("Loading data from MongoDB and training Word2Vec model...")
        song_vectors, w2v_model, top_similarities_json, doc_id_to_index_map = load_mongo_and_train()
    


    if w2v_model is not None:
        # vocabulary = list(w2v_model.wv.key_to_index.keys())
        
        song_vectors_matrix = np.array(song_vectors)
        # 打印矩阵的形状
        print("song_vectors_matrix shape:", song_vectors_matrix.shape)
        
        doc_id = '6678efa85e93215877cdfce9'  # 替换为你要查询的文档ID
        doc_vector = get_vector_by_doc_id(doc_id, doc_id_to_index_map, song_vectors)
        if doc_vector is not None:
            print(f"Vector representation for document ID {doc_id}: {doc_vector}")
        
        
        
        # 查找相似词
        words_to_find = ['love', 'university', 'cat', 'car', 'night', 'apple', 'bed']
        for word in words_to_find:
            similar_words = find_most_similar_words(word, w2v_model)
            print(f"Words most similar to '{word}': {similar_words}")
            print()
        
        # Print vocabulary
        # print("Vocabulary size:", len(w2v_model.wv))
        # print("Vocabulary:")
        # for word in w2v_model.wv.key_to_index:
        #     print(word)
        

