import numpy as np
from pymongo import MongoClient
from sklearn.metrics.pairwise import cosine_similarity
from preprocessor import Preprocessor
import json
from gensim.models import Word2Vec
import multiprocessing
from tqdm import tqdm
import os
import time
import matplotlib.pyplot as plt

class EpochLogger:
    def __init__(self):
        self.epoch = 0
        self.start_time = time.time()
        self.previous_loss = None
        self.epochs = []
        self.losses = []
        
    def on_train_begin(self, model):
        print("Training started")

    def on_train_end(self, model):
        print("Training finished")
        total_time = time.time() - self.start_time
        print(f"Total training time: {total_time:.2f} seconds")

    def on_epoch_begin(self, model):
        print(f"Epoch {self.epoch + 1} started")

    def on_epoch_end(self, model):
        self.epoch += 1
        current_loss = model.get_latest_training_loss()
        if self.previous_loss is None:
            loss = current_loss
        else:
            loss = current_loss - self.previous_loss
        self.previous_loss = current_loss

        # 归一化 loss（假设每个 epoch 处理相同数量的单词）
        words_per_epoch = sum(model.corpus_count for _ in range(model.epochs))
        normalized_loss = loss / words_per_epoch

        self.epochs.append(self.epoch)
        self.losses.append(normalized_loss)
        print(f'Epoch: {self.epoch}, Normalized Loss: {normalized_loss:.6f}')

    def plot_loss(self, output_dir):
        plt.figure(figsize=(10, 6))
        plt.plot(self.epochs, self.losses)
        plt.title('Word2Vec Training Loss (Normalized)')
        plt.xlabel('Epoch')
        plt.ylabel('Normalized Loss')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'loss_plot.png'))
        plt.close()

class Word2VecManager:
    def __init__(self, mongo_uri='mongodb://localhost:27017/', db_name='MusicBuddyVue', collection_name='tracks'):
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.w2v_model = None
        self.song_vectors = None
        self.top_similarities_json = None
        self.doc_id_to_index_map = None
        self.tracks_documents = None
        self.preprocessor = Preprocessor()
        self.processed_lyrics = None

    def token_documents(self, documents):
        token_documents = []
        for doc in documents:
            tokens = doc.split()
            token_documents.append(tokens)
        print(f"Length of token_documents: {len(token_documents)}")
        return token_documents
    
    def load_preprocessed_data(self):
        client = MongoClient(self.mongo_uri)
        db = client[self.db_name]
        collection = db[self.collection_name]
        
        try:
            tracks_documents = list(collection.find())
            if not tracks_documents:
                print("No documents found in MongoDB collection.")
            else:
                print(f"Found {len(tracks_documents)} documents in MongoDB collection.")
            self.tracks_documents = tracks_documents
        except Exception as e:
            print(f"Error fetching documents from MongoDB: {e}")
            return None
        
        # Extract lyrics and preprocess them
        if os.path.exists('processed_lyrics.txt'):
            # 如果存在本地文件，则读取本地文件
            with open('processed_lyrics.txt', 'r', encoding='utf-8') as f:
                self.processed_lyrics = [line.strip() for line in f.readlines()]
            print(f"Loaded {len(self.processed_lyrics)} PreProcessed documents from local file.")

        else:
            # 不存在则重新处理
            lyrics = [doc.get('lyric', '') for doc in self.tracks_documents if isinstance(doc.get('lyric', None), str)]
            self.processed_lyrics = self.preprocessor.preprocess_lyrics(lyrics)

    def load_mongo_and_train(self, N=20, output_dir="word2vec"):
        self.load_preprocessed_data()
        print('Preprocessed data loaded')
        
        processed_lyrics = self.token_documents(self.processed_lyrics)
        
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
                          epochs=200,
                          callbacks=[epoch_logger],
                          compute_loss=True)
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.w2v_model.save(os.path.join(output_dir, 'w2v_model.model'))
        
        epoch_logger.plot_loss(output_dir)
        
        doc_ids = [str(doc['_id']) for doc in self.tracks_documents]
        self.song_vectors = [self.get_song_vector(lyrics) for lyrics in tqdm(processed_lyrics, desc="Generating Song Vectors")]
        self.doc_id_to_index_map = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
        
        w2v_similarity_matrix = cosine_similarity(self.song_vectors)
        top_n_similarities = np.argsort(w2v_similarity_matrix, axis=1)[:, -N-1:-1]
        
        self.top_similarities_json = []
        for i, doc in enumerate(tqdm(self.tracks_documents, desc="Preparing JSON for Top Similarities")):
            top_similar_docs = [
                {
                    "track": {"$oid": str(self.tracks_documents[idx]['_id'])},
                    "value": float(w2v_similarity_matrix[i, idx])
                }
                for idx in top_n_similarities[i]
            ]
            self.top_similarities_json.append({
                "track": {"$oid": str(doc['_id'])},
                "topsimilar": top_similar_docs
            })
        

        
        with open(os.path.join(output_dir, 'doc_id_to_index_map.json'), 'w') as f:
            json.dump(self.doc_id_to_index_map, f, indent=2)
        
        with open(os.path.join(output_dir, 'top_similarities.json'), 'w') as f:
            json.dump(self.top_similarities_json, f, indent=2)

        np.save(os.path.join(output_dir, 'song_vectors.npy'), self.song_vectors)
        

    def get_song_vector(self, lyrics):
        vectors = [self.w2v_model.wv[word] for word in lyrics if word in self.w2v_model.wv]
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(self.w2v_model.vector_size)

    def load_from_file(self, input_dir="word2vec"):
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
    
    def get_similar_documents_for_lyrics(self, input_lyrics_list, top_n=20):
        # 确保输入是一个列表
        if not isinstance(input_lyrics_list, list):
            input_lyrics_list = [input_lyrics_list]

        # 计算平均向量
        average_vector = self.compute_mean_vector(input_lyrics_list)
    
        # 计算平均向量与所有文档的余弦相似度
        cosine_similarities = cosine_similarity([average_vector], self.song_vectors)[0]
    
        # 获取相似度最高的 top_n 个文档的索引
        top_indices = cosine_similarities.argsort()[-top_n:][::-1]
    
        # 准备结果
        similar_documents = []
        for idx in top_indices:
            doc_id = next(id for id, index in self.doc_id_to_index_map.items() if index == idx)
            similar_documents.append({
                "track": {"$oid": doc_id},
                "similarity": float(cosine_similarities[idx])
            })
    
        return similar_documents

    def compute_mean_vector(self,lyrics):
        # 预处理输入的歌词列表
        processed_inputs = self.preprocessor.preprocess_lyrics(lyrics)
        
        # 计算每首歌的 W2V 向量
        input_vectors = []
        for lyrics in processed_inputs:
            tokens = lyrics.split()
            song_vector = self.get_song_vector(tokens)
            input_vectors.append(song_vector)
        
        # 计算平均向量
        average_vector = np.mean(input_vectors, axis=0)
        
        return average_vector
    

if __name__ == "__main__":
    w2v_manager = Word2VecManager()
    input_dir = 'word2vec'

    if all(os.path.exists(os.path.join(input_dir, f)) for f in ['song_vectors.npy', 'w2v_model.model', 'top_similarities.json', 'doc_id_to_index_map.json']):
        print("Loading word2Vec results from files...")
        w2v_manager.load_from_file(input_dir)
    else:
        print("Loading data from MongoDB and training Word2Vec model...")
        w2v_manager.load_mongo_and_train()

    if w2v_manager.w2v_model is not None:
        
        lyric="If he's cheatin', I'm doin' him worse (Like) No Uno, I hit the reverse (Grrah) I ain't trippin', the grip in my purse (Grrah) I don't care 'cause he did it first (Like) If he's cheatin', I'm doin' him worse (Damn) I ain't trippin', I— (I ain't trippin', I—) I ain't trippin', the grip in my purse (Like) I don't care 'cause he did it first"
        lyric2="Honey, I'm a good man, but I'm a cheatin' man And I'll do all I can, to get a lady's love And I wanna do right, I don't wanna hurt nobody If I slip, well then I'm sorry, yes I am"
        
        similar_documents = w2v_manager.get_similar_documents_for_lyrics([lyric,lyric2])
        print(similar_documents)
        
        # song_vectors_matrix = np.array(w2v_manager.song_vectors)
        # print("song_vectors_matrix shape:", song_vectors_matrix.shape)
        
        # doc_id = '6678efa85e93215877cdfce9'
        # doc_vector = w2v_manager.get_vector_by_doc_id(doc_id)
        # if doc_vector is not None:
        #     print(f"Vector representation for document ID {doc_id}: {doc_vector}")
        
        # words_to_find = ['love', 'university', 'cat', 'car', 'night', 'apple', 'bed']
        # for word in words_to_find:
        #     similar_words = w2v_manager.find_most_similar_words(word)
        #     print(f"Words most similar to '{word}': {similar_words}")
        #     print()