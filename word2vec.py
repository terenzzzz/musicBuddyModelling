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
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from gensim.models import KeyedVectors
import gensim.downloader as api
import logging


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
    def __init__(self, mongo_uri='mongodb+srv://terence592592:592592@musicbuddy.grxyfb1.mongodb.net/', db_name='MusicBuddyVue', collection_name='tracks'):
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.w2v_model = None
        self.song_vectors = None
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
                                  vector_size=200,               # Dimensionality of the word vectors
                                  window=12,                      # Maximum distance between the current and predicted word within a sentence
                                  min_count=1,                   # Ignores all words with total frequency lower than this
                                  workers=multiprocessing.cpu_count(),  # Number of CPU cores to use for training
                                  sg=1,                          # Training algorithm: 1 for Skip-Gram; 0 for CBOW
                                  hs=1,                          # If 1, hierarchical softmax will be used for model training
                                  negative=6,                    # Number of negative samples; if > 0, negative sampling will be used
                                  alpha=0.025,                   # The initial learning rate
                                  min_alpha=0.0001,              # Learning rate will linearly drop to min_alpha as training progresses
                                  epochs=60,                    # Number of iterations (epochs) over the corpus
                                  callbacks=[epoch_logger],      # List of callbacks to be called during training (e.g., logging)
                                  compute_loss=True)             # If True, stores loss value during training

        

        
        
        # # 初始化模型,但不立即训练
        # self.w2v_model = Word2Vec(vector_size=300,
        #                           window=10,
        #                           min_count=3,
        #                           workers=multiprocessing.cpu_count(),
        #                           sg=1,
        #                           hs=1,
        #                           negative=6,
        #                           alpha=0.025,
        #                           min_alpha=0.0001,
        #                           compute_loss=True)
        
        # # 构建词汇表
        # self.w2v_model.build_vocab(self.processed_lyrics)
        
        # # 获取词汇表
        # vocab = set(self.w2v_model.wv.key_to_index.keys())
        
        # try:
        #     # 加载预训练向量,只保留词汇表中的词
        #     logging.info("Loading pre-trained vectors...")
        #     pretrained_vectors = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
        #     pretrained_vectors = {word: pretrained_vectors[word] for word in pretrained_vectors.key_to_index.keys() & vocab}
        #     logging.info(f"Loaded {len(pretrained_vectors)} pre-trained word vectors")
            
        #     # 使用预训练向量初始化
        #     self.w2v_model.build_vocab([list(pretrained_vectors.keys())], update=True)
        #     self.w2v_model.intersect_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True, lockf=0.0)
            
        # except FileNotFoundError:
        #     logging.warning("Pre-trained vector file not found. Training from scratch.")
        # except Exception as e:
        #     logging.error(f"An error occurred while loading pre-trained vectors: {str(e)}")
        
        # # 训练模型
        # logging.info("Training Word2Vec model...")
        # self.w2v_model.train(self.processed_lyrics, 
        #                      total_examples=self.w2v_model.corpus_count, 
        #                      epochs=80, 
        #                      callbacks=[epoch_logger])
        
        # logging.info("Model training completed")
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.w2v_model.save(os.path.join(output_dir, 'w2v_model.model'))
        
        epoch_logger.plot_loss(output_dir)
        
        doc_ids = [str(doc['_id']) for doc in self.tracks_documents]
        self.song_vectors = [self.get_song_vector(lyrics) for lyrics in tqdm(processed_lyrics, desc="Generating Song Vectors")]
        self.doc_id_to_index_map = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
        
        with open(os.path.join(output_dir, 'doc_id_to_index_map.json'), 'w') as f:
            json.dump(self.doc_id_to_index_map, f, indent=2)
        
        np.save(os.path.join(output_dir, 'song_vectors.npy'), self.song_vectors)
        
    def load_from_file(self, input_dir="word2vec"):
        self.song_vectors = np.load(os.path.join(input_dir, 'song_vectors.npy'))
        self.w2v_model = Word2Vec.load(os.path.join(input_dir, 'w2v_model.model'))
        with open(os.path.join(input_dir, 'doc_id_to_index_map.json'), 'r') as f:
            self.doc_id_to_index_map = json.load(f)
        
    def get_song_vector(self, lyrics):
        vectors = [self.w2v_model.wv[word] for word in lyrics if word in self.w2v_model.wv]
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(self.w2v_model.vector_size)

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
        
    def get_similarity_between_words(self, word1, word2):
        if self.w2v_model:
            try:
                similarity = self.w2v_model.wv.similarity(word1, word2)
                return similarity
            except KeyError:
                print(f"Word not in vocabulary.")
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
    
    def get_similarity_between_lyrics(self, lyrics_1, lyrics_2):
        vector_1 = self.compute_mean_vector([lyrics_1])
        vector_2 = self.compute_mean_vector([lyrics_2])

        # 计算两个向量的余弦相似度
        cosine_sim = cosine_similarity([vector_1], [vector_2])[0][0]
    
        return cosine_sim
    
    

if __name__ == "__main__":
    w2v_manager = Word2VecManager()
    input_dir = 'word2vec'

    if all(os.path.exists(os.path.join(input_dir, f)) for f in ['song_vectors.npy', 
                                                                'w2v_model.model', 
                                                                'doc_id_to_index_map.json']):
        print("Loading word2Vec results from files...")
        w2v_manager.load_from_file(input_dir)
    else:
        print("Loading data from MongoDB and training Word2Vec model...")
        w2v_manager.load_mongo_and_train()

    if w2v_manager.w2v_model is not None:
        
        # lyric="If he's cheatin', I'm doin' him worse (Like) No Uno, I hit the reverse (Grrah) I ain't trippin', the grip in my purse (Grrah) I don't care 'cause he did it first (Like) If he's cheatin', I'm doin' him worse (Damn) I ain't trippin', I— (I ain't trippin', I—) I ain't trippin', the grip in my purse (Like) I don't care 'cause he did it first"
        # lyric2="Honey, I'm a good man, but I'm a cheatin' man And I'll do all I can, to get a lady's love And I wanna do right, I don't wanna hurt nobody If I slip, well then I'm sorry, yes I am"
        
        # similar_documents = w2v_manager.get_similar_documents_for_lyrics([lyric,lyric2])
        # print(similar_documents)
        
        # song_vectors_matrix = np.array(w2v_manager.song_vectors)
        # print("song_vectors_matrix shape:", song_vectors_matrix.shape)
        
        # doc_id = '6678efa85e93215877cdfce9'
        # doc_vector = w2v_manager.get_vector_by_doc_id(doc_id)
        # if doc_vector is not None:
        #     print(f"Vector representation for document ID {doc_id}: {doc_vector}")
        
            
            
        # Testing
        # words_to_check = ['love', 'car', 'ice', 'night']
        # for word in words_to_check:
        #     similar_words = w2v_manager.find_most_similar_words(word)
        #     print(f"Words most similar to '{word}': {similar_words}")

            
        # from sklearn.manifold import TSNE

        # # 获取前500个最常用的词
        # vocab = list(w2v_manager.w2v_model.wv.index_to_key[:500])
        # word_vectors = w2v_manager.w2v_model.wv[vocab]
        
        # # 使用t-SNE进行降维
        # tsne = TSNE(n_components=2)
        # word_vecs_2d = tsne.fit_transform(word_vectors)
        
        # # 绘制图形
        # plt.figure(figsize=(15, 10))
        # plt.scatter(word_vecs_2d[:, 0], word_vecs_2d[:, 1])
        
        # for i, word in enumerate(vocab):
        #     plt.annotate(word, (word_vecs_2d[i, 0], word_vecs_2d[i, 1]))
        
        # plt.show()
        
        
        pre_trained_model = api.load("word2vec-google-news-300")
        print('========================== Evaluation 1 =============================')
        print(f"词汇表大小: {len(w2v_manager.w2v_model.wv.key_to_index)}")
        
        word_pairs = [
            # 上下文词对
            ('ocean', 'wave'),
            ('sun', 'shine'),
            ('stage', 'performance'),
            ('city', 'street'),
            ('night', 'star'),
        
            # 同义词和近义词
            ('love', 'affection'),
            ('sadness', 'sorrow'),
            ('joy', 'happiness'),
            ('song', 'melody'),
            ('hope', 'optimism'),
        
            # 反义词和对比词对
            ('love', 'hate'),
            ('light', 'darkness'),
            ('joy', 'sorrow'),
            ('freedom', 'captivity'),
            ('peace', 'conflict'),
        
        ]
        
        
        # 测试并打印每对词的相似性
        for word1, word2 in word_pairs:
            similarity = w2v_manager.get_similarity_between_words(word1, word2)
            google_news_similarity = pre_trained_model.similarity(word1, word2)
            print(f"'{word1}' : '{word2}' = {similarity} : {google_news_similarity}")
            
        
        print('========================== Evaluation 3 =============================')
        # 词汇相似度检索
        words_to_check = ['love', 'car', 'ice', 'night', 'gun']
        for word in words_to_check:
            similar_words = w2v_manager.find_most_similar_words(word)
            print(f"Words most similar to '{word}': {similar_words}")
            
            # google_similar_words = pre_trained_model.most_similar(word)
            # print(f"Words most similar for google to '{word}': {google_similar_words}")
            
        
        
        print('========================== Evaluation 4 =============================')
        # 歌词相似度
        # 定义测试歌词
        lyrics_similar_1 = "Underneath the starlit sky, we walk hand in hand, Your smile lights up the night, making my heart expand. Every touch and every glance feels like a sweet embrace, In the timeless dance of love, we find our sacred space."
        lyrics_similar_2 = "In the quiet of the evening, we share our dreams and fears, Your laughter is my melody, calming all my tears. Every whisper in the dark is a promise, soft and true, In this endless night of love, I find my world in you."
        lyrics_contrasting = "In the hustle of the city, where the noise never fades, We navigate through daily trials, in a world that never sways. Every challenge and every setback is a step towards the goal, In the fast pace of life, we strive to find our role."
        
        
        similarity_1_2 = w2v_manager.get_similarity_between_lyrics(lyrics_similar_1,lyrics_similar_2)
        similarity_1_3 = w2v_manager.get_similarity_between_lyrics(lyrics_similar_1,lyrics_contrasting)
        similarity_2_3 = w2v_manager.get_similarity_between_lyrics(lyrics_similar_2,lyrics_contrasting)
        print(similarity_1_2)
        print(similarity_1_3)
        print(similarity_2_3)
        
        



            



        

        


        
        