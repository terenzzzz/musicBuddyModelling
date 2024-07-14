from pymongo import MongoClient
from preprocessor import Preprocessor
from tqdm import tqdm
import gensim
from gensim import corpora
from nltk.tokenize import word_tokenize
from pprint import pprint
import os
import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
import matplotlib.pyplot as plt
import numpy as np


class LDAModelManager:
    def __init__(self, mongo_uri='mongodb://localhost:27017/', db_name='MusicBuddyVue', collection_name='tracks'):
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.dictionary = None
        self.corpus = None
        self.lda_model = None
        self.texts = None
        self.doc_id_to_index_map = None
        self.tracks_documents = None
        self.preprocessor = Preprocessor()
        self.processed_lyrics = None
        
         
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


    def load_mongo_and_train(self, num_topics=50, output_dir='lda'):
        self.load_preprocessed_data()
        print('Preprocessed data loaded')

        self.texts = [word_tokenize(lyric) for lyric in self.processed_lyrics]
        
        self.dictionary = corpora.Dictionary(self.texts)
        self.corpus = [self.dictionary.doc2bow(text) for text in self.texts]

        print(f"Original document count: {len(self.tracks_documents)}")
        print(f"Documents after preprocessing: {len(self.processed_lyrics)}")
        print(f"Documents after tokenization: {len(self.texts)}")
        print(f"Corpus size: {len(self.corpus)}")
        
        doc_ids = [str(doc['_id']) for doc in self.tracks_documents]
        self.doc_id_to_index_map = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
        
        self.lda_model = gensim.models.ldamodel.LdaModel(
            corpus=self.corpus, 
            num_topics=num_topics, 
            id2word=self.dictionary, 
            alpha='auto', eta='auto', 
            passes=10
        )
        
        doc_topic_matrix = self.get_document_topic_matrix(num_topics)
        top_similarities_json = self.compute_top_similar_songs(doc_topic_matrix, self.tracks_documents, top_n=20)
        
        os.makedirs(output_dir, exist_ok=True)
        
        np.save(os.path.join(output_dir, 'lda_matrix.npy'), doc_topic_matrix)
        
        with open(os.path.join(output_dir, 'doc_id_to_index_map.json'), 'w') as f:
            json.dump(self.doc_id_to_index_map, f, indent=2)
        
        with open(os.path.join(output_dir, 'top_similarities.json'), 'w') as f:
            json.dump(top_similarities_json, f, indent=2)
        
        self.dictionary.save(os.path.join(output_dir, 'dictionary.gensim'))
        corpora.MmCorpus.serialize(os.path.join(output_dir, 'corpus.mm'), self.corpus)
        self.lda_model.save(os.path.join(output_dir, 'lda_model.gensim'))
        
        with open(os.path.join(output_dir, 'texts.txt'), 'w', encoding='utf-8') as f:
            for text in self.texts:
                f.write(' '.join(text) + '\n')
        
        print(f"Model outputs saved to {output_dir}")
        
        
    def get_song_topics(self, song_id):
        # 找到歌曲在语料库中的索引
        index = self.doc_id_to_index_map.get(song_id)
    
        # 获取该歌曲的主题分布
        topic_distribution = self.lda_model[self.corpus[index]]
    
        # 按照主题概率排序
        sorted_topics = sorted(topic_distribution, key=lambda x: x[1], reverse=True)
    
        # 返回主题分布和主题词
        result = []
        for topic_id, prob in sorted_topics:
            topic_words = self.lda_model.show_topic(topic_id, topn=10)
            result.append({
                'topic_id': int(topic_id),  # 确保 topic_id 是普通的 Python int
                'probability': float(prob),  # 将 numpy.float32 转换为 Python float
                'top_words': [word for word, _ in topic_words]
            })
    
        print(result)
        return result
    

    def evaluate_model(self):
        topic_nums = list(range(10, 101, 5))
        coherence_scores = []
        perplexity_scores = []
        
        for num_topics in tqdm(topic_nums, desc="Evaluating LDA Models"):
            lda_model = gensim.models.ldamodel.LdaModel(corpus=self.corpus, id2word=self.dictionary, num_topics=num_topics, random_state=42)
            
            coherence_score = self.get_coherence(lda_model)
            coherence_scores.append(coherence_score)
            
            perplexity_score = self.get_perplexity(lda_model)
            perplexity_scores.append(perplexity_score)
        
        self.plot_evaluation_results(topic_nums, coherence_scores, perplexity_scores)
        
        for num_topics, coherence_score, perplexity_score in zip(topic_nums, coherence_scores, perplexity_scores):
            print(f"Num Topics: {num_topics}, Coherence Score: {coherence_score}, Perplexity: {perplexity_score}")

    def plot_evaluation_results(self, topic_nums, coherence_scores, perplexity_scores):
        fig, ax1 = plt.subplots()
        
        color = 'tab:red'
        ax1.set_xlabel('Number of Topics')
        ax1.set_ylabel('Coherence Score', color=color)
        ax1.plot(topic_nums, coherence_scores, color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Perplexity', color=color)
        ax2.plot(topic_nums, perplexity_scores, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        
        fig.tight_layout()
        plt.title("Coherence Score and Perplexity vs Number of Topics")
        plt.show()

    def load_from_file(self, input_dir='lda'):
        try:
            with open(os.path.join(input_dir, 'doc_id_to_index_map.json'), 'r') as f:
                self.doc_id_to_index_map = json.load(f)
            self.dictionary = corpora.Dictionary.load(os.path.join(input_dir, 'dictionary.gensim'))
            self.corpus = corpora.MmCorpus(os.path.join(input_dir, 'corpus.mm'))
            self.lda_model = gensim.models.ldamodel.LdaModel.load(os.path.join(input_dir, 'lda_model.gensim'))
            
            with open(os.path.join(input_dir, 'texts.txt'), 'r', encoding='utf-8') as f:
                self.texts = [line.strip().split() for line in f]
            
            print(f"All files loaded from {input_dir}")
            return True
        except Exception as e:
            print(f"Error loading files: {e}")
            return False

    def get_coherence(self, lda_model=None):
        model = lda_model or self.lda_model
        coherence_model = gensim.models.CoherenceModel(model=model, texts=self.texts, dictionary=self.dictionary, coherence='c_v')
        coherence_score = coherence_model.get_coherence()
        print(f"Coherence Score: {coherence_score}")
        return coherence_score

    def get_perplexity(self, lda_model=None):
        model = lda_model or self.lda_model
        perplexity = model.log_perplexity(self.corpus)
        print(f"Perplexity: {perplexity}")
        return perplexity

    def topic_prediction(self, new_lyric):
        new_lyric_processed = self.preprocessor.preprocess_lyrics([new_lyric])[0]
        new_lyric_bow = self.dictionary.doc2bow(word_tokenize(new_lyric_processed))
        print("New lyric:", new_lyric)
        print("Processed lyric:", new_lyric_processed)
        print("Topic distribution:")
        pprint(sorted(self.lda_model[new_lyric_bow], key=lambda x: x[1], reverse=True))


    def get_document_topic_matrix(self, num_topics):
        doc_topic_matrix = np.zeros((len(self.corpus), num_topics))
        for i, doc_bow in enumerate(self.corpus):
            doc_topics = self.lda_model.get_document_topics(doc_bow, minimum_probability=0)
            for topic, prob in doc_topics:
                doc_topic_matrix[i, topic] = prob
        print(f"Matrix shape: {doc_topic_matrix.shape}")
        print("Sample of first 5 rows and 5 columns:")
        print(doc_topic_matrix[:5, :5])
        return doc_topic_matrix

    def compute_top_similar_songs(self, song_topic_matrix, tracks_documents, top_n=20):
        num_songs = song_topic_matrix.shape[0]
        similarities = cosine_similarity(song_topic_matrix)
        top_similarities_json = []

        for i in tqdm(range(num_songs), desc="Computing similar songs"):
            song_similarities = similarities[i]
            song_similarities[i] = -1
            top_similar_indices = np.argsort(song_similarities)[::-1][:top_n]
            top_similar_docs = [
                {
                    "track": {"$oid": str(tracks_documents[idx]['_id'])},
                    "value": float(song_similarities[idx])
                }
                for idx in top_similar_indices
            ]
            top_similarities_json.append({
                "track": {"$oid": str(self.tracks_documents[i]['_id'])},
                "topsimilar": top_similar_docs
            })

        return top_similarities_json
    

    
    

if __name__ == "__main__":
    lda_manager = LDAModelManager()
    num_topics = 20
    input_dir = 'lda'

    # 检查是否存在已保存的模型文件
    if all(os.path.exists(os.path.join(input_dir, f)) for f in ['dictionary.gensim', 'corpus.mm', 'lda_model.gensim', 'texts.txt']):
        print("Loading LDA results from files...")
        if lda_manager.load_from_file(input_dir):
            print("LDA model loaded successfully.")
        else:
            print("Failed to load LDA model. Training a new one...")
            lda_manager.load_mongo_and_train(num_topics=num_topics)
    else:
        print("Loading data from MongoDB and training LDA model...")
        lda_manager.load_mongo_and_train(num_topics=num_topics)

    if lda_manager.lda_model is not None:

        
        
        # lda_manager.get_song_topics('65ffc183c1ab936c978f29a8')
        
        print("\n1. Top 5 words for each topic:")
        pprint(lda_manager.lda_model.print_topics(num_topics=num_topics, num_words=5))

        print("\n2. Calculate topic coherence score:")
        lda_manager.get_coherence()
        
        print("\n3. Calculate topic perplexity score:")
        lda_manager.get_perplexity()
        
        print("\n4. Predict topics for a new lyric:")
        lda_manager.topic_prediction("Accepting Your grace with Love")

        
        print("\n5. Generate document-topic matrix:")
        lda_manager.get_document_topic_matrix(num_topics)

    # 如果您想评估不同主题数量的模型性能,可以取消注释下面的行
    # lda_manager.evaluate_model()