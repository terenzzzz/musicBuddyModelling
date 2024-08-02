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


class LDAModelManager:
    def __init__(self, mongo_uri='mongodb://localhost:27017/', db_name='MusicBuddyVue', collection_name='tracks'):
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.dictionary = None
        self.corpus = None
        self.lda_model = None
        self.doc_topic_matrix = None
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

    def load_mongo_and_train(self, num_topics=15, output_dir='lda'):
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
            passes=1
        )
        
        self.doc_topic_matrix = self.get_document_topic_matrix(num_topics)
    
        
        os.makedirs(output_dir, exist_ok=True)
        
        np.save(os.path.join(output_dir, 'lda_matrix.npy'), self.doc_topic_matrix)
        
        with open(os.path.join(output_dir, 'doc_id_to_index_map.json'), 'w') as f:
            json.dump(self.doc_id_to_index_map, f, indent=2)
        
        
        self.dictionary.save(os.path.join(output_dir, 'dictionary.gensim'))
        corpora.MmCorpus.serialize(os.path.join(output_dir, 'corpus.mm'), self.corpus)
        self.lda_model.save(os.path.join(output_dir, 'lda_model.gensim'))
        
        with open(os.path.join(output_dir, 'texts.txt'), 'w', encoding='utf-8') as f:
            for text in self.texts:
                f.write(' '.join(text) + '\n')
                
        self.save_topics_to_json()
        
        print(f"Model outputs saved to {output_dir}")

    def evaluate_model(self):
        topic_nums = list(range(5, 51, 5))
        coherence_scores = []
        perplexity_scores = []
        
        for num_topics in tqdm(topic_nums, desc="Evaluating LDA Models"):
            lda_model = gensim.models.ldamodel.LdaModel(
                corpus=self.corpus, 
                id2word=self.dictionary, 
                num_topics=num_topics, 
                random_state=42,
                alpha='auto', eta='auto', 
                passes=100)
            
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
            # 加载所有文档的主题分布矩阵
            self.doc_topic_matrix = np.load('lda/lda_matrix.npy')
            
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
    
    
    def get_topics_by_lyric(self, input_lyrics_list):
        if not isinstance(input_lyrics_list, list):
            input_lyrics_list = [input_lyrics_list]
    
        # 获取平均向量的主题分布
        average_vector = self.compute_mean_vector(input_lyrics_list)
    
        # 将平均向量转换为 (主题编号, 概率) 的格式
        # num_topics = len(average_vector)
        topic_distribution = [(i, float(prob)) for i, prob in enumerate(average_vector)]
    
        # 按照主题概率排序
        sorted_topics = sorted(topic_distribution, key=lambda x: x[1], reverse=True)
    
        # 返回主题分布和主题词
        result = []
        for topic_id, prob in sorted_topics:
            topic_words = self.lda_model.show_topic(topic_id, topn=10)
            result.append({
                'topic_id': int(topic_id),
                'probability': float(prob),
                'top_words': [word for word, _ in topic_words]
            })
    
        return result


    
    def get_similar_documents_for_lyrics(self, input_lyrics_list, top_n=20):
        # 确保输入是一个列表
        if not isinstance(input_lyrics_list, list):
            input_lyrics_list = [input_lyrics_list]
    
        
        # 计算平均LDA向量
        average_vector = self.compute_mean_vector(input_lyrics_list)
    

        # 计算平均向量与所有文档的余弦相似度
        cosine_similarities = cosine_similarity([average_vector], self.doc_topic_matrix)[0]
    
        # 获取相似度最高的top_n个文档的索引
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
    
        # 将预处理后的歌词转换为LDA向量
        input_vectors = []
        for lyrics in processed_inputs:
            bow = self.dictionary.doc2bow(word_tokenize(lyrics))
            vector = self.lda_model.get_document_topics(bow, minimum_probability=0)
            vector = [prob for (_, prob) in sorted(vector)]
            input_vectors.append(vector)
    
        # 计算平均LDA向量
        average_vector = np.mean(input_vectors, axis=0)
        
        return average_vector
    
    def save_topics_to_json(self, output_dir='lda'):
        # 获取主题信息
        topics = self.lda_model.print_topics(num_topics=num_topics, num_words=10)
        
        # 解析主题信息
        topics_list = []
        for topic_id, topic in topics:
            # topic 格式为 '0.077"hear" + 0.066"head" + ...'
            words = topic.split(' + ')
            topic_words = []
            for word in words:
                weight, term = word.split('*')
                weight = float(weight)
                term = term.replace('"', '')
                topic_words.append({'word': term, 'weight': weight})
            
            # 添加主题到列表
            topic_dict = {
                'topic_id': topic_id,
                'name': "",
                'words': topic_words
            }
            topics_list.append(topic_dict)
        
        # 将结果保存为JSON文件
        with open(os.path.join(output_dir, 'topics.json'), 'w', encoding='utf-8') as f:
            json.dump(topics_list, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    lda_manager = LDAModelManager()
    num_topics = 15
    input_dir = 'lda'

    # 检查是否存在已保存的模型文件
    if all(os.path.exists(os.path.join(input_dir, f)) for f in ['dictionary.gensim', 
                                                                'corpus.mm', 
                                                                'lda_model.gensim', 
                                                                'texts.txt']):
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

        lyric="If he's cheatin', I'm doin' him worse (Like) No Uno, I hit the reverse (Grrah) I ain't trippin', the grip in my purse (Grrah) I don't care 'cause he did it first (Like) If he's cheatin', I'm doin' him worse (Damn) I ain't trippin', I— (I ain't trippin', I—) I ain't trippin', the grip in my purse (Like) I don't care 'cause he did it first"
        lyric2="Honey, I'm a good man, but I'm a cheatin' man And I'll do all I can, to get a lady's love And I wanna do right, I don't wanna hurt nobody If I slip, well then I'm sorry, yes I am"
        
        # similar_documents = lda_manager.get_similar_documents_for_lyrics([lyric,lyric2])
        # print(similar_documents)
        
        track_topic_by_lyric = lda_manager.get_topics_by_lyric([lyric,lyric2])
        print(f"track_topic_by_lyric: {track_topic_by_lyric}")
        

        
        # print("\n1. Top 10 words for each topic:")
        # pprint(lda_manager.lda_model.print_topics(num_topics=num_topics, num_words=10))

        # print("\n2. Calculate topic coherence score:")
        # lda_manager.get_coherence()
        
        # print("\n3. Calculate topic perplexity score:")
        # lda_manager.get_perplexity()
        

        
    # 保存模型
    # lda_manager.save_topics_to_json()

    # 如果您想评估不同主题数量的模型性能,可以取消注释下面的行
    # lda_manager.evaluate_model()