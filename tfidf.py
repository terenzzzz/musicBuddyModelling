import numpy as np
import pickle
import os
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import json
from tqdm import tqdm
from preprocessor import Preprocessor


class TFIDFManager:
    def __init__(self, mongo_uri='mongodb://localhost:27017/', db_name='MusicBuddyVue', collection_name='tracks', output_dir='tfidf'):
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.output_dir = output_dir
        self.tfidf_matrix = None
        self.top_similarities_per_doc = None
        self.feature_names = None
        self.doc_id_to_index_map = None
        self.vectorizer = None
        self.top_keywords_per_doc = None
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
            

    def load_mongo_and_train(self, N=20):
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.load_preprocessed_data()
        print('Preprocessed data loaded')
        

        # Calculate TF-IDF using TfidfVectorizer
        self.vectorizer = TfidfVectorizer()
        
        print("Calculating tf_idf_matrix...")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.processed_lyrics)
        print("tfidf_matrix shape", self.tfidf_matrix.shape)
    
        
        # Calculate cosine similarities
        print("Calculating cosine_similarities...")
        cosine_similarities = cosine_similarity(self.tfidf_matrix)

        
        # Get feature names (vocabulary)
        print("Getting feature names (vocabulary)...")
        self.feature_names = self.vectorizer.get_feature_names_out()
        

        # Save TF-IDF matrix 
        with open(os.path.join(self.output_dir, 'tfidf_matrix.pkl'), 'wb') as f:
            pickle.dump(self.tfidf_matrix, f)

        
        # Sort indices to get top N similarities (in descending order)
        top_n_similarities = np.argsort(cosine_similarities, axis=1)[:, -N-1:-1][:, ::-1]
        
        doc_ids = [str(doc['_id']) for doc in self.tracks_documents]
        self.doc_id_to_index_map = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
        
        # Compute top N similarities for each document and save as JSON
        print("Computing top N similarities for each document...")
        self.top_similarities_per_doc = []
        for i, doc_id in enumerate(tqdm(doc_ids, desc="Preparing JSON for Top Similarities")):
            # Exclude self-similarity by skipping the first element in top_n_similarities[i]
            similar_docs = [
                {
                    "track": {"$oid": doc_ids[idx]},
                    "value": float(cosine_similarities[i, idx])
                }
                for idx in top_n_similarities[i]
                if idx != i  # Exclude self
            ]
            doc_data = {
                "track": {"$oid": doc_id},
                "topsimilar": similar_docs
            }
            self.top_similarities_per_doc.append(doc_data)

        
        # Save top similarities in JSON format
        with open(os.path.join(self.output_dir, 'top_similarities.json'), 'w') as f:
            json.dump(self.top_similarities_per_doc, f, indent=2)
            
        # Save feature names
        with open(os.path.join(self.output_dir, 'feature_names.pkl'), 'wb') as f:
            pickle.dump(self.feature_names, f)
        
        # Save document IDs
        with open(os.path.join(self.output_dir, 'doc_id_to_index_map.json'), 'w') as f:
            json.dump(self.doc_id_to_index_map, f, indent=2)
            
        # Save TfidfVectorizer model
        joblib.dump(self.vectorizer, os.path.join(self.output_dir, 'tfidf_vectorizer.joblib'))
        
        # Compute top N keywords for each document
        self.top_keywords_per_doc = []
        for i, doc_id in enumerate(doc_ids):
            tfidf_vector = self.tfidf_matrix[i]
            sorted_indices = tfidf_vector.toarray().argsort()[0][::-1][:N]
            top_keywords = [{"word": self.feature_names[idx], "value": tfidf_vector[0, idx]} for idx in sorted_indices]
            doc_data = {
                "track": {"$oid": doc_id},
                "topwords": top_keywords
            }
            self.top_keywords_per_doc.append(doc_data)
        
        # Save top keywords in JSON format
        with open(os.path.join(self.output_dir, 'top_keywords.json'), 'w') as f:
            json.dump(self.top_keywords_per_doc, f, indent=2)
            
        print("TF-IDF results saved successfully.")

    def load_from_file(self,input_dir='tfidf'):
        try:
            # Load TF-IDF matrix 
            with open(os.path.join(input_dir, 'tfidf_matrix.pkl'), 'rb') as f:
                self.tfidf_matrix = pickle.load(f)
                print("tfidf_matrix shape", self.tfidf_matrix.shape)
            
            with open(os.path.join(input_dir, 'feature_names.pkl'), 'rb') as f:
                self.feature_names = pickle.load(f)
            
            # Load document IDs
            with open(os.path.join(input_dir, 'doc_id_to_index_map.json'), 'r') as f:
                self.doc_id_to_index_map = json.load(f)
            
            # Load TfidfVectorizer model
            self.vectorizer = joblib.load(os.path.join(input_dir, 'tfidf_vectorizer.joblib'))
            
            # Load top keywords from JSON file
            with open(os.path.join(input_dir, 'top_keywords.json'), 'r') as f:
                self.top_keywords_per_doc = json.load(f)
                
            # Load top similarity from JSON file
            with open(os.path.join(input_dir, 'top_similarities.json'), 'r') as f:
                self.top_similarities_per_doc = json.load(f)


        except FileNotFoundError as e:
            print(f"Error loading from files: {e}")
            return False
        return True
    
    def compute_top_similar_songs(self, similarity_matrix, tracks_documents, top_n=20):
        num_songs = similarity_matrix.shape[0]
        similarities = cosine_similarity(similarity_matrix)
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

    def get_similar_documents(self, doc_id):
        similar_documents = []
        for item in self.top_similarities_per_doc:
            if item['track']['$oid'] == doc_id:
                similar_documents.extend([doc['track']['$oid'] for doc in item['topsimilar']])
                break
        return similar_documents

    def get_top_words(self, doc_id):
        top_words = None
        for item in self.top_keywords_per_doc:
            if item['track']['$oid'] == doc_id:
                top_words = item['topwords']
                break
        return top_words
    
    def get_top_words_by_lyric(self, input_lyrics_list, top_n=20):
        # 确保输入是一个列表
        if not isinstance(input_lyrics_list, list):
            input_lyrics_list = [input_lyrics_list]
        
        # 预处理输入的歌词数组
        processed_inputs = self.preprocessor.preprocess_lyrics(input_lyrics_list)
        
        # 使用训练好的 TF-IDF 模型转换预处理后的歌词
        tfidf_matrix = self.vectorizer.transform(processed_inputs)
        
        # 计算平均 TF-IDF 向量
        avg_tfidf_vector = np.mean(tfidf_matrix.toarray(), axis=0)
        
        # 获取特征名称（单词）
        feature_names = self.vectorizer.get_feature_names_out()
        
        # 创建 (单词, 平均 TF-IDF 分数) 的元组列表，只包括分数不为 0 的词
        word_scores = [(word, score) for word, score in zip(feature_names, avg_tfidf_vector) if score > 0]
        
        # 按平均 TF-IDF 分数降序排序
        word_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 获取前 top_n 个单词及其分数
        top_words = word_scores[:top_n]
        
        # 转换为字典格式，将分数转换为 float 类型
        top_words_dict = [{'word': word, 'value': float(score)} for word, score in top_words]
        
        return top_words_dict


        return top_words_dict
    
    def get_similar_documents_for_lyrics(self, input_lyrics_list, top_n=20):
        # 确保输入是一个列表
        if not isinstance(input_lyrics_list, list):
            input_lyrics_list = [input_lyrics_list]
    
        # 预处理输入的歌词列表
        processed_inputs = self.preprocessor.preprocess_lyrics(input_lyrics_list)
    
        # 使用已训练的vectorizer将预处理后的歌词转换为TF-IDF向量
        input_vectors = self.vectorizer.transform(processed_inputs)
    
        # 计算平均TF-IDF向量并转换为numpy数组
        average_vector = np.asarray(input_vectors.mean(axis=0)).flatten()
    
        # 计算平均向量与所有文档的余弦相似度
        cosine_similarities = cosine_similarity(average_vector.reshape(1, -1), self.tfidf_matrix).flatten()
    
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

if __name__ == "__main__":
    tfidf_manager = TFIDFManager()
    input_dir = 'tfidf'  # or other directory where you save files
    if (os.path.exists(os.path.join(input_dir, 'tfidf_matrix.pkl')) and
        os.path.exists(os.path.join(input_dir, 'top_similarities.json')) and
        os.path.exists(os.path.join(input_dir, 'top_keywords.json')) and
        os.path.exists(os.path.join(input_dir, 'feature_names.pkl')) and
        os.path.exists(os.path.join(input_dir, 'doc_id_to_index_map.json')) and
        os.path.exists(os.path.join(input_dir, 'tfidf_vectorizer.joblib'))):
    
        print("Loading TF-IDF results from files...")
        tfidf_manager.load_from_file(input_dir)
    else:
        print("Loading data from MongoDB and training TF-IDF model...")
        tfidf_manager.load_mongo_and_train()
    
    if tfidf_manager.tfidf_matrix is not None:
        print("TF-IDF Matrix is ready.")
        
        
        lyric="If he's cheatin', I'm doin' him worse (Like) No Uno, I hit the reverse (Grrah) I ain't trippin', the grip in my purse (Grrah) I don't care 'cause he did it first (Like) If he's cheatin', I'm doin' him worse (Damn) I ain't trippin', I— (I ain't trippin', I—) I ain't trippin', the grip in my purse (Like) I don't care 'cause he did it first"
        lyric2="Honey, I'm a good man, but I'm a cheatin' man And I'll do all I can, to get a lady's love And I wanna do right, I don't wanna hurt nobody If I slip, well then I'm sorry, yes I am"
        
        similar_documents = tfidf_manager.get_similar_documents_for_lyrics([lyric,lyric2])
        print(similar_documents)
        
        
        top_words = tfidf_manager.get_top_words_by_lyric([lyric,lyric2])
        print(top_words)
        
        
        
        # query_doc_id = '65ffc183c1ab936c978f29a8'

        # similar_docs = tfidf_manager.get_similar_documents(query_doc_id)
        # print(f"Similar documents for document {query_doc_id}:")
        # print(similar_docs)
        
        # top_words = tfidf_manager.get_top_words(query_doc_id)
        # print(f"Top words for document {query_doc_id}:")
        # print(top_words)
        
        
        
        
        
        
        
        