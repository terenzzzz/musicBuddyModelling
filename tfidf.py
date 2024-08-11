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
from scipy.sparse import csr_matrix
from memory_profiler import profile
from scipy.sparse import issparse

class TFIDFManager:
    def __init__(self, mongo_uri='mongodb+srv://terence592592:592592@musicbuddy.grxyfb1.mongodb.net/', db_name='MusicBuddyVue', collection_name='tracks', output_dir='tfidf'):
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.output_dir = output_dir
        self.tfidf_matrix = None
        self.doc_id_to_index_map = None
        self.vectorizer = None
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
        self.vectorizer = TfidfVectorizer(
                            max_df=0.8,              # Ignore terms that appear in more than 80% of the documents
                            max_features=20000,      # Keep only the top 20,000 most important features
                            stop_words='english',    # Remove English stop words
                            sublinear_tf=True         # Apply sublinear term frequency scaling replace tf with 1 + log(tf)
                        )
        
        print("Calculating tf_idf_matrix...")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.processed_lyrics)
        print("tfidf_matrix shape", self.tfidf_matrix.shape)
    

        # Save TF-IDF matrix 
        with open(os.path.join(self.output_dir, 'tfidf_matrix.pkl'), 'wb') as f:
            pickle.dump(self.tfidf_matrix, f)


        doc_ids = [str(doc['_id']) for doc in self.tracks_documents]
        self.doc_id_to_index_map = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
        

        # Save document IDs
        with open(os.path.join(self.output_dir, 'doc_id_to_index_map.json'), 'w') as f:
            json.dump(self.doc_id_to_index_map, f, indent=2)
            
        # Save TfidfVectorizer model
        joblib.dump(self.vectorizer, os.path.join(self.output_dir, 'tfidf_vectorizer.joblib'))
            
        print("TF-IDF results saved successfully.")

    def load_from_file(self,input_dir='tfidf'):
        try:
            # Load TF-IDF matrix 
            with open(os.path.join(input_dir, 'tfidf_matrix.pkl'), 'rb') as f:
                self.tfidf_matrix = pickle.load(f)
                print("tfidf_matrix shape", self.tfidf_matrix.shape)
            
            # Load document IDs
            with open(os.path.join(input_dir, 'doc_id_to_index_map.json'), 'r') as f:
                self.doc_id_to_index_map = json.load(f)
            
            # Load TfidfVectorizer model
            self.vectorizer = joblib.load(os.path.join(input_dir, 'tfidf_vectorizer.joblib'))
            


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

    def get_top_words_by_lyric(self, input_lyrics_list, top_n=20):
        # 根据传入的歌词计算tfidf值最高的词(TF-IDF 值高的词代表在特定文档中具有更高的重要性)
        # 确保输入是一个列表
        if not isinstance(input_lyrics_list, list):
            input_lyrics_list = [input_lyrics_list]
        
        
        # 计算平均 TF-IDF 向量
        average_vector = self.compute_mean_vector(input_lyrics_list)
        
        # 获取特征名称（单词）
        feature_names = self.vectorizer.get_feature_names_out()
        
        # 创建 (单词, 平均 TF-IDF 分数) 的元组列表，只包括分数不为 0 的词
        word_scores = [(word, score) for word, score in zip(feature_names, average_vector) if score > 0]
        
        # 按平均 TF-IDF 分数降序排序
        word_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 获取前 top_n 个单词及其分数
        top_words = word_scores[:top_n]
        
        # 转换为字典格式，将分数转换为 float 类型
        top_words_dict = [{'word': word, 'value': float(score)} for word, score in top_words]
        
        return top_words_dict


    def get_similar_documents_for_lyrics(self, input_lyrics_list, top_n=20):
        if not isinstance(input_lyrics_list, list):
            input_lyrics_list = [input_lyrics_list]
    
        # 计算平均TF-IDF向量
        average_vector = self.compute_mean_vector(input_lyrics_list)
        average_vector_sparse = csr_matrix(average_vector.reshape(1, -1))
        
        # 计算平均向量与所有文档的余弦相似度
        cosine_similarities = cosine_similarity(average_vector_sparse, self.tfidf_matrix).flatten()
        
        # 获取相似度最高的top_n个文档的索引
        top_indices = np.argpartition(-cosine_similarities, top_n)[:top_n]
        top_indices = top_indices[np.argsort(-cosine_similarities[top_indices])]
    
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
        # 预处理输入的歌词数组
        processed_inputs = self.preprocessor.preprocess_lyrics(lyrics)
        
        # 使用训练好的 TF-IDF 模型转换预处理后的歌词
        tfidf_matrix = self.vectorizer.transform(processed_inputs)
        
        # 计算平均 TF-IDF 向量
        average_vector = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
        
        return average_vector
    
    def compute_similarity_between_two_doc(self, doc1, doc2):
        
        processed_doc_1 = self.preprocessor.preprocess_lyrics([doc1])
        processed_doc_2 = self.preprocessor.preprocess_lyrics([doc2])
        
        tfidf_matrix_1 = self.vectorizer.transform(processed_doc_1)
        tfidf_matrix_2 = self.vectorizer.transform(processed_doc_2)
        
        similarities = cosine_similarity(tfidf_matrix_1, tfidf_matrix_2)
        
        # 提取矩阵中的唯一值（即相似度）
        similarity_value = similarities[0, 0]
        return similarity_value
        
        

if __name__ == "__main__":
    tfidf_manager = TFIDFManager()
    input_dir = 'tfidf'  # or other directory where you save files
    if (os.path.exists(os.path.join(input_dir, 'tfidf_matrix.pkl')) and
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
        
        
        # top_words_1 = tfidf_manager.get_top_words_by_lyric([lyrics_1])
        # top_words_2 = tfidf_manager.get_top_words_by_lyric([lyrics_2])
        # top_words_3 = tfidf_manager.get_top_words_by_lyric([lyrics_3])
        # print(f'Top Words for lyrics_1: {top_words_1}' )
        # print(f'Top Words for lyrics_2: {top_words_2}' )
        # print(f'Top Words for lyrics_3: {top_words_3}' )
        
        
        # Evaluation
        print('========================== Evaluation 1 =============================')
        lyrics_1 = "The stars in the night sky are shining bright"
        lyrics_2 = "Bright stars light up the dark night sky"
        lyrics_3 = "The rain falls gently on the green grass"
    
        
        
        
        similar_1_2 = tfidf_manager.compute_similarity_between_two_doc(lyrics_1,lyrics_2)
        similar_1_3 = tfidf_manager.compute_similarity_between_two_doc(lyrics_1,lyrics_3)
        similar_2_3 = tfidf_manager.compute_similarity_between_two_doc(lyrics_2,lyrics_3)
        print(f'Similarity for lyrics_1 and lyrics_2: {similar_1_2}' )
        print(f'Similarity for lyrics_1 and lyrics_3: {similar_1_3}' )
        print(f'Similarity for lyrics_2 and lyrics_3: {similar_2_3}' )
        

        
        print('========================== Comprehensive TFIDF Model Evaluation =============================')

        test_cases = [
            # 1. 完全相同的歌词
            ("I'm walking on sunshine, whoa", "I'm walking on sunshine, whoa"),
        
            # 2. 词序略有变化
            ("Love is in the air, everywhere I look around", "Everywhere I look around, love is in the air"),
        
            # 3. 同义词替换
            ("You are beautiful, you are kind", "You're gorgeous, you're nice"),
        
            # 4. 部分重叠
            ("Dancing in the moonlight, everybody's feeling warm and bright",
             "Dancing in the moonlight, it's such a fine and natural sight"),
        
            # 5. 相似主题，不同表达
            ("The sun goes down, the stars come out", "As daylight fades, the night sky gleams"),
        
            # 6. 完全不同的歌词
            ("Twinkle, twinkle, little star", "We will rock you, rock you"),
        
            # 7. 重复结构，不同词汇
            ("I love you, you love me", "She sings high, he sings low"),
        
            # 8. 包含常见歌词术语
            ("Verse: The story begins\nChorus: This is our song", 
             "First verse: Once upon a time\nChorus: Sing it loud"),
        
            # 9. 不同时态
            ("I am singing in the rain", "I sang in the rain")
        ]
        
        # 计算相似度
        for i, (lyrics1, lyrics2) in enumerate(test_cases, 1):
            similarity = tfidf_manager.compute_similarity_between_two_doc(lyrics1, lyrics2)
            print(f'Test case {i}: Similarity = {similarity}')
            print(f'Lyrics 1: {lyrics1}')
            print(f'Lyrics 2: {lyrics2}')
            print('-' * 50)
        
        
        
        
        