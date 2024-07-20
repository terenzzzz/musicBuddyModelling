from pymongo import MongoClient
import numpy as np
import os
import json
from sklearn.metrics.pairwise import cosine_similarity
from lda import LDAModelManager
from word2vec import Word2VecManager
from tfidf import TFIDFManager

class ArtistManager:
    def __init__(self, tfidf_manager,w2v_manager,lda_manager,mongo_uri='mongodb://localhost:27017/', 
                 db_name='MusicBuddyVue', tracks_collection_name='tracks', 
                 artists_collection_name= "artists", output_dir='artists', ):
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.tracks_collection = self.db[tracks_collection_name]
        self.artists_collection = self.db[artists_collection_name]
        self.output_dir = output_dir
        self.artists_documents = None
        
        # 三种模型
        self.tfidf_manager = tfidf_manager
        self.w2v_manager = w2v_manager
        self.lda_manager = lda_manager
        
        self.artist_id_to_index = None
        self.tfidf_matrix = None
        self.w2v_matrix = None
        self.lda_matrix = None
        

        self.load_artist_id_to_index()
        
     # 加载模型文件
        
    def load_artist_from_database(self, mapping_file='artist_id_mapping.json'):
        try:
            artists_documents = list(self.artists_collection.find())
            if not artists_documents:
                print("No documents found in MongoDB collection.")
                return None
            else:
                print(f"Found {len(artists_documents)} documents in MongoDB collection.")
            self.artists_documents = artists_documents
            self.artist_id_to_index = {}
            for idx, artist in enumerate(self.artists_documents):
                artist_id = artist['_id']
                self.artist_id_to_index[str(artist_id)] = idx
                
            os.makedirs(self.output_dir, exist_ok=True)
                # 保存艺术家ID到索引的映射到文件
            with open(os.path.join(self.output_dir, mapping_file), 'w') as f:
                json.dump(self.artist_id_to_index, f)
            
                
        except Exception as e:
            print(f"Error fetching documents from MongoDB: {e}")
            return None

    def load_artist_id_to_index(self, id_mapping_path='artists/artist_id_mapping.json'):
        if os.path.exists(id_mapping_path):
            with open(id_mapping_path, 'r') as f:
                 self.artist_id_to_index = json.load(f)
            print("Loaded artist_id_to_index")
        else:
            self.load_artist_from_database()
            
    def load_tfidf_matrix(self, tfidf_matrix_path='artists/artists_tfidf.npy'):
        if (os.path.exists(tfidf_matrix_path)):
            self.tfidf_matrix = np.load(tfidf_matrix_path)
        else:
            print("Calculating TFIDF")
            self.compute_artists_tfidf_vector()
        print("Loaded tfidf matrix")
             
    def load_w2v_matrix(self, w2v_matrix_path='artists/artists_w2v.npy'):
        if (os.path.exists(w2v_matrix_path)):
            self.w2v_matrix = np.load(w2v_matrix_path)
        else:
            print("Calculating w2v")
            self.compute_artists_w2v_vector()
        print("Loaded w2v matrix")
    
    def load_lda_matrix(self, lda_matrix_path='artists/artists_lda.npy'):
        if (os.path.exists(lda_matrix_path)):
            self.lda_matrix = np.load(lda_matrix_path)
        else:
            print("Calculating lda")
            self.compute_artists_lda_vector()
        print("Loaded lda matrix")
        
    # 计算并保存模型矩阵

    def compute_artists_tfidf_vector(self, output_file='artists_tfidf.npy'):
        self.tfidf_matrix = self.compute_artist_matrix(self.tfidf_manager.compute_mean_vector, output_file)

    def compute_artists_w2v_vector(self,output_file='artists_w2v.npy'):
        self.w2v_matrix = self.compute_artist_matrix(self.w2v_manager.compute_mean_vector, output_file)
        
    def compute_artists_lda_vector(self,output_file='artists_lda.npy'):
        self.lda_matrix = self.compute_artist_matrix(self.lda_manager.compute_mean_vector, output_file)
    
        
    def compute_artist_matrix(self, mean_func, output_file):
        artist_vectors = []
        
        for idx, artist in enumerate(self.artists_documents):
            artist_id = artist['_id']
            tracks = list(self.tracks_collection.find({'artist': artist_id}))
            
            if not tracks:
                print(f"No tracks found for artist {artist['name']}.")
                continue
            
            lyrics = [track['lyric'] for track in tracks if 'lyric' in track]
            
            if not lyrics:
                print(f"No lyrics found for artist {artist['name']}.")
                continue
            
            average_vector = mean_func(lyrics)
            artist_vectors.append(average_vector)

        
        # 转换为 NumPy 矩阵
        matrix = np.array(artist_vectors)
        
        os.makedirs(self.output_dir, exist_ok=True)


        # 保存 TF-IDF 矩阵到文件
        np.save(os.path.join(self.output_dir, output_file), matrix)
        return matrix
    

    # 通过模型矩阵获取最高相似度的歌手
    
    def get_tfidf_similar_artists_by_artist(self, artist_id):
        return self.get_top_n_similar_artists(artist_id, self.tfidf_matrix, self.artist_id_to_index)
    
    def get_w2v_similar_artists_by_artist(self, artist_id):
        return self.get_top_n_similar_artists(artist_id, self.w2v_matrix, self.artist_id_to_index)
    
    def get_lda_similar_artists_by_artist(self, artist_id):
        return self.get_top_n_similar_artists(artist_id, self.lda_matrix, self.artist_id_to_index)
    
    def get_top_n_similar_artists(self, artist_id, matrix, id_to_index):

        similarities = self.get_similarities(artist_id, matrix, id_to_index)
        
        # 获取最高的20个相似艺术家
        similar_indices = similarities.argsort()[::-1][1:21]  # 排除自己
        similar_artists = [(index, similarities[index]) for index in similar_indices]
        
        # 将索引转换为艺术家ID
        index_to_artist_id = {v: k for k, v in id_to_index.items()}
        similar_artists_with_ids = [
            {
                'artist': {'$oid': index_to_artist_id[idx]},
                'similarity': float(sim)
            } for idx, sim in similar_artists
        ]
        
        return similar_artists_with_ids
    
    def get_similarities(self, artist_id, matrix, id_to_index):
        if matrix is None or id_to_index is None:
            raise ValueError("Matrix or ID mapping is not loaded.")
        
        artist_id_str = str(artist_id)
        if artist_id_str not in id_to_index:
            print(f"Artist ID {artist_id} not found in the mapping.")
            return None

        # 获取艺术家的索引
        artist_index = id_to_index[artist_id_str]
        
        # 计算与其他艺术家的余弦相似度
        artist_vector = matrix[artist_index].reshape(1, -1)
        similarities = cosine_similarity(artist_vector, matrix).flatten()
        
        return similarities


    def validate_weights(self, tfidf_weight, word2vec_weight, lda_weight):
        """
        检查给出的weighting是否总和为1
        """
        if not np.isclose(tfidf_weight + word2vec_weight + lda_weight, 1):
            raise ValueError("The sum of weights must be equal to 1")

    # 根据权重计算相似度
    def get_weighted_similar_artists_by_artist(self, artist_id, tfidf_weight, w2v_weight, lda_weight):
        self.validate_weights(tfidf_weight,w2v_weight,lda_weight)
        
        # 计算各模型的相似度
        tfidf_similarities = self.get_similarities(artist_id, self.tfidf_matrix, self.artist_id_to_index)
        w2v_similarities = self.get_similarities(artist_id, self.w2v_matrix, self.artist_id_to_index)
        lda_similarities = self.get_similarities(artist_id, self.lda_matrix, self.artist_id_to_index)
        
        if tfidf_similarities is None or w2v_similarities is None or lda_similarities is None:
            return []

        # 计算加权相似度
        combined_similarities = (tfidf_weight * tfidf_similarities +
                                 w2v_weight * w2v_similarities +
                                 lda_weight * lda_similarities)

        # 获取最高的20个相似艺术家
        similar_indices = combined_similarities.argsort()[::-1][1:21]  # 排除自己
        similar_artists = [(index, combined_similarities[index]) for index in similar_indices]
        
        # 将索引转换为艺术家ID
        index_to_artist_id = {v: k for k, v in self.artist_id_to_index.items()}
        similar_artists_with_ids = [
            {
                'artist': {'$oid': index_to_artist_id[idx]},
                'similarity': sim
            } for idx, sim in similar_artists
        ]
        
        return similar_artists_with_ids
             
    

    

if __name__ == "__main__":
    # Load tfidf model
    tfidf_manager = TFIDFManager()
    tfidf_manager.load_from_file("tfidf")

    # # Load word2vec model
    w2v_manager = Word2VecManager()
    w2v_manager.load_from_file("word2vec")

    # Load lda model
    lda_manager = LDAModelManager()
    lda_manager.load_from_file("lda")
    
    
    artist_manager = ArtistManager(tfidf_manager,w2v_manager,lda_manager)
    artist_manager.load_tfidf_matrix()
    artist_manager.load_w2v_matrix()
    artist_manager.load_lda_matrix()
    
    similar_artist_tfidf = artist_manager.get_tfidf_similar_artists_by_artist("65ff82ee564ab6bcd8592ea6")
    similar_artist_w2v = artist_manager.get_w2v_similar_artists_by_artist("65ff82ee564ab6bcd8592ea6")
    similar_artist_lda = artist_manager.get_lda_similar_artists_by_artist("65ff82ee564ab6bcd8592ea6")
    similar_artist_weighted = artist_manager.get_weighted_similar_artists_by_artist("65ff82ee564ab6bcd8592ea6", 0.33, 0.33, 0.34)
    
    print(f"Similar artist for artist 65ff82ee564ab6bcd8592ea6 in TFIDF: {similar_artist_tfidf}")
    print(f"Similar artist for artist 65ff82ee564ab6bcd8592ea6 in W2V: {similar_artist_w2v}")
    print(f"Similar artist for artist 65ff82ee564ab6bcd8592ea6 in LDA: {similar_artist_lda}")
    print(f"Similar artist for artist 65ff82ee564ab6bcd8592ea6 in Weighted: {similar_artist_weighted}")
    
