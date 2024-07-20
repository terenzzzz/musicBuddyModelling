from weightedManager import weightedManager
from pymongo import MongoClient
import numpy as np
import os
import json
from sklearn.metrics.pairwise import cosine_similarity


class ArtistManager:
    def __init__(self, mongo_uri='mongodb://localhost:27017/', db_name='MusicBuddyVue', 
                 tracks_collection_name='tracks', artists_collection_name= "artists", output_dir='artists'):
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.tracks_collection = self.db[tracks_collection_name]
        self.artists_collection = self.db[artists_collection_name]
        self.output_dir = output_dir
        self.weighted_manager = weightedManager(0.33, 0.33, 0.34,"tfidf/doc_id_to_index_map.json")
        self.artists_documents = None
        
        self.tfidf_matrix = None
        self.artist_id_to_index = None
        
        self.load_artist_from_database()
        
    def load_artist_from_database(self):
        try:
            artists_documents = list(self.artists_collection.find())
            if not artists_documents:
                print("No documents found in MongoDB collection.")
                return None
            else:
                print(f"Found {len(artists_documents)} documents in MongoDB collection.")
            self.artists_documents = artists_documents
        except Exception as e:
            print(f"Error fetching documents from MongoDB: {e}")
            return None
        


    def compute_artists_tfidf_vector(self, output_file='artists_tfidf.npy', mapping_file='artist_id_mapping.json'):
        artist_vectors = []
        artist_id_to_index = {}
        
        for idx, artist in enumerate(self.artists_documents):
            artist_id = artist['_id']
            artist_id_to_index[str(artist_id)] = idx
            tracks = list(self.tracks_collection.find({'artist': artist_id}))
            
            if not tracks:
                print(f"No tracks found for artist {artist['name']}.")
                continue
            
            lyrics = [track['lyric'] for track in tracks if 'lyric' in track]
            
            if not lyrics:
                print(f"No lyrics found for artist {artist['name']}.")
                continue
        
            average_tfidf = self.weighted_manager.tfidf_manager.compute_mean_vector(lyrics)
            artist_vectors.append(average_tfidf)
            print(f"{artist_id}: {average_tfidf}")
        
        # 转换为 NumPy 矩阵
        self.tfidf_matrix = np.array(artist_vectors)
        self.artist_id_to_index = artist_id_to_index
        
        os.makedirs(self.output_dir, exist_ok=True)


        # 保存 TF-IDF 矩阵到文件
        np.save(os.path.join(self.output_dir, output_file), self.tfidf_matrix)

        # 保存艺术家ID到索引的映射到文件
        with open(os.path.join(self.output_dir, mapping_file), 'w') as f:
            json.dump(artist_id_to_index, f)
        
        return 0
    
    def load_tfidf_matrix(self, tfidf_matrix_path='artists/artists_tfidf.npy', id_mapping_path='artists/artist_id_mapping.json'):
        if (os.path.exists(tfidf_matrix_path) and
            os.path.exists(id_mapping_path) and False):
            print("Loading tfidf from files")
            # 读取 TF-IDF 矩阵
            self.tfidf_matrix = np.load(tfidf_matrix_path)
             
             # 读取艺术家ID到索引的映射
            with open(id_mapping_path, 'r') as f:
                 self.artist_id_to_index = json.load(f)
        else:
            print("Calculating TFIDF")
            self.compute_artists_tfidf_vector()
             
    def get_tfidf_similar_artists_by_artist(self, artist_id):
        if self.tfidf_matrix is None or self.artist_id_to_index is None:
            self.load_tfidf_matrix()

        if str(artist_id) not in self.artist_id_to_index:
            print(f"Artist ID {artist_id} not found in the mapping.")
            return []

        # 获取艺术家的索引
        artist_index = self.artist_id_to_index[str(artist_id)]
        
        # 计算与其他艺术家的余弦相似度
        artist_vector = self.tfidf_matrix[artist_index].reshape(1, -1)
        similarities = cosine_similarity(artist_vector, self.tfidf_matrix).flatten()
        
        # 获取最高的20个相似艺术家
        similar_indices = similarities.argsort()[::-1][1:21]  # 排除自己
        similar_artists = [(index, similarities[index]) for index in similar_indices]
        
        # 将索引转换为艺术家ID
        index_to_artist_id = {v: k for k, v in self.artist_id_to_index.items()}
        similar_artists_with_ids = [(index_to_artist_id[idx], sim) for idx, sim in similar_artists]
        
        return similar_artists_with_ids
             

    
   
    
    def compute_artists_w2v_vector(self):
        return 0
    
    def compute_artists_lda_vector(self):
        return 0
    
    
    def get_similar_artist(self):
        return 0
    

if __name__ == "__main__":
    artist_manager = ArtistManager()
    top_similar = artist_manager.get_tfidf_similar_artists_by_artist("65ff82ee564ab6bcd8592ea6")
    print(top_similar)