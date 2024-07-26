# -*- coding: utf-8 -*-
from pymongo import MongoClient
import json
from collections import OrderedDict
import os
import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity
from bson import ObjectId

# 定义一个自定义的 JSONEncoder 来处理 ObjectId
class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        return super(JSONEncoder, self).default(o)

class collaborateManager:
    def __init__(self, mongo_uri='mongodb://localhost:27017/', db_name='MusicBuddyVue', 
                 tracks_collection_name='tracks', 
                 users_collection_name= "users",
                 ratings_collection_name= "ratings",
                 output_dir='collaborate'):
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.tracks_collection = self.db[tracks_collection_name]
        self.users_collection = self.db[users_collection_name]
        self.ratings_collection = self.db[ratings_collection_name]
        self.output_dir = output_dir
        
        self.ratings_documents = None

        self.user_map = None
        self.track_map = None
        
        self.user_track_matrix = None
        self.user_similar_matrix = None
        self.track_similar_matrix = None
        
        
        self.user_map_path = os.path.exists(os.path.join(self.output_dir, 'user_map.json'))
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
            
    def load_user_map(self, user_map_path):
        if os.path.exists(user_map_path):
            # Load user_map
            with open(user_map_path, 'r') as f:
                items = json.load(f)
                self.user_map = OrderedDict(items)
            print("User map loaded From File")
        else:
            try:
                users_documents = list(self.users_collection.find({}, {'_id': 1}))
                if not users_documents:
                    print("No documents found in MongoDB collection.")
                else:
                    print(f"Found {len(users_documents)} users documents in MongoDB collection.")
                
                self.user_map = OrderedDict((str(doc['_id']), idx) for idx, doc in enumerate(users_documents))
                
                with open(os.path.join(self.output_dir, 'user_map.json'), 'w') as f:
                    json.dump(list(self.user_map.items()), f)
                print("User map saved")
                
            except Exception as e:
                print(f"Error fetching documents from MongoDB: {e}")
               
    
    def load_tracks_map(self, track_map_path):
        if os.path.exists(track_map_path):
            # Load track_map
            with open(track_map_path, 'r') as f:
                items = json.load(f)
                self.track_map = OrderedDict(items)
            print("Track map loaded From File")
        else:
            try:
                tracks_documents = list(self.tracks_collection.find({}, {'_id': 1}))
                if not tracks_documents:
                    print("No documents found in MongoDB collection.")
                else:
                    print(f"Found {len(tracks_documents)} users documents in MongoDB collection.")
                
                self.track_map = OrderedDict((str(doc['_id']), idx) for idx, doc in enumerate(tracks_documents))
                
                with open(os.path.join(self.output_dir, 'track_map.json'), 'w') as f:
                    json.dump(list(self.track_map.items()), f)
                print("Track map saved")
                
            except Exception as e:
                print(f"Error fetching documents from MongoDB: {e}") 
    
    def load_ratings_documents(self, ratings_documents_path):
        # 尝试从文件加载
        if os.path.exists(ratings_documents_path):
            try:
                with open(ratings_documents_path, 'r') as f:
                    self.ratings_documents = json.load(f)
                print(f"Loaded {len(self.ratings_documents)} ratings from file")
            except Exception as e:
                print(f"Error loading ratings from file: {e}")
                self.ratings_documents = []
        else:
            self.ratings_documents = []
        
            # 尝试从 MongoDB 加载
            try:
                mongo_ratings_documents = list(self.ratings_collection.find(
                    {'itemType': 'Track'}, {'user': 1, 'item': 1, 'rate': 1}
                ))
                if not mongo_ratings_documents:
                    print("No documents found in MongoDB collection.")
                else:
                    print(f"Found {len(mongo_ratings_documents)} users documents in MongoDB collection.")
                    self.ratings_documents = mongo_ratings_documents
                    
                    # 确保输出目录存在
                    if not os.path.exists(self.output_dir):
                        os.makedirs(self.output_dir)
                    
                    # 保存到文件
                    try:
                        with open(os.path.join(self.output_dir, 'ratings_documents.json'), 'w') as f:
                            json.dump(self.ratings_documents, f, cls=JSONEncoder)
                        print("Ratings saved successfully.")
                    except Exception as e:
                        print(f"Error saving ratings to file: {e}")
                        
            except Exception as e:
                print(f"Error fetching documents from MongoDB: {e}")

    def construct_user_track_matrix(self):
        # 初始化评分矩阵
        self.user_track_matrix = np.zeros((len(self.user_map), len(self.track_map)))

            # 填充评分矩阵
        for rating in self.ratings_documents:
            user_id = str(rating['user'])
            track_id = str(rating['item'])
            user_idx = self.user_map.get(user_id)
            track_idx = self.track_map.get(track_id)
            if user_idx is not None and track_idx is not None:
                self.user_track_matrix[user_idx, track_idx] = rating['rate']
                
        np.save(os.path.join(self.output_dir, 'user_track_matrix.npy'), self.user_track_matrix)
        return self.user_track_matrix
    
    def construct_user_similar_matrix(self):
        # 确保评分矩阵已经被填充
        if self.user_track_matrix is None:
            print("Error: Rating matrix is not initialized. Please load data first.")
            return None

        # 计算用户相似度矩阵
        self.user_similar_matrix = cosine_similarity(self.user_track_matrix)

        # 将对角线元素设置为0，因为用户与自己的相似度不需要考虑
        np.fill_diagonal(self.user_similar_matrix, 0)

        np.save(os.path.join(self.output_dir, 'user_similar_matrix.npy'), self.user_similar_matrix)
        return self.user_similar_matrix
    
    def construct_track_similar_matrix(self):
        # 确保评分矩阵已经被填充
        if self.user_track_matrix is None:
            print("Error: Rating matrix is not initialized. Please load data first.")
            return None

        # 转置矩阵，使得每一行代表一个曲目
        track_user_matrix = self.user_track_matrix.T

        # 计算曲目相似度矩阵
        self.track_similar_matrix = cosine_similarity(track_user_matrix)

        # 将对角线元素设置为0，因为曲目与自己的相似度不需要考虑
        np.fill_diagonal(self.track_similar_matrix, 0)

        np.save(os.path.join(self.output_dir, 'track_similar_matrix.npy'), self.track_similar_matrix)
        return self.track_similar_matrix
    

    def get_top_n_similar(self, similarities, item_type='user', n=20):
        # 创建一个包含索引和相似度的列表
        similar_items = list(enumerate(similarities))
        
        # 按相似度降序排序，但排除索引为0的项（自身）
        similar_items.sort(key=lambda x: x[1], reverse=True)
        similar_items = [item for item in similar_items if item[0] != 0][:n]
        
        # 根据item_type选择正确的映射
        if item_type == 'user':
            index_to_id = {v: k for k, v in self.user_map.items()}
            id_field = 'user'
        elif item_type == 'track':
            index_to_id = {v: k for k, v in self.track_map.items()}
            id_field = 'track'
        else:
            raise ValueError("Invalid item_type. Must be 'user' or 'track'.")
        
        similar_items_with_ids = [
            {
                id_field: {'$oid': index_to_id[idx]},
                'similarity': float(sim)
            } for idx, sim in similar_items if sim > 0  # 只包含相似度大于0的项
        ]
        
        return similar_items_with_ids
    
    def get_similar_users(self, user_id, n=20):
        user_index = self.get_index_by_id(self.user_map, user_id)
        user_similarities = self.user_similar_matrix[user_index]
        print(user_similarities)
        return self.get_top_n_similar(user_similarities, item_type='user', n=n)
    
    def get_similar_users_tracks(self, user_id, n=20, top_tracks=10):
        # 获取相似用户
        similar_users = self.get_similar_users(user_id, n)
        if not similar_users:
            return []

        # 获取输入用户的索引
        user_index = self.get_index_by_id(self.user_map, user_id)

        # 初始化一个字典来存储每个曲目的加权评分
        weighted_ratings = {}

        # 对于每个相似用户
        for similar_user in similar_users:
            similar_user_id = similar_user['user']['$oid']
            similarity = similar_user['similarity']
            similar_user_index = self.get_index_by_id(self.user_map, similar_user_id)

            # 获取该用户的所有评分
            user_ratings = self.user_track_matrix[similar_user_index]
            

            # 对于该用户评分过的每个曲目
            for track_index, rating in enumerate(user_ratings):
                if rating >= 3:  # 只考虑用户实际评分过3分的曲目
                    track_id = self.get_id_by_index(self.track_map,track_index)
                    print(track_id)
                    # 如果输入用户没有评价过这个曲目
                    if self.user_track_matrix[user_index][track_index] == 0:
                        if track_id not in weighted_ratings:
                            weighted_ratings[track_id] = 0
                        # 加权评分：评分 * 用户相似度
                        weighted_ratings[track_id] += rating * similarity

        # 对加权评分进行排序
        sorted_tracks = sorted(weighted_ratings.items(), key=lambda x: x[1], reverse=True)
        print(f"sorted_tracks: {sorted_tracks}")

        # 返回评分最高的 top_tracks 个曲目
        return [{'track': {'$oid': track_id}, 'score': score} for track_id, score in sorted_tracks[:top_tracks]]

    def get_similar_tracks(self, track_id, n=20):
        track_index = self.get_index_by_id(self.track_map, track_id)
        track_similarities = self.track_similar_matrix[track_index]
        return self.get_top_n_similar(track_similarities, item_type='track', n=n)





    # Helper method

    def get_index_by_id(self, mapping, user_id):
        return mapping.get(user_id)

    def get_id_by_index(self, mapping, index):
        return list(mapping.keys())[index] if 0 <= index < len(mapping) else None
    
    def get_rating(self, user_id, track_id):
        user_idx = self.get_index_by_user_id(self.user_map, user_id)
        track_idx = self.get_index_by_user_id(self.track_map,track_id)
        if user_idx is not None and track_idx is not None:
            return self.user_track_matrix[user_idx, track_idx]
        return None

    def validate_random_ratings(self, num_checks=5):
        print("\nValidating Random Ratings:")
        for _ in range(num_checks):
            rating = random.choice(self.ratings_documents)
            user_id = str(rating['user'])
            track_id = str(rating['item'])
            expected_rating = rating['rate']
            
            matrix_rating = self.get_rating(user_id, track_id)
            
            print(f"User: {user_id}, Track: {track_id}")
            print(f"Expected rating: {expected_rating}")
            print(f"Matrix rating: {matrix_rating}")
            print(f"Match: {'Yes' if expected_rating == matrix_rating else 'No'}")
            print()
        

    # 构建用户-项目评分矩阵
    # 行代表用户，列代表项目，矩阵中的值是用户对项目的评分
    
if __name__ == "__main__":
    input_dir = "collaborate"
    user_map_path = os.path.join(input_dir, 'user_map.json')
    track_map_path = os.path.join(input_dir, 'track_map.json')
    ratings_documents_path = os.path.join(input_dir, 'ratings_documents.json')
    
    
    collaborate_manager = collaborateManager()
    
    collaborate_manager.load_user_map(user_map_path)
    collaborate_manager.load_tracks_map(track_map_path)
    collaborate_manager.load_ratings_documents(ratings_documents_path)
    
    user_id = collaborate_manager.get_index_by_id(collaborate_manager.user_map,0)
    user_index = collaborate_manager.get_index_by_id(collaborate_manager.user_map, '6600f6201d59bf62169dca5e')
    print(f"user_id for index 0: {user_id}")
    print(f"user_index for user_id <6600f6201d59bf62169dca5e> : {user_index} \n")
    
    
    user_track_matrix = collaborate_manager.construct_user_track_matrix()
    print(f"user_track_matrix shape: {user_track_matrix.shape} \n")

    
    user_similarity_matrix = collaborate_manager.construct_user_similar_matrix()
    print(f"user_similarity_matrix shape: {user_similarity_matrix.shape} \n")

    
    track_similarity_matrix = collaborate_manager.construct_track_similar_matrix()
    print(f"track_similarity_matrix shape: {track_similarity_matrix.shape} \n")
    

    
    # 获取与特定用户相似的用户
    user_id = '6600f6201d59bf62169dca5e'
    similar_users = collaborate_manager.get_similar_users(user_id, n=10)
    print(f"similar_users: {similar_users} \n")
    
    # 获取与特定曲目相似的曲目
    track_id = '6678efc77b1dbd108405080e'
    similar_tracks = collaborate_manager.get_similar_tracks(track_id, n=10)
    print(f"similar_tracks: {similar_tracks} \n")

    # 获取用户协同过滤推荐
    recommended_tracks = collaborate_manager.get_similar_users_tracks(user_id, n=20, top_tracks=10)
    print(f"recommended_tracks by similar user`: {recommended_tracks}")

    
    