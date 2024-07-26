# -*- coding: utf-8 -*-
from pymongo import MongoClient
import json
from collections import OrderedDict
import os
import numpy as np
import random

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
        
        self.users_documents = None
        self.tracks_documents = None
        self.ratings_documents = None

        self.user_map = None
        self.track_map = None
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
            
    def load_users_from_db(self):
        try:
            self.users_documents = list(self.users_collection.find({}, {'_id': 1}))
            if not self.users_documents:
                print("No documents found in MongoDB collection.")
            else:
                print(f"Found {len(self.users_documents)} users documents in MongoDB collection.")
            
            self.user_map = OrderedDict((str(doc['_id']), idx) for idx, doc in enumerate(self.users_documents))
            
            with open(os.path.join(self.output_dir, 'user_map.json'), 'w') as f:
                json.dump(list(self.user_map.items()), f)
            print("User map saved")
            
        except Exception as e:
            print(f"Error fetching documents from MongoDB: {e}")
    
    def load_tracks_from_db(self):
        try:
            self.tracks_documents = list(self.tracks_collection.find({}, {'_id': 1}))
            if not self.tracks_documents:
                print("No documents found in MongoDB collection.")
            else:
                print(f"Found {len(self.tracks_documents)} users documents in MongoDB collection.")
            
            self.track_map = OrderedDict((str(doc['_id']), idx) for idx, doc in enumerate(self.tracks_documents))
            
            with open(os.path.join(self.output_dir, 'track_map.json'), 'w') as f:
                json.dump(list(self.track_map.items()), f)
            print("Track map saved")
            
        except Exception as e:
            print(f"Error fetching documents from MongoDB: {e}") 
    
    def load_ratings_from_db(self):
        try:
            self.ratings_documents = list(self.ratings_collection.find({'itemType': 'Track'}, 
                                                                       {'user': 1, 'item':1, 'rate': 1}))
            if not self.ratings_documents:
                print("No documents found in MongoDB collection.")
            else:
                print(f"Found {len(self.ratings_documents)} users documents in MongoDB collection.")

            with open(os.path.join(self.output_dir, 'track_map.json'), 'w') as f:
                json.dump(list(self.track_map.items()), f)
            print("Track map saved")
            
        except Exception as e:
            print(f"Error fetching documents from MongoDB: {e}")

    def construct_user_track_matrix(self):
        # 初始化评分矩阵
        self.rating_matrix = np.zeros((len(self.user_map), len(self.track_map)))

            # 填充评分矩阵
        for rating in self.ratings_documents:
            user_id = str(rating['user'])
            track_id = str(rating['item'])
            user_idx = self.user_map.get(user_id)
            track_idx = self.track_map.get(track_id)
            if user_idx is not None and track_idx is not None:
                self.rating_matrix[user_idx, track_idx] = rating['rate']
        
        return self.rating_matrix




    def load_from_file(self):
        
        try:
            with open(self.file_path, 'r') as f:
                items = json.load(f)
                self.user_map = OrderedDict(items)
            print(f"User map loaded from {self.file_path}")
        except FileNotFoundError:
            print(f"File {self.file_path} not found. Creating a new user map.")
            self.user_map = OrderedDict()
            

    


    # Helper method

    def get_index_by_user_id(self, mapping, user_id):
        return mapping.get(user_id)

    def get_user_id_by_index(self, mapping, index):
        return list(mapping.keys())[index] if 0 <= index < len(mapping) else None
    
    def get_rating(self, user_id, track_id):
        user_idx = self.get_index_by_user_id(self.user_map, user_id)
        track_idx = self.get_index_by_user_id(self.track_map,track_id)
        if user_idx is not None and track_idx is not None:
            return self.rating_matrix[user_idx, track_idx]
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
    collaborate_manager = collaborateManager()
    
    collaborate_manager.load_users_from_db()
    collaborate_manager.load_tracks_from_db()
    collaborate_manager.load_ratings_from_db()
    
    user_id = collaborate_manager.get_user_id_by_index(collaborate_manager.user_map,0)
    user_index = collaborate_manager.get_index_by_user_id(collaborate_manager.user_map, '6600f6201d59bf62169dca5e')
    print(user_id)
    print(user_index)
    
    
    user_track_matrix = collaborate_manager.construct_user_track_matrix()
    print(user_track_matrix)
    print(user_track_matrix.shape)
    
    collaborate_manager.validate_random_ratings()
    
    