import numpy as np
import pickle
import os
import pandas as pd
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import nomalizor
import json
from tqdm import tqdm

def load_mongo_and_train(N=10):
    # Connect to MongoDB
    client = MongoClient('mongodb://localhost:27017/')
    db = client['MusicBuddyVue']
    collection = db['tracks']
    
    try:
        # Fetch lyric documents from MongoDB
        lyrics_documents = list(collection.find())
        if not lyrics_documents:
            print("No documents found in MongoDB collection.")
            return None, None, None, None, None, None
        else:
            print(f"Found {len(lyrics_documents)} documents in MongoDB collection.")
    except Exception as e:
        print(f"Error fetching documents from MongoDB: {e}")
        return None, None, None, None, None, None
    
    # Extract lyrics and preprocess them
    lyrics = [doc.get('lyric', '') for doc in lyrics_documents if isinstance(doc.get('lyric', None), str)]
    processed_lyrics = nomalizor.preprocess_lyrics(lyrics)

    # Calculate TF-IDF using TfidfVectorizer
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_lyrics)
    
    # Calculate cosine similarities
    cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Get feature names (vocabulary)
    feature_names = vectorizer.get_feature_names_out()
    
    # Convert _id to string for document indexing
    doc_ids = [str(doc['_id']) for doc in lyrics_documents]
    
    # Create DataFrame for visualization
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names, index=doc_ids)

    # Save TF-IDF matrix and feature names
    with open('tfidf/tfidf_matrix.pkl', 'wb') as f:
        pickle.dump(tfidf_matrix, f)
    
    # Sort indices to get top N similarities
    top_n_similarities = np.argsort(cosine_similarities, axis=1)[:, -N-1:-1]
    
    # Compute top N similarities for each document and save as JSON
    top_similarities_per_doc = []
    for i, doc_id in enumerate(tqdm(doc_ids, desc="Preparing JSON for Top Similarities")):
        similar_doc_indices = top_n_similarities[i]
        similar_docs = [
            {
                "$oid": doc_ids[idx],
                "value": float(cosine_similarities[i, idx])  # 直接从 cosine_similarities 获取值
            }
            for idx in similar_doc_indices
        ]
        doc_data = {
            "track": {"$oid": doc_id},
            "topsimilar": similar_docs
        }
        top_similarities_per_doc.append(doc_data)
    
    # Save top similarities in JSON format
    with open('tfidf/top_similarities.json', 'w') as f:
        json.dump(top_similarities_per_doc, f, indent=2)
        
    # Save feature names
    with open('tfidf/feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
    
    # Save document IDs
    with open('tfidf/doc_ids.pkl', 'wb') as f:
        pickle.dump(doc_ids, f)
        
    # Save TfidfVectorizer model
    joblib.dump(vectorizer, 'tfidf/tfidf_vectorizer.joblib')
    
    # Compute top N keywords for each document
    top_keywords_per_doc = []
    for i, doc_id in enumerate(doc_ids):
        tfidf_vector = tfidf_matrix[i]
        sorted_indices = tfidf_vector.toarray().argsort()[0][::-1][:N]
        top_keywords = [{"word": feature_names[idx], "value": tfidf_vector[0, idx]} for idx in sorted_indices]
        doc_data = {
            "track": {"$oid": doc_id},
            "topwords": top_keywords
        }
        top_keywords_per_doc.append(doc_data)
    
    # Save top keywords in JSON format
    with open('tfidf/top_keywords.json', 'w') as f:
        json.dump(top_keywords_per_doc, f, indent=2)
        
    print("TF-IDF results saved successfully.")
    
    return tfidf_matrix, top_similarities_per_doc, feature_names, doc_ids, vectorizer, tfidf_df, top_keywords_per_doc

def load_from_file():
    try:
        # 加载 TF-IDF 矩阵和特征名称
        with open('tfidf/tfidf_matrix.pkl', 'rb') as f:
            tfidf_matrix = pickle.load(f)
            
        
        with open('tfidf/feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        
        # 加载文档 ID
        with open('tfidf/doc_ids.pkl', 'rb') as f:
            doc_ids = pickle.load(f)
        
        # 加载 TfidfVectorizer 模型
        vectorizer = joblib.load('tfidf/tfidf_vectorizer.joblib')
        
        # 转换为DataFrame以便查看，使用 track._id 作为索引
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names, index=doc_ids)
        
        # Load top keywords from JSON file
        with open('tfidf/top_keywords.json', 'r') as f:
            top_keywords_per_doc = json.load(f)
            
        # Load top similarity from JSON file
        with open('tfidf/top_similarities.json', 'r') as f:
            top_similarities_per_doc = json.load(f)

        # Debug prints to check contents
        print("TF-IDF matrix shape:", tfidf_matrix.shape)
        print("Top N similarities shape:", len(top_similarities_per_doc))
        print("Feature names:", feature_names[:10])  # Print first 10 feature names
        print("Document IDs sample:", doc_ids[:10])  # Print first 10 document IDs


        return tfidf_matrix, top_similarities_per_doc, feature_names, doc_ids, vectorizer, tfidf_df, top_keywords_per_doc
    except FileNotFoundError as e:
        print(f"Error loading from files: {e}")
        return None
    
def get_similar_documents(doc_id, top_similarities_per_doc):
    similar_documents = []
    for item in top_similarities_per_doc:
        if item['track']['$oid'] == doc_id:
            similar_documents.extend([doc['$oid'] for doc in item['topsimilar']])
            break
    return similar_documents

def get_top_words(doc_id, top_keywords_per_doc):
    top_words = None
    for item in top_keywords_per_doc:
        if item['track']['$oid'] == doc_id:
            top_words = item['topwords']
            break
    return top_words









if __name__ == "__main__":
    if os.path.exists('tfidf/tfidf_matrix.pkl') and os.path.exists('tfidf/top_similarities.json') and os.path.exists('tfidf/top_keywords.json') and os.path.exists('tfidf/feature_names.pkl') and os.path.exists('tfidf/doc_ids.pkl') and os.path.exists('tfidf/tfidf_vectorizer.joblib'):
        print("Loading TF-IDF results from files...")
        tfidf_matrix, top_similarities_per_doc, feature_names, doc_ids, vectorizer, tfidf_df, top_keywords_per_doc = load_from_file()
    else:
        print("Loading data from MongoDB and training TF-IDF model...")
        tfidf_matrix, top_similarities_per_doc, feature_names, doc_ids, vectorizer, tfidf_df, top_keywords_per_doc = load_mongo_and_train()
    
    if tfidf_df is not None:
        print("TF-IDF DataFrame is ready.")
        

        
        query_doc_id = '65ffc183c1ab936c978f29a8'
        top_words = get_top_words(query_doc_id, top_keywords_per_doc)
        similar_docs = get_similar_documents(query_doc_id, top_similarities_per_doc)
        print(f"Similar documents for document {query_doc_id}:")
        print(similar_docs)
        
        print(f"Top words for document {query_doc_id}:")
        print(top_words)
        
        
        
        

