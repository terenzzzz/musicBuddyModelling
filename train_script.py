# -*- coding: utf-8 -*-
from preprocessor import Preprocessor
from lda import LDAModelManager
from word2vec import Word2VecManager
from tfidf import TFIDFManager
from artist import ArtistManager
from pymongo import MongoClient
from weightedManager import weightedManager
import os

if __name__ == "__main__":
    
    client = MongoClient('mongodb://localhost:27017/')
    db = client['MusicBuddyVue']
    tracks_collection = db['tracks']
    
    
    # Preprocess
    print("Handling Preprocess DATA......")
    preprocessor = Preprocessor()
    if not os.path.exists("processed_lyrics.txt"):
        preprocessor.load_data('mongodb://localhost:27017/', 'MusicBuddyVue', 'tracks')
        lyrics = [doc.get('lyric', '') for doc in preprocessor.tracks_documents if isinstance(doc.get('lyric', None), str)]
        processed_lyrics = preprocessor.preprocess_lyrics(lyrics)
        with open("processed_lyrics.txt", 'w', encoding='utf-8') as f:
            for lyrics in processed_lyrics:
                f.write(lyrics + '\n')
        print(f"Saved {len(processed_lyrics)} documents to processed_lyrics.txt")
    
    
    # Load tfidf model
    print("Handling TFIDF MODEL......")
    tfidf_manager = TFIDFManager()
    tfidf_input_dir = 'tfidf'  # or other directory where you save files
    if not (os.path.exists(os.path.join(tfidf_input_dir, 'tfidf_matrix.pkl')) and
        os.path.exists(os.path.join(tfidf_input_dir, 'top_similarities.json')) and
        os.path.exists(os.path.join(tfidf_input_dir, 'top_keywords.json')) and
        os.path.exists(os.path.join(tfidf_input_dir, 'feature_names.pkl')) and
        os.path.exists(os.path.join(tfidf_input_dir, 'doc_id_to_index_map.json')) and
        os.path.exists(os.path.join(tfidf_input_dir, 'tfidf_vectorizer.joblib'))):
    
        print("Training TF-IDF model...")
        tfidf_manager.load_mongo_and_train()
        print("TF-IDF model trained successful!")
    tfidf_manager.load_from_file("tfidf")
        
    # Load W2V model
    print("Handling W2V model......")
    w2v_manager = Word2VecManager()
    w2v_input_dir = 'word2vec'
    if not all(os.path.exists(os.path.join(w2v_input_dir, f)) for f in ['song_vectors.npy', 'w2v_model.model', 'top_similarities.json', 'doc_id_to_index_map.json']):
        print("Training W2V model...")
        w2v_manager.load_mongo_and_train()
        print("W2V model trained successful!")
    w2v_manager.load_from_file("word2vec")
        
    # Load Lda model
    print("Handling LDA model......")
    lda_manager = LDAModelManager()
    num_topics = 20
    lda_input_dir = 'lda'

    # 检查是否存在已保存的模型文件
    if not all(os.path.exists(os.path.join(lda_input_dir, f)) for f in ['dictionary.gensim', 'corpus.mm', 'lda_model.gensim', 'texts.txt']):
        lda_manager.load_mongo_and_train(num_topics=num_topics)
        print("Training LDA model...")
        lda_manager.load_mongo_and_train(num_topics=num_topics)
        print("LDA model trained successful!")
    lda_manager.load_from_file("lda")
        
    # Load Weighted model
    print("Handling Weighted model......")
    N = 20
    tfidf_weight = 0.2
    w2v_weight = 0.4
    lda_weight = 0.4
    
    weighted_manager = weightedManager(tfidf_manager,w2v_manager,lda_manager, 
                                        tfidf_weight, w2v_weight, lda_weight, 
                                        "tfidf/doc_id_to_index_map.json")
    
    # To Generate Similarity matrix
    tfidf_similarity_path = "tfidf_similarity_matrix.npz"
    w2v_similarity_path = "w2v_similarity_matrix.npz"
    lda_similarity_path = "lda_similarity_matrix.npz"
    weighted_similarity_path = "weighted_similarity.npz"
    
    
    weighted_manager.load_similarity_matrix(tfidf_similarity_path, w2v_similarity_path, lda_similarity_path)
    weighted_manager.load_weighted_similarity_matrix(weighted_similarity_path)
    
    # Load Artist
    print("Handling Artist ......")
    artist_manager = ArtistManager(tfidf_manager,w2v_manager,lda_manager)
    artist_manager.load_tfidf_matrix()
    artist_manager.load_w2v_matrix()
    artist_manager.load_lda_matrix()