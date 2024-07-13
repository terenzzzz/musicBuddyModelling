from scipy.sparse import csr_matrix, save_npz, lil_matrix
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import numpy as np
import multiprocessing
import json
import os


# Define process_chunk function outside of the class
def process_chunk(start_row, end_row, tfidf_similarity_matrix, w2v_similarity_matrix, lda_similarity_matrix,
                  tfidf_weight, word2vec_weight, lda_weight, result_queue):
    weighted_chunk = (
        tfidf_weight * tfidf_similarity_matrix[start_row:end_row, :] +
        word2vec_weight * w2v_similarity_matrix[start_row:end_row, :] +
        lda_weight * lda_similarity_matrix[start_row:end_row, :]
    )
    result_queue.put((start_row, end_row, weighted_chunk))
        
        

class WeightedCalculator:
    
    def __init__(self):
        self.doc_id_to_index_map = None
        self.tfidf_matrix = None
        self.w2v_matrix = None
        self.lda_matrix = None
        
        self.tfidf_similarity_matrix = None
        self.w2v_similarity_matrix = None
        self.lda_similarity_matrix = None
        
        self.weighted_similarity_matrix = None
        
        
    def load_matrix(self, tfidf_matrix_path, w2v_matrix_path, lda_matrix_path):
        if all(os.path.exists(f) for f in [tfidf_matrix_path, w2v_matrix_path, lda_matrix_path]):
            with open(tfidf_matrix_path, 'rb') as f:
                self.tfidf_matrix = pickle.load(f)
            self.w2v_matrix = np.load(w2v_matrix_path)
            self.lda_matrix = np.load(lda_matrix_path)
            print("TF-IDF shape:", self.tfidf_matrix.shape)
            print("W2V shape:", self.w2v_matrix.shape)
            print("LDA shape:", self.lda_matrix.shape)
        else:
            print("Files required did not achieve.")
        
    def load_doc_id_to_index_map(self, doc_id_to_index_map_path):
        if os.path.exists(doc_id_to_index_map_path):
            with open(doc_id_to_index_map_path, 'r') as f:
                self.doc_id_to_index_map = json.load(f)
            print("Document Length :", len(self.doc_id_to_index_map))
        else:
            print("Files required did not achieve.")
        


    def load_similarity_matrix(self, tfidf_similarity_path, w2v_similarity_path, lda_similarity_path):
        if all(os.path.exists(f) for f in [tfidf_similarity_path, w2v_similarity_path, lda_similarity_path]):
            self.tfidf_similarity_matrix = self.rebuild_similarity_matrix(tfidf_similarity_path)
            self.w2v_similarity_matrix = self.rebuild_similarity_matrix(w2v_similarity_path)
            self.lda_similarity_matrix = self.rebuild_similarity_matrix(lda_similarity_path)  
        else:
            print("Files required did not achieve.")
        
    def rebuild_similarity_matrix(self, filename):
        # loaded = np.load(filename)
        # data = loaded['data']
        # shape = loaded['shape']
        
        # # 重建完整矩阵
        # matrix = np.zeros(shape, dtype=np.float16)
        # i, j = np.triu_indices_from(matrix)
        # matrix[i, j] = data
        # matrix = matrix + matrix.T - np.diag(np.diag(matrix))
        loaded = np.load(filename)
        matrix = loaded['matrix']
        
        print(f"Similarity matrix loaded from {filename}")
        print(f"Matrix shape: {matrix.shape}")
        
        return matrix
        
    def validate_weights(self, tfidf_weight, word2vec_weight, lda_weight):
        if not np.isclose(tfidf_weight + word2vec_weight + lda_weight, 1):
            raise ValueError("The sum of weights must be equal to 1")
            
    def check_doc_id_to_index_map(self):
        if self.doc_id_to_index_map is None:
            print("doc_id_to_index_map not found")

            
    def check_similarity_matrices(self):
        if self.tfidf_similarity_matrix is None or self.w2v_similarity_matrix is None or self.lda_similarity_matrix is None:
            print("Similarity matrices not found, re-calculating")
            self.get_similarity_matrix()
    
    def calculate_similarity_matrix(self, matrix, name):
        """
        输入矩阵和名字
        输出该矩阵的相似度矩阵
        只保留上三角矩阵并转换为float16以节省空间
        """
        print(f"Calculating {name} Similarities Matrix")
        sim_matrix = cosine_similarity(matrix)
        print(f"{name} Similarities Matrix shape:", sim_matrix.shape)
        print(f"{name}_similarities_matrix[0][1]:", sim_matrix[0][1])
        
        # 只保留上三角矩阵（包括对角线）
        # upper_tri = np.triu(sim_matrix)
        
        # 转换为float16以节省空间
        # upper_tri = upper_tri.astype(np.float16)
        sim_matrix = sim_matrix.astype(np.float16)
        
        return sim_matrix

    def save_similarity_matrix(self, matrix, filename):
        # # 获取上三角部分的索引
        # i, j = np.triu_indices_from(matrix)
        # # 只保存上三角部分的值和矩阵的大小
        # np.savez_compressed(filename, data=matrix[i, j], shape=matrix.shape)
        
        np.savez_compressed(filename, matrix=matrix)
        print(f"Similarity matrix saved to {filename}")



    def process_and_save_tfidf(self, tfidf_matrix, filename):
        sim_matrix = self.calculate_similarity_matrix(tfidf_matrix, "TF-IDF")
        self.save_similarity_matrix(sim_matrix, filename)

    def process_and_save_w2v(self, w2v_matrix, filename):
        sim_matrix = self.calculate_similarity_matrix(w2v_matrix, "Word2Vec")
        self.save_similarity_matrix(sim_matrix, filename)

    def process_and_save_lda(self, lda_matrix, filename):
        sim_matrix = self.calculate_similarity_matrix(lda_matrix, "LDA")
        self.save_similarity_matrix(sim_matrix, filename)
    
    def get_weighted_similarity_matrix(self, tfidf_weight, w2v_weight, lda_weight):
        # 验证权重
        self.validate_weights(tfidf_weight, w2v_weight, lda_weight)
        
        # 检查必要的数据是否已加载
        self.check_doc_id_to_index_map()
        self.check_similarity_matrices()
        
        # 确保所有相似度矩阵都已加载
        if self.tfidf_similarity_matrix is None or self.w2v_similarity_matrix is None or self.lda_similarity_matrix is None:
            raise ValueError("All similarity matrices must be loaded before calculating weighted similarity")
        
        # 计算加权相似度矩阵
        self.weighted_similarity_matrix = (
            tfidf_weight * self.tfidf_similarity_matrix +
            w2v_weight * self.w2v_similarity_matrix +
            lda_weight * self.lda_similarity_matrix
        )
        
        print("Weighted Similarity Matrix shape:", self.weighted_similarity_matrix.shape)
        np.savez_compressed("weighted_similarity", matrix=self.weighted_similarity_matrix)
        return self.weighted_similarity_matrix

    
    def get_weighted_similarity(self, tfidf_weight, word2vec_weight, lda_weight, doc1, doc2):
        """
        输入权重和两个文档的id
        输出两个文档的加权相似度
        """
        # Validate the weights
        self.validate_weights(tfidf_weight, word2vec_weight, lda_weight)
        
        self.check_doc_id_to_index_map()
        
        self.check_similarity_matrices()
        
        doc1_index = self.doc_id_to_index_map.get(doc1)
        doc2_index = self.doc_id_to_index_map.get(doc2)
        
        print("doc1_index: ", doc1_index)
        print("doc2_index: ", doc2_index)
        
        tfidf_similarity = self.tfidf_similarity_matrix[doc1_index][doc2_index]
        w2v_similarity = self.w2v_similarity_matrix[doc1_index][doc2_index]
        lda_similarity = self.lda_similarity_matrix[doc1_index][doc2_index]
        
        print("tfidf_similarity: ", tfidf_similarity)
        print("w2v_similarity: ", w2v_similarity)
        print("lda_similarity: ", lda_similarity)
        
        
        
        weighted_similarity = (tfidf_weight * tfidf_similarity + 
                               word2vec_weight * w2v_similarity + 
                               lda_weight * lda_similarity)
             
        return weighted_similarity


if __name__ == "__main__":
    k = 20
    
    weightedCalculator = WeightedCalculator()
    weightedCalculator.load_matrix('tfidf/tfidf_matrix.pkl', 'word2vec/song_vectors.npy','lda/lda_matrix.npy')
    weightedCalculator.load_doc_id_to_index_map('tfidf/doc_id_to_index_map.json')
                                   
                                 
    
    # print('Weighted similarity for <65ffbfa9c1ab936c978e4dad> and <65ffb8b4c1ab936c978b016c> :',weighted_similarity)
    

    # weightedCalculator.process_and_save_tfidf(weightedCalculator.tfidf_matrix, "tfidf_similarity.npz")
    # weightedCalculator.process_and_save_w2v(weightedCalculator.w2v_matrix, "w2v_similarity.npz")
    # weightedCalculator.process_and_save_lda(weightedCalculator.lda_matrix, "lda_similarity.npz")
    
    weightedCalculator.load_similarity_matrix("tfidf_similarity.npz", "w2v_similarity.npz", "lda_similarity.npz")
    print("Testing...")
    print(f'{weightedCalculator.tfidf_similarity_matrix[50][100]} should equal to {weightedCalculator.tfidf_similarity_matrix[100][50]}')
    print(f'{weightedCalculator.w2v_similarity_matrix[50][100]} should equal to {weightedCalculator.w2v_similarity_matrix[100][50]}')
    print(f'{weightedCalculator.lda_similarity_matrix[50][100]} should equal to {weightedCalculator.lda_similarity_matrix[100][50]}')
    
    
    weightedCalculator.get_weighted_similarity_matrix(0.2, 0.4, 0.4)
    print(f'Weighted similarity for [50][100]: {weightedCalculator.weighted_similarity_matrix[50][100]}')