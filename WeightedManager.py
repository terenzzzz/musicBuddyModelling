from scipy.sparse import csr_matrix, save_npz, lil_matrix
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import numpy as np
import multiprocessing
import json
import os
import tqdm


# Define process_chunk function outside of the class
def process_chunk(start_row, end_row, tfidf_similarity_matrix, w2v_similarity_matrix, lda_similarity_matrix,
                  tfidf_weight, word2vec_weight, lda_weight, result_queue):
    weighted_chunk = (
        tfidf_weight * tfidf_similarity_matrix[start_row:end_row, :] +
        word2vec_weight * w2v_similarity_matrix[start_row:end_row, :] +
        lda_weight * lda_similarity_matrix[start_row:end_row, :]
    )
    result_queue.put((start_row, end_row, weighted_chunk))
        
        

class WeightedManager:
    
    def __init__(self):
        self.doc_id_to_index_map = None
        self.tfidf_matrix = None
        self.w2v_matrix = None
        self.lda_matrix = None
        
        self.tfidf_similarity_matrix = None
        self.w2v_similarity_matrix = None
        self.lda_similarity_matrix = None
        
        self.weighted_similarity_matrix = None
        self.top_n_similarities = None
        
        
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
    
    def load_weighted_similarity_matrix(self, weighted_similarity_path):
        
        if os.path.exists(weighted_similarity_path):
            self.weighted_similarity_matrix = self.rebuild_similarity_matrix(weighted_similarity_path)
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
        loaded = np.load(filename, allow_pickle=True)
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

            
    def load_top_n_weighted_similarity_matrix(self, top_n_weighted_similarity_path):
        if os.path.exists(top_n_weighted_similarity_path):
            with open(top_n_weighted_similarity_path, 'r') as f:
                self.top_n_similarities = json.load(f)
        else:
            print("Files required did not achieve.")


    def check_similarity_matrices(self):
        if self.tfidf_similarity_matrix is None or self.w2v_similarity_matrix is None or self.lda_similarity_matrix is None:
            print("Similarity matrices not found, re-calculating")


    def process_and_save_tfidf(self, tfidf_matrix):
        self.tfidf_similarity_matrix = self.calculate_similarity_matrix(tfidf_matrix, "tfidf")


    def process_and_save_w2v(self, w2v_matrix):
        self.w2v_similarity_matrix = self.calculate_similarity_matrix(w2v_matrix, "w2v")


    def process_and_save_lda(self, lda_matrix):
        self.lda_similarity_matrix = self.calculate_similarity_matrix(lda_matrix, "lda")
    
    def calculate_similarity_matrix(self, matrix, name):
        """
        输入矩阵和名字
        输出该矩阵的相似度矩阵
        只保留上三角矩阵并转换为float16以节省空间
        """
        print(f"Calculating {name} Similarities Matrix")
        sim_matrix = cosine_similarity(matrix)
        print(f"{name} Similarity Matrix shape:", sim_matrix.shape)
        print(f"{name}_similarity_matrix[0][1]:", sim_matrix[0][1])
        
        # 只保留上三角矩阵（包括对角线）
        # upper_tri = np.triu(sim_matrix)
        
        # 转换为float16以节省空间
        # upper_tri = upper_tri.astype(np.float16)
        sim_matrix = sim_matrix.astype(np.float16)
        
        # # 获取上三角部分的索引
        # i, j = np.triu_indices_from(matrix)
        # # 只保存上三角部分的值和矩阵的大小
        # np.savez_compressed(filename, data=matrix[i, j], shape=matrix.shape)
        
        np.savez_compressed(f"{name}_similarity_matrix.npz", matrix=sim_matrix, allow_pickle=True)
        print(f"Similarity matrix saved to {name}_similarity_matrix.npz")
        
        return sim_matrix
    
    def calculate_top_N_similarity(self, N=20):
        if not hasattr(self, 'weighted_similarity_matrix'):
            raise AttributeError("weighted_similarity_matrix has not been calculated yet.")
    
        num_docs = self.weighted_similarity_matrix.shape[0]
    
        self.top_n_similarity = []
    
        np.fill_diagonal(self.weighted_similarity_matrix, -np.inf)
    
        top_n_indices = np.argpartition(-self.weighted_similarity_matrix, N, axis=1)[:, :N]
    
        # 创建索引到 ID 的反向映射
        index_to_id_map = {index: id for id, index in self.doc_id_to_index_map.items()}
    
        for i in range(num_docs):
            top_indices = top_n_indices[i]
            
            top_similar_docs = [
                {
                    "track": {"$oid": str(index_to_id_map[idx])},
                    "value": float(self.weighted_similarity_matrix[i, idx])
                }
                for idx in top_indices
            ]
            
            top_similar_docs.sort(key=lambda x: x['value'], reverse=True)
            
            self.top_n_similarity.append({
                "track": {"$oid": str(index_to_id_map[i])},
                "topsimilar": top_similar_docs
            })
    
        np.fill_diagonal(self.weighted_similarity_matrix, 1)
        
        with open('top_weighted_similarity.json', 'w') as f:
            json.dump(self.top_n_similarity, f, indent=2)
            
        print(f"\nTotal number of documents processed: {len(self.top_n_similarity)}")
        print(f"Sample of final result (first item): {self.top_n_similarity[0]}")
    
        return self.top_n_similarity 


    
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
        np.savez_compressed("weighted_similarity", matrix=self.weighted_similarity_matrix, allow_pickle=True)
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
    N = 20
    tfidf_weight = 0.2
    w2v_weight = 0.4
    lda_weight = 0.4
    weighted_manager = WeightedManager()
    
    # Load matrix
    if all(os.path.exists(f) for f in ['tfidf/tfidf_matrix.pkl', 'word2vec/song_vectors.npy', 'lda/lda_matrix.npy']):
        weighted_manager.load_matrix('tfidf/tfidf_matrix.pkl', 'word2vec/song_vectors.npy','lda/lda_matrix.npy')
        
    # Load ids mapping
    if os.path.exists('tfidf/doc_id_to_index_map.json'):
        weighted_manager.load_doc_id_to_index_map('tfidf/doc_id_to_index_map.json')
         
        
    # Load similarity matrix
    if all(os.path.exists(f) for f in ["tfidf_similarity_matrix.npz", "w2v_similarity_matrix.npz", "lda_similarity_matrix.npz"]):
        weighted_manager.load_similarity_matrix("tfidf_similarity_matrix.npz", "w2v_similarity_matrix.npz", "lda_similarity_matrix.npz")
    else:
        print("Files required not achieved, recalculating similarity matrix")
        weighted_manager.process_and_save_tfidf(weighted_manager.tfidf_matrix)
        weighted_manager.process_and_save_w2v(weighted_manager.w2v_matrix)
        weighted_manager.process_and_save_lda(weighted_manager.lda_matrix)
    
    # Load weighted similarity matrix
    if os.path.exists('weighted_similarity.npz'):
        weighted_manager.load_weighted_similarity_matrix('weighted_similarity.npz')
    else:
        print("Files required not achieved, recalculating weighted similarity matrix'")
        weighted_manager.get_weighted_similarity_matrix(tfidf_weight, w2v_weight, lda_weight)
        
    # Load top N Similar Weighted Similarity
    if os.path.exists('top_weighted_similarity.json'):
        weighted_manager.load_top_n_weighted_similarity_matrix('top_weighted_similarity.json')

    else:
        print("Files required not achieved, recalculating top N weighted similarity matrix'")
        weighted_manager.calculate_top_N_similarity(N)
        
        
        
    
    weighted_similarity_from_matrix = weighted_manager.weighted_similarity_matrix[180][30913]
    print('Weighted similarity for [180][30913] :', weighted_similarity_from_matrix)
        
    weighted_similarity = weighted_manager.get_weighted_similarity(tfidf_weight, w2v_weight, lda_weight,'65ffbfa9c1ab936c978e4dad','66858f1bc8fd49c0eaff1904')
    print('Manual Calculated Weighted similarity for <65ffbfa9c1ab936c978e4dad>[180] and <66858f1bc8fd49c0eaff1904>[30913] :', weighted_similarity)