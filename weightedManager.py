"""
不保存相似度矩阵,直接计算
"""


from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import os
from preprocessor import Preprocessor
from lda import LDAModelManager
from word2vec import Word2VecManager
from tfidf import TFIDFManager

        
class weightedManager:
    
    def __init__(self,tfidf_manager,w2v_manager,lda_manager,
                 tfidf_weight, w2v_weight, lda_weight):
        self.preprocessor = Preprocessor() # 预处理器
        self.doc_id_to_index_map = tfidf_manager.doc_id_to_index_map # 文档id和索引映射, 用来通过id查找索引
        
        # 三种模型
        self.tfidf_manager = tfidf_manager
        self.w2v_manager = w2v_manager
        self.lda_manager = lda_manager
        
        # 相似度矩阵
        self.tfidf_similarity_matrix = None
        self.w2v_similarity_matrix = None
        self.lda_similarity_matrix = None
        self.weighted_similarity_matrix = None
        self.top_n_similarities = None

        # 默认加权数
        self.tfidf_weight = tfidf_weight
        self.w2v_weight = w2v_weight
        self.lda_weight = lda_weight
        
        # 初始化
        self.validate_weights(tfidf_weight, w2v_weight, lda_weight)
        # self.load_doc_id_to_index_map(doc_id_to_index_map_path) #从文件中加载文档id与索引的映射
        # self.init_manager(tfidf_weight, w2v_weight, lda_weight)
        

        
        
    def load_doc_id_to_index_map(self, doc_id_to_index_map_path):
        """
        从文件中加载文档id与索引的映射
        """
        if os.path.exists(doc_id_to_index_map_path):
            with open(doc_id_to_index_map_path, 'r') as f:
                self.doc_id_to_index_map = json.load(f)
            print("Document Length :", len(self.doc_id_to_index_map))
        else:
            print("Files required did not achieve.")

    def load_similarity_matrix(self, tfidf_similarity_path, w2v_similarity_path, lda_similarity_path):
        """
        从文件中加载计算好的三种模型的相似度矩阵
        """
        if all(os.path.exists(f) for f in [tfidf_similarity_path, w2v_similarity_path, lda_similarity_path]):
            self.tfidf_similarity_matrix = self.rebuild_similarity_matrix(tfidf_similarity_path)
            self.w2v_similarity_matrix = self.rebuild_similarity_matrix(w2v_similarity_path)
            self.lda_similarity_matrix = self.rebuild_similarity_matrix(lda_similarity_path)  
        else:
            print("Files required did not achieve.")
            self.process_and_save_tfidf(self.tfidf_manager.tfidf_matrix)
            self.process_and_save_w2v(self.w2v_manager.song_vectors)
            self.process_and_save_lda(self.lda_manager.doc_topic_matrix)
            
        print("Similarity_matrix Loadded Successful")
    
    def load_weighted_similarity_matrix(self, weighted_similarity_path):
        """
        从文件中加载计算好的相似度矩阵
        """
        if os.path.exists(weighted_similarity_path):
            self.weighted_similarity_matrix = self.rebuild_similarity_matrix(weighted_similarity_path)  
            print(f"Similarity matrix loaded from {weighted_similarity_path}")
            print(f"Matrix shape: {self.weighted_similarity_matrix.shape}")
        else:
            print("Files required did not achieve.")
            self.calculate_and_save_weighted_similarity_matrix(
                tfidf_weight = self.tfidf_weight, 
                w2v_weight = self.w2v_weight, 
                lda_weight = self.lda_weight)
            
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
            
    def load_top_n_weighted_similarity(self, top_n_weighted_similarity_path):
        """
        加载已经计算好的top_n加权相似度
        """
        if os.path.exists(top_n_weighted_similarity_path):
            with open(top_n_weighted_similarity_path, 'r') as f:
                self.top_n_similarities = json.load(f)
        else:
            print("Files required did not achieve.")   
            self.calculate_and_save_top_N_similarity(N)
        
    def validate_weights(self, tfidf_weight, word2vec_weight, lda_weight):
        """
        检查给出的weighting是否总和为1
        """
        if not np.isclose(tfidf_weight + word2vec_weight + lda_weight, 1):
            raise ValueError("The sum of weights must be equal to 1")
            
    def check_doc_id_to_index_map(self):
        """
        检查doc_id_to_index_map是否加载到类实例
        """
        if self.doc_id_to_index_map is None:
            print("doc_id_to_index_map not found")

    def check_similarity_matrices(self):
        """
        检查相似度矩阵是否存在于该实例
        """
        if self.tfidf_similarity_matrix is None or self.w2v_similarity_matrix is None or self.lda_similarity_matrix is None:
            print("Similarity matrices not found, re-calculating")
            self.load_similarity_matrix("tfidf_similarity_matrix.npz", "w2v_similarity_matrix.npz", "lda_similarity_matrix.npz")
        
    def process_and_save_tfidf(self, tfidf_matrix):
        """
        计算并储存TFIDF相似度矩阵
        """
        self.tfidf_similarity_matrix = self.calculate_similarity_matrix(tfidf_matrix, "tfidf")

    def process_and_save_w2v(self, w2v_matrix):
        """
        计算并储存W2V相似度矩阵
        """
        self.w2v_similarity_matrix = self.calculate_similarity_matrix(w2v_matrix, "w2v")

    def process_and_save_lda(self, lda_matrix):
        """
        计算并储存LDA相似度矩阵
        """
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
    
    def calculate_and_save_top_N_similarity(self, N=20):
        """
        对每一个文档计算并储存最大的N个相似度文档为JSON
        """
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
    
    def calculate_and_save_weighted_similarity_matrix(self, tfidf_weight, w2v_weight, lda_weight):
        """
        使用三种模型的相似度举证进行相似度加权 返回加权后的加权相似度矩阵
        """
        # 验证权重
        self.validate_weights(tfidf_weight, w2v_weight, lda_weight)
        
        # 检查必要的数据是否已加载
        self.check_doc_id_to_index_map()
        self.check_similarity_matrices()
        
        # 计算加权相似度矩阵
        self.weighted_similarity_matrix = (
            tfidf_weight * self.tfidf_similarity_matrix +
            w2v_weight * self.w2v_similarity_matrix +
            lda_weight * self.lda_similarity_matrix
        )
        
        print("Weighted Similarity Matrix shape:", self.weighted_similarity_matrix.shape)
        np.savez_compressed("weighted_similarity", matrix=self.weighted_similarity_matrix, allow_pickle=True)
        return self.weighted_similarity_matrix

    def get_weighted_similarity_by_id(self, tfidf_weight, word2vec_weight, lda_weight, doc1, doc2):
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
    
    def get_similar_documents_for_lyrics(self, input_lyrics_list, tfidf_weight=0.33, w2v_weight=0.33, lda_weight=0.34, top_n=20):
        """
        输入一个歌词列表计算这些歌词在三个模型中的平均向量, 再计算各模型中当前平均向量与每个文档的相似度
        对计算到的相似度进行加权, 最终返回top-n个相似度最高的文档
        返回JSON格式
        """
        # 确保输入是一个列表
        if not isinstance(input_lyrics_list, list):
            input_lyrics_list = [input_lyrics_list]
            
        self.validate_weights(tfidf_weight,w2v_weight,lda_weight)
    
        # 预处理输入的歌词列表
        preprocessor = Preprocessor()
        processed_inputs = preprocessor.preprocess_lyrics(input_lyrics_list)
        
    
        # 初始化三种模型的向量列表
        tfidf_vectors = []
        w2v_vectors = []
        lda_vectors = []
        
        for lyrics in processed_inputs:
            if not lyrics.strip():  # 如果歌词为空，跳过
                continue
            
            # TF-IDF 向量
            tfidf_vector = self.tfidf_manager.vectorizer.transform([lyrics]).toarray()[0]
            tfidf_vectors.append(tfidf_vector)
            
            # Word2Vec 向量
            tokens = lyrics.split()
            valid_vectors = [self.w2v_manager.w2v_model.wv[word] for word in tokens if word in self.w2v_manager.w2v_model.wv]
            if valid_vectors:
                w2v_vector = np.mean(valid_vectors, axis=0)
            else:
                w2v_vector = np.zeros(self.w2v_manager.w2v_model.vector_size)
            w2v_vectors.append(w2v_vector)
            
            # LDA 向量
            bow = self.lda_manager.dictionary.doc2bow(tokens)
            lda_vector = [prob for (_, prob) in self.lda_manager.lda_model.get_document_topics(bow, minimum_probability=0)]
            if not lda_vector:
                lda_vector = np.zeros(self.lda_manager.lda_model.num_topics)
            lda_vectors.append(lda_vector)
        
        # 计算每种模型的平均向量，确保不会得到 NaN
        tfidf_avg_vector = np.nanmean(tfidf_vectors, axis=0)
        w2v_avg_vector = np.nanmean(w2v_vectors, axis=0)
        lda_avg_vector = np.nanmean(lda_vectors, axis=0)
        
        # 确保平均向量是 2D 数组
        tfidf_avg_vector = tfidf_avg_vector.reshape(1, -1)
        w2v_avg_vector = w2v_avg_vector.reshape(1, -1)
        lda_avg_vector = lda_avg_vector.reshape(1, -1)
        
        # 分别计算每种模型的相似度
        tfidf_similarities = cosine_similarity(tfidf_avg_vector, self.tfidf_manager.tfidf_matrix)[0]
        w2v_similarities = cosine_similarity(w2v_avg_vector, self.w2v_manager.song_vectors)[0]
        lda_similarities = cosine_similarity(lda_avg_vector, self.lda_manager.doc_topic_matrix)[0]
    
        # 对相似度进行加权
        weighted_similarities = (
            tfidf_weight * tfidf_similarities +
            w2v_weight * w2v_similarities +
            lda_weight * lda_similarities
        )
    
        # 获取相似度最高的top_n个文档的索引
        top_indices = weighted_similarities.argsort()[-top_n:][::-1]
    
        # 准备结果
        similar_documents = []
        for idx in top_indices:
            doc_id = next(id for id, index in self.doc_id_to_index_map.items() if index == idx)
            similar_documents.append({
                "track": {"$oid": doc_id},
                "similarity": float(weighted_similarities[idx])
            })
    
        return similar_documents
    


if __name__ == "__main__":
    N = 20
    tfidf_weight = 0.2
    w2v_weight = 0.4
    lda_weight = 0.4
    
    # Load tfidf model
    tfidf_manager = TFIDFManager()
    tfidf_manager.load_from_file("tfidf")

    # # Load word2vec model
    w2v_manager = Word2VecManager()
    w2v_manager.load_from_file("word2vec")

    # Load lda model
    lda_manager = LDAModelManager()
    lda_manager.load_from_file("lda")
    
    
    weighted_manager = weightedManager(tfidf_manager,w2v_manager,lda_manager, 
                                       tfidf_weight, w2v_weight, lda_weight)

    # Default Use to get similar documents by lyrics.
    lyric="If he's cheatin', I'm doin' him worse (Like) No Uno, I hit the reverse (Grrah) I ain't trippin', the grip in my purse (Grrah) I don't care 'cause he did it first (Like) If he's cheatin', I'm doin' him worse (Damn) I ain't trippin', I— (I ain't trippin', I—) I ain't trippin', the grip in my purse (Like) I don't care 'cause he did it first"
    lyric2="Honey, I'm a good man, but I'm a cheatin' man And I'll do all I can, to get a lady's love And I wanna do right, I don't wanna hurt nobody If I slip, well then I'm sorry, yes I am"
    
    similar_documents = weighted_manager.get_similar_documents_for_lyrics([lyric,lyric2],tfidf_weight, w2v_weight, lda_weight)
    print(similar_documents)
    
    # To Generate Similarity matrix
    # tfidf_similarity_path = "tfidf_similarity_matrix.npz"
    # w2v_similarity_path = "w2v_similarity_matrix.npz"
    # lda_similarity_path = "lda_similarity_matrix.npz"
    # weighted_similarity_path = "weighted_similarity.npz"
    
    
    # weighted_manager.load_similarity_matrix(tfidf_similarity_path, w2v_similarity_path, lda_similarity_path)
    # weighted_manager.load_weighted_similarity_matrix(weighted_similarity_path)
    
    # weighted_similarity_from_matrix = weighted_manager.weighted_similarity_matrix[180][30913]
    # print('Weighted similarity for [180][30913] :', weighted_similarity_from_matrix)
        
    # weighted_similarity = weighted_manager.get_weighted_similarity_by_id(tfidf_weight, w2v_weight, lda_weight,'65ffbfa9c1ab936c978e4dad','66858f1bc8fd49c0eaff1904')
    # print('Manual Calculated Weighted similarity for <65ffbfa9c1ab936c978e4dad>[180] and <66858f1bc8fd49c0eaff1904>[30913] :', weighted_similarity)
    