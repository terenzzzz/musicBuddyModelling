import pickle
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from sklearn.decomposition import TruncatedSVD
import faiss
import os
import json

def load_data():
    with open('tfidf/tfidf_matrix.pkl', 'rb') as f:
        tfidf_matrix = pickle.load(f)
    w2v_matrix = np.load('word2vec/song_vectors.npy')
    print("TF-IDF shape:", tfidf_matrix.shape)
    print("W2V shape:", w2v_matrix.shape)
    return tfidf_matrix, w2v_matrix

def normalize_matrix(matrix):
    return matrix / np.linalg.norm(matrix, axis=1)[:, np.newaxis]

def build_faiss_index(matrix):
    d = matrix.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(matrix.astype('float32'))
    return index

def compute_similarities_for_chunk(args):
    start, end, tfidf_index, w2v_index, tfidf_matrix_norm, w2v_matrix_norm, tfidf_weight, w2v_weight, k = args
    chunk_similarities = []
    
    for i in range(start, end):
        tfidf_similarities, tfidf_indices = tfidf_index.search(tfidf_matrix_norm[i:i+1], k+1)
        w2v_similarities, w2v_indices = w2v_index.search(w2v_matrix_norm[i:i+1], k+1)
        
        weighted_similarities = tfidf_weight * tfidf_similarities[0] + w2v_weight * w2v_similarities[0]
        top_indices = np.argsort(weighted_similarities)[::-1][1:k+1]  # Exclude self
        
        top_k_similar_docs = [(weighted_similarities[j], tfidf_indices[0][j]) for j in top_indices]
        chunk_similarities.append((i, top_k_similar_docs))
    
    return chunk_similarities

def compute_top_k_similar_documents(tfidf_matrix, w2v_matrix, tfidf_weight=0.5, w2v_weight=0.5, num_components=500, k=20):
    # Reduce TF-IDF dimensionality
    svd = TruncatedSVD(n_components=num_components, random_state=42)
    tfidf_matrix_reduced = svd.fit_transform(tfidf_matrix)
    
    # Normalize matrices
    tfidf_matrix_norm = normalize_matrix(tfidf_matrix_reduced)
    w2v_matrix_norm = normalize_matrix(w2v_matrix)
    
    # Build FAISS indices
    tfidf_index = build_faiss_index(tfidf_matrix_norm)
    w2v_index = build_faiss_index(w2v_matrix_norm)
    
    num_docs = tfidf_matrix.shape[0]
    num_processes = cpu_count()
    chunk_size = max(1, num_docs // num_processes)
    chunks = [(i * chunk_size, min((i + 1) * chunk_size, num_docs)) for i in range(num_processes)]
    
    args = [(start, end, tfidf_index, w2v_index, tfidf_matrix_norm, w2v_matrix_norm, tfidf_weight, w2v_weight, k) for start, end in chunks]
    
    with Pool(num_processes) as pool:
        results = list(tqdm(pool.imap(compute_similarities_for_chunk, args), total=len(args), desc="计算相似度"))
    
    top_k_similarities = [item for sublist in results for item in sublist]
    
    # 使用字典存储结果
    weighted_similarities = {}
    for i, similarities in top_k_similarities:
        weighted_similarities[i] = similarities
        
    # 加载 doc_ids.pkl
    with open('tfidf/doc_ids.pkl', 'rb') as f:
        doc_ids = pickle.load(f)
    
    # 准备 MongoDB 可导入的 JSON 格式
    top_similarities_json = []
    for i in tqdm(range(len(weighted_similarities)), desc="准备 MongoDB JSON"):
        doc_id = doc_ids[i]
        top_similar_docs = [
            {
                "track": {"$oid": doc_ids[idx]},
                "value": float(similarity)  # 转换为 float 以确保 JSON 兼容性
            }
            for similarity, idx in weighted_similarities[i]
        ]
        
        top_similarities_json.append({
            "track": {"$oid": doc_id},
            "topsimilar": top_similar_docs
        })
    
    # 保存为 JSON 文件
    with open(f"top_{k}_similarities_for_mongodb.json", 'w') as file:
        json.dump(top_similarities_json, file)
    print(f"结果已保存为 'top_{k}_similarities_for_mongodb.json'")
    
    return weighted_similarities

if __name__ == "__main__":
    k = 20
    
    if os.path.exists(f"top_{k}_weighted_similarities.pkl"):
        print(f"Loading weighted top {k} similarities from files top_{k}_weighted_similarities.pkl")
        with open(f'top_{k}_weighted_similarities.pkl', 'rb') as f:
            top_k_weighted_similarities = pickle.load(f)
    else:
        print(f"Loading TFIDF matrix and Word2Vec matrix from file and calculating top {k} similarities")
        tfidf_matrix, word2vec_matrix = load_data()
        tfidf_weight = 0.5
        word2vec_weight = 0.5
        top_k_weighted_similarities = compute_top_k_similar_documents(tfidf_matrix, word2vec_matrix, tfidf_weight, word2vec_weight, k=k)
    
    if top_k_weighted_similarities is not None:
        print(f"top_{k}_weighted_similarities is ready.")
    

    
        # print("Top", k, "Weighted Similarities Matrix Shape:", top_k_weighted_similarities.shape)
        print(f"计算完成。每个文档的 top {k} 相似文档已找到。")
        print(f"结果包含 {len(top_k_weighted_similarities)} 个文档的相似性信息。")
        
        print(top_k_weighted_similarities[0])
    
    
    
    
    
    
    
    
    
    
    
# 降维和索引构建:

# 对TF-IDF矩阵进行降维处理。
# 对TF-IDF和Word2Vec矩阵进行归一化。
# 使用FAISS库为这两个矩阵构建索引。这一步是为了加速后续的相似度搜索。


# 并行处理:

# 将文档集分成几个块,每个块由一个独立的进程处理。


# 对每个文档:

# 使用FAISS索引快速找出TF-IDF和Word2Vec表示下最相似的k+1个文档(包括文档自身)。
# 将TF-IDF和Word2Vec的相似度结果进行加权组合。
# 排序并选择前k个最相似的文档(排除文档自身)。


# 结果整合:

# 收集所有进程的结果,形成最终的相似度信息。



# 这种方法的关键在于:

# 它没有计算所有文档对之间的相似度,而是利用FAISS索引快速找出每个文档最相似的k个文档。
# 通过并行处理,大大提高了计算效率。
# 只保存了每个文档的前k个最相似文档的信息,而不是完整的相似度矩阵。

# 这种方法的优点是:

# 效率高:不需要计算完整的相似度矩阵。
# 内存友好:只存储每个文档的top k个相似文档,大大减少了内存使用。
# 适用于大规模数据:即使对于非常大的文档集,也能高效地找出每个文档的最相似文档。