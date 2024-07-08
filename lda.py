from pymongo import MongoClient
import nomalizor
from tqdm import tqdm
import gensim
from gensim import corpora
from nltk.tokenize import word_tokenize
from pprint import pprint
import os
import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json



logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def load_mongo_and_train(num_topics=50, output_dir='lda'):
    # Connect to MongoDB
    client = MongoClient('mongodb://localhost:27017/')
    db = client['MusicBuddyVue']
    collection = db['tracks']
    
    try:
        # Fetch lyric documents from MongoDB
        lyrics_documents = list(collection.find())
        if not lyrics_documents:
            print("No documents found in MongoDB collection.")
            return None, None, None, None
        else:
            print(f"Found {len(lyrics_documents)} documents in MongoDB collection.")
    except Exception as e:
        print(f"Error fetching documents from MongoDB: {e}")
        return None, None, None, None

    
    # Extract lyrics and preprocess them
    lyrics = [doc.get('lyric', '') for doc in lyrics_documents if isinstance(doc.get('lyric', None), str)]
    processed_lyrics = list(tqdm(nomalizor.preprocess_lyrics(lyrics), desc="Preprocessing Lyrics"))
    
    # 将预处理后的歌词转换为词列表
    # texts = [['hello', 'world'], ['machine', 'learning', 'is', 'fun']]
    texts = [word_tokenize(lyric) for lyric in processed_lyrics if lyric]
    
    # 创建词典
    # {'hello': 0, 'world': 1, 'machine': 2, 'learning': 3, 'is': 4, 'fun': 5}
    dictionary = corpora.Dictionary(texts)
    
    # 创建词袋模型
    # [[(0, 1), (1, 1)], [(2, 1), (3, 1), (4, 1), (5, 1)]]
    corpus = [dictionary.doc2bow(text) for text in texts]
    

    # 训练LDA模型
    # [(0, '0.200*"hello" + 0.200*"world" + 0.200*"fun"'),(1, '0.200*"machine" + 0.200*"learning" + 0.200*"is"')]
    lda_model = gensim.models.ldamodel.LdaModel(
                                    corpus, 
                                    num_topics=num_topics, 
                                    id2word=dictionary, 
                                    alpha='auto', eta='auto', 
                                    passes=15
                                )
    
    # 文档-主题 矩阵
    doc_topic_matrix = get_document_topic_matrix(lda_model, corpus, num_topics)

    
    # 计算每首歌的前20个最相似的歌，并格式化为指定的JSON结构
    top_similarities_json = compute_top_similar_songs(doc_topic_matrix, lyrics_documents, top_n=20)
    
    
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存结果到文件
    with open(os.path.join(output_dir, 'top_similarities.json'), 'w') as f:
        json.dump(top_similarities_json, f, indent=2)
    
    # Save dictionary
    dictionary.save(os.path.join(output_dir, 'dictionary.gensim'))
    
    # Save corpus
    corpora.MmCorpus.serialize(os.path.join(output_dir, 'corpus.mm'), corpus)
    
    # Save LDA model
    lda_model.save(os.path.join(output_dir, 'lda_model.gensim'))
    
    # Save texts (optional, as it might be large)
    with open(os.path.join(output_dir, 'texts.txt'), 'w', encoding='utf-8') as f:
        for text in texts:
            f.write(' '.join(text) + '\n')
    
    print(f"Model outputs saved to {output_dir}")
    return dictionary, corpus, lda_model, texts

def load_files(input_dir='lda'):
    try:
        # Load dictionary
        dictionary_path = os.path.join(input_dir, 'dictionary.gensim')
        dictionary = corpora.Dictionary.load(dictionary_path)
        print(f"Dictionary loaded from {dictionary_path}")

        # Load corpus
        corpus_path = os.path.join(input_dir, 'corpus.mm')
        corpus = corpora.MmCorpus(corpus_path)
        print(f"Corpus loaded from {corpus_path}")

        # Load LDA model
        lda_model_path = os.path.join(input_dir, 'lda_model.gensim')
        lda_model = gensim.models.ldamodel.LdaModel.load(lda_model_path)
        print(f"LDA model loaded from {lda_model_path}")

        # Load texts
        texts_path = os.path.join(input_dir, 'texts.txt')
        with open(texts_path, 'r', encoding='utf-8') as f:
            texts = [line.strip().split() for line in f]
        print(f"Texts loaded from {texts_path}")

        return dictionary, corpus, lda_model, texts

    except Exception as e:
        print(f"Error loading files: {e}")
        return None, None, None, None
    
def get_coherence(lda_model,texts,dictionary):
    coherence_model_lda = gensim.models.CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print(f"主题连贯性分数: {coherence_lda}")
    return coherence_lda
    
def topic_prediction(new_lyric, lda_model):
    
    new_lyric_processed = nomalizor.preprocess_lyrics([new_lyric])[0]
    new_lyric_bow = dictionary.doc2bow(word_tokenize(new_lyric_processed))
    print("新歌词:", new_lyric)
    print("处理后的歌词:", new_lyric_processed)
    print("主题分布:")
    pprint(sorted(lda_model[new_lyric_bow], key=lambda x: x[1], reverse=True))
    

def get_most_representative_doc(lda_model, corpus):
    topic_probs = [max(prob, key=lambda x: x[1])[1] for prob in lda_model[corpus]]
    most_rep_index = topic_probs.index(max(topic_probs))
    print(f"最具代表性的文档索引: {most_rep_index}")
    print(f"该文档属于其最主要主题的概率: {max(topic_probs):.4f}")
    print("文档内容:")
    print(" ".join(texts[most_rep_index]))
    return most_rep_index, max(topic_probs)

def get_document_topic_matrix(lda_model, corpus, num_topics):
    # 创建一个空矩阵,行数等于文档数,列数等于主题数
    doc_topic_matrix = np.zeros((len(corpus), num_topics))
    
    # 遍历每个文档
    for i, doc_bow in enumerate(corpus):
        # 获取文档的主题分布
        doc_topics = lda_model.get_document_topics(doc_bow, minimum_probability=0)
        # 填充矩阵
        for topic, prob in doc_topics:
            doc_topic_matrix[i, topic] = prob
            
    print(f"矩阵形状: {doc_topic_matrix.shape}")
    print("前5行和5列的样本:")
    print(doc_topic_matrix[:5, :5])
    
    return doc_topic_matrix

def compute_top_similar_songs(song_topic_matrix, lyrics_documents, top_n=20):
    """
    计算每首歌的前N个最相似的歌曲，并以指定的JSON结构格式化结果
    
    :param song_topic_matrix: 所有歌曲的主题分布矩阵
    :param lyrics_documents: 包含歌曲信息的原始文档列表
    :param top_n: 为每首歌保留的最相似歌曲数量
    :return: 格式化的JSON结构
    """
    num_songs = song_topic_matrix.shape[0]
    
    # 计算所有歌曲之间的余弦相似度
    similarities = cosine_similarity(song_topic_matrix)

    # 准备JSON结构
    top_similarities_json = []

    # 对每首歌找出最相似的歌曲
    for i in tqdm(range(num_songs), desc="Computing similar songs"):
        # 获取相似度，排除自身
        song_similarities = similarities[i]
        song_similarities[i] = -1  # 将自身的相似度设为-1，确保不会被选中

        # 找出前N个最相似的歌曲
        top_similar_indices = np.argsort(song_similarities)[::-1][:top_n]
        
        # 准备当前歌曲的相似歌曲列表
        top_similar_docs = [
            {
                "track": {"$oid": str(lyrics_documents[idx]['_id'])},
                "value": float(song_similarities[idx])
            }
            for idx in top_similar_indices
        ]

        # 添加到最终的JSON结构
        top_similarities_json.append({
            "track": {"$oid": str(lyrics_documents[i]['_id'])},
            "topsimilar": top_similar_docs
        })

    return top_similarities_json



if __name__ == "__main__":
    num_topics = 50
    input_dir = 'lda'  # 或者你保存文件的其他目录
    if (os.path.exists(os.path.join(input_dir, 'dictionary.gensim')) and
        os.path.exists(os.path.join(input_dir, 'corpus.mm')) and
        os.path.exists(os.path.join(input_dir, 'lda_model.gensim')) and
        os.path.exists(os.path.join(input_dir, 'texts.txt'))):
        
        print("Loading LDA results from files...")
        dictionary, corpus, lda_model, texts = load_files(input_dir)
    else:
        print("Loading data from MongoDB and training LDA model...")
        dictionary, corpus, lda_model, texts = load_mongo_and_train(num_topics=num_topics)
    
    if lda_model is None:
        print("模型训练失败，请检查数据和参数。")
    else:
        print("\n1. 每个主题的前10个词语：")
        pprint(lda_model.print_topics(num_topics=num_topics, num_words=10))

        print("\n2. 计算主题连贯性分数：")
        get_coherence(lda_model,texts,dictionary)
        
        
        print("\n3. 对一个新的歌词进行主题推断：")
        topic_prediction("Accepting Your grace with Love", lda_model)


        print("\n4. 显示最具代表性的文档：")
        most_rep_index, prob = get_most_representative_doc(lda_model, corpus)
        
        print("\n5. 生成文档-主题矩阵:")
        doc_topic_matrix = get_document_topic_matrix(lda_model, corpus, num_topics)
        

        

    
    
        

