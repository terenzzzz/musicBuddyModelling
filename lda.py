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
import matplotlib.pyplot as plt

class LDAModelManager:
    def __init__(self, mongo_uri='mongodb://localhost:27017/', db_name='MusicBuddyVue', collection_name='tracks'):
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        self.dictionary = None
        self.corpus = None
        self.lda_model = None
        self.texts = None
        self.lyrics_documents = None

    def load_mongo_and_train(self, num_topics=10, output_dir='lda'):
        try:
            self.lyrics_documents = list(self.collection.find())
            if not self.lyrics_documents:
                print("No documents found in MongoDB collection.")
                return
            print(f"Found {len(self.lyrics_documents)} documents in MongoDB collection.")
        except Exception as e:
            print(f"Error fetching documents from MongoDB: {e}")
            return

        lyrics = [doc.get('lyric', '') for doc in self.lyrics_documents if isinstance(doc.get('lyric', None), str)]
        processed_lyrics = list(tqdm(nomalizor.preprocess_lyrics(lyrics), desc="Preprocessing Lyrics"))
        self.texts = [word_tokenize(lyric) for lyric in processed_lyrics]
        
        self.dictionary = corpora.Dictionary(self.texts)
        self.corpus = [self.dictionary.doc2bow(text) for text in self.texts]

        print(f"Original document count: {len(self.lyrics_documents)}")
        print(f"Documents after lyric extraction: {len(lyrics)}")
        print(f"Documents after preprocessing: {len(processed_lyrics)}")
        print(f"Documents after tokenization: {len(self.texts)}")
        print(f"Corpus size: {len(self.corpus)}")
        
        self.lda_model = gensim.models.ldamodel.LdaModel(
            corpus=self.corpus, 
            num_topics=num_topics, 
            id2word=self.dictionary, 
            alpha='auto', eta='auto', 
            passes=20
        )
        
        doc_topic_matrix = self.get_document_topic_matrix(num_topics)
        top_similarities_json = self.compute_top_similar_songs(doc_topic_matrix, top_n=20)
        
        os.makedirs(output_dir, exist_ok=True)
        
        with open(os.path.join(output_dir, 'top_similarities.json'), 'w') as f:
            json.dump(top_similarities_json, f, indent=2)
        
        self.dictionary.save(os.path.join(output_dir, 'dictionary.gensim'))
        corpora.MmCorpus.serialize(os.path.join(output_dir, 'corpus.mm'), self.corpus)
        self.lda_model.save(os.path.join(output_dir, 'lda_model.gensim'))
        
        with open(os.path.join(output_dir, 'texts.txt'), 'w', encoding='utf-8') as f:
            for text in self.texts:
                f.write(' '.join(text) + '\n')
        
        print(f"Model outputs saved to {output_dir}")

    def evaluate_model(self):
        topic_nums = list(range(10, 101, 5))
        coherence_scores = []
        perplexity_scores = []
        
        for num_topics in tqdm(topic_nums, desc="Evaluating LDA Models"):
            lda_model = gensim.models.ldamodel.LdaModel(corpus=self.corpus, id2word=self.dictionary, num_topics=num_topics, random_state=42)
            
            coherence_score = self.get_coherence(lda_model)
            coherence_scores.append(coherence_score)
            
            perplexity_score = self.get_perplexity(lda_model)
            perplexity_scores.append(perplexity_score)
        
        self.plot_evaluation_results(topic_nums, coherence_scores, perplexity_scores)
        
        for num_topics, coherence_score, perplexity_score in zip(topic_nums, coherence_scores, perplexity_scores):
            print(f"Num Topics: {num_topics}, Coherence Score: {coherence_score}, Perplexity: {perplexity_score}")

    def plot_evaluation_results(self, topic_nums, coherence_scores, perplexity_scores):
        fig, ax1 = plt.subplots()
        
        color = 'tab:red'
        ax1.set_xlabel('Number of Topics')
        ax1.set_ylabel('Coherence Score', color=color)
        ax1.plot(topic_nums, coherence_scores, color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Perplexity', color=color)
        ax2.plot(topic_nums, perplexity_scores, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        
        fig.tight_layout()
        plt.title("Coherence Score and Perplexity vs Number of Topics")
        plt.show()

    def load_files(self, input_dir='lda'):
        try:
            self.dictionary = corpora.Dictionary.load(os.path.join(input_dir, 'dictionary.gensim'))
            self.corpus = corpora.MmCorpus(os.path.join(input_dir, 'corpus.mm'))
            self.lda_model = gensim.models.ldamodel.LdaModel.load(os.path.join(input_dir, 'lda_model.gensim'))
            
            with open(os.path.join(input_dir, 'texts.txt'), 'r', encoding='utf-8') as f:
                self.texts = [line.strip().split() for line in f]
            
            print(f"All files loaded from {input_dir}")
            return True
        except Exception as e:
            print(f"Error loading files: {e}")
            return False

    def get_coherence(self, lda_model=None):
        model = lda_model or self.lda_model
        coherence_model = gensim.models.CoherenceModel(model=model, texts=self.texts, dictionary=self.dictionary, coherence='c_v')
        coherence_score = coherence_model.get_coherence()
        print(f"Coherence Score: {coherence_score}")
        return coherence_score

    def get_perplexity(self, lda_model=None):
        model = lda_model or self.lda_model
        perplexity = model.log_perplexity(self.corpus)
        print(f"Perplexity: {perplexity}")
        return perplexity

    def topic_prediction(self, new_lyric):
        new_lyric_processed = nomalizor.preprocess_lyrics([new_lyric])[0]
        new_lyric_bow = self.dictionary.doc2bow(word_tokenize(new_lyric_processed))
        print("New lyric:", new_lyric)
        print("Processed lyric:", new_lyric_processed)
        print("Topic distribution:")
        pprint(sorted(self.lda_model[new_lyric_bow], key=lambda x: x[1], reverse=True))

    def get_most_representative_doc(self):
        topic_probs = [max(prob, key=lambda x: x[1])[1] for prob in self.lda_model[self.corpus]]
        most_rep_index = topic_probs.index(max(topic_probs))
        print(f"Most representative document index: {most_rep_index}")
        print(f"Probability of its main topic: {max(topic_probs):.4f}")
        print("Document content:")
        print(" ".join(self.texts[most_rep_index]))
        return most_rep_index, max(topic_probs)

    def get_document_topic_matrix(self, num_topics):
        doc_topic_matrix = np.zeros((len(self.corpus), num_topics))
        for i, doc_bow in enumerate(self.corpus):
            doc_topics = self.lda_model.get_document_topics(doc_bow, minimum_probability=0)
            for topic, prob in doc_topics:
                doc_topic_matrix[i, topic] = prob
        print(f"Matrix shape: {doc_topic_matrix.shape}")
        print("Sample of first 5 rows and 5 columns:")
        print(doc_topic_matrix[:5, :5])
        return doc_topic_matrix

    def compute_top_similar_songs(self, song_topic_matrix, top_n=20):
        num_songs = song_topic_matrix.shape[0]
        similarities = cosine_similarity(song_topic_matrix)
        top_similarities_json = []

        for i in tqdm(range(num_songs), desc="Computing similar songs"):
            song_similarities = similarities[i]
            song_similarities[i] = -1
            top_similar_indices = np.argsort(song_similarities)[::-1][:top_n]
            top_similar_docs = [
                {
                    "track": {"$oid": str(self.lyrics_documents[idx]['_id'])},
                    "value": float(song_similarities[idx])
                }
                for idx in top_similar_indices
            ]
            top_similarities_json.append({
                "track": {"$oid": str(self.lyrics_documents[i]['_id'])},
                "topsimilar": top_similar_docs
            })

        return top_similarities_json

if __name__ == "__main__":
    lda_manager = LDAModelManager()
    num_topics = 10
    input_dir = 'lda'

    # 检查是否存在已保存的模型文件
    if all(os.path.exists(os.path.join(input_dir, f)) for f in ['dictionary.gensim', 'corpus.mm', 'lda_model.gensim', 'texts.txt']):
        print("Loading LDA results from files...")
        if lda_manager.load_files(input_dir):
            print("LDA model loaded successfully.")
        else:
            print("Failed to load LDA model. Training a new one...")
            lda_manager.load_mongo_and_train(num_topics=num_topics)
    else:
        print("Loading data from MongoDB and training LDA model...")
        lda_manager.load_mongo_and_train(num_topics=num_topics)

    if lda_manager.lda_model is not None:
        print("\n1. Top 5 words for each topic:")
        pprint(lda_manager.lda_model.print_topics(num_topics=num_topics, num_words=5))

        print("\n2. Calculate topic coherence score:")
        lda_manager.get_coherence()
        
        print("\n3. Calculate topic perplexity score:")
        lda_manager.get_perplexity()
        
        print("\n4. Predict topics for a new lyric:")
        lda_manager.topic_prediction("Accepting Your grace with Love")

        print("\n5. Show the most representative document:")
        lda_manager.get_most_representative_doc()
        
        print("\n6. Generate document-topic matrix:")
        lda_manager.get_document_topic_matrix(num_topics)

    # 如果您想评估不同主题数量的模型性能,可以取消注释下面的行
    # lda_manager.evaluate_model()