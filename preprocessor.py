import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from functools import lru_cache
from tqdm import tqdm
from langdetect import detect_langs
import re
from pymongo import MongoClient
import json




class Preprocessor:
    def __init__(self, stopwords_file='stopwords-en.txt'):
        self.custom_stopwords = self.load_custom_stopwords(stopwords_file)
        self.lemmatizer = WordNetLemmatizer()
        self.alpha_pattern = re.compile("^[a-zA-Z]+$")
        self.english_words = set(nltk.corpus.words.words())
        self.tracks_documents = []

    def load_custom_stopwords(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            stopwords = [line.strip() for line in file]
        return set(stopwords)
    
    @lru_cache(maxsize=1000000)
    def is_english_word(self, word):
        if not self.alpha_pattern.match(word):
            return False
        
        if word.lower() in self.english_words:
            return True
        
        # 使用 langdetect 作为后备检查
        try:
            detected_languages = detect_langs(word)
            return any(lang.lang == 'en' for lang in detected_languages)
        except:
            return False
        
    def get_wordnet_pos(self, treebank_tag):
        if treebank_tag.startswith('J'):
            return 'a'  # 形容词
        elif treebank_tag.startswith('V'):
            return 'v'  # 动词
        elif treebank_tag.startswith('N'):
            return 'n'  # 名词
        elif treebank_tag.startswith('R'):
            return 'r'  # 副词
        else:
            return 'n'  # 默认使用名词


    def preprocess_lyrics(self, lyrics):
        processed_lyrics = []
        input_count = len(lyrics)
        empty_line_count = 0
        non_empty_line_count = 0
    
        for index, lyric in enumerate(tqdm(lyrics, desc="Processing lyrics")):
            stripped_lyric = lyric.strip()
            if stripped_lyric:  # 如果歌词行不为空
                words = word_tokenize(stripped_lyric)
                tagged_words = nltk.pos_tag(words)
                lemmatized_words = [
                    self.lemmatizer.lemmatize(word.lower(), self.get_wordnet_pos(tag))
                    for word, tag in tagged_words
                    if word.isalpha() and word.lower() not in self.custom_stopwords and self.is_english_word(word)
                ]
                processed_line = ' '.join(lemmatized_words)
                processed_lyrics.append(processed_line)
                non_empty_line_count += 1
            else:
                processed_lyrics.append('')  # 保留占位符用于空歌词行
                empty_line_count += 1
                print(f"Empty line at index {index}")  # 打印空行的索引
    
        output_count = len(processed_lyrics)
        print(f"Input lyrics count: {input_count}")
        print(f"Output lyrics count: {output_count}")
        print(f"Non-empty lines processed: {non_empty_line_count}")
        print(f"Empty lines encountered: {empty_line_count}")
    
        if input_count != output_count:
            print("Warning: Input and output counts do not match!")
            # 可以在这里添加更多的诊断信息
    
        return processed_lyrics
    
    
    def load_data(self, mongo_uri='mongodb://localhost:27017/', 
                  db_name='MusicBuddyVue', 
                  collection_name='tracks'):
        client = MongoClient(mongo_uri)
        db = client[db_name]
        collection = db[collection_name]
        
        try:
            tracks_documents = list(collection.find())
            if not tracks_documents:
                print("No documents found in MongoDB collection.")
            print(f"Found {len(tracks_documents)} documents in MongoDB collection.")
            self.tracks_documents = tracks_documents

        except Exception as e:
            print(f"Error fetching documents from MongoDB: {e}")
            

if __name__ == "__main__":
    preprocessor = Preprocessor()
    
    lyric1 = "What people do for money Anything for that green What people do for money It's a scam"
    lyric2 = "Her smile was filled with a feeling of warmth and happiness. She feels happy"
    lyric3 = "今天是个好日子, people is a good money green scam birds cats hands cups cup"
    lyric4 = "De festival en festival nos vamos cruzando ¿qué tal si cambiamos de escenario? fucking bitch"
    lyric5 = "Je guette tes pas Je suis amoureux Ou fou de toi Les deux si tu veux"
    
    processed_lyrics = preprocessor.preprocess_lyrics([lyric1, lyric2,lyric3,lyric4,lyric5])
    print(processed_lyrics)
    
    
    
    preprocessor.load_data('mongodb://localhost:27017/', 'MusicBuddyVue', 'tracks')
    lyrics = [doc.get('lyric', '') for doc in preprocessor.tracks_documents if isinstance(doc.get('lyric', None), str)]
    processed_lyrics = preprocessor.preprocess_lyrics(lyrics)
    with open("processed_lyrics.txt", 'w', encoding='utf-8') as f:
        for lyrics in processed_lyrics:
            f.write(lyrics + '\n')
    print(f"Saved {len(processed_lyrics)} documents to processed_lyrics.txt")

