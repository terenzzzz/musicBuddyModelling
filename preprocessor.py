import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

class Preprocessor:
    def __init__(self, stopwords_file='stopwords-en.txt'):
        # 下载必要的nltk数据
        # nltk.download('punkt')
        # nltk.download('stopwords')
        # nltk.download('wordnet')
        # nltk.download('averaged_perceptron_tagger')
        
        self.custom_stopwords = self.load_custom_stopwords(stopwords_file)
        self.lemmatizer = WordNetLemmatizer()

    def load_custom_stopwords(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            stopwords = [line.strip() for line in file]
        return set(stopwords)

    def get_wordnet_pos(self, treebank_tag):
        if treebank_tag.startswith('J'):
            return nltk.corpus.wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return nltk.corpus.wordnet.VERB
        elif treebank_tag.startswith('N'):
            return nltk.corpus.wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return nltk.corpus.wordnet.ADV
        else:
            return nltk.corpus.wordnet.NOUN

    def preprocess_lyrics(self, lyrics):
        processed_lyrics = []
        for lyric in lyrics:
            if lyric.strip():  # If lyric is not empty
                words = word_tokenize(lyric)
                tagged_words = pos_tag(words)
                words = [self.lemmatizer.lemmatize(word.lower(), self.get_wordnet_pos(tag)) 
                         for word, tag in tagged_words 
                         if word.isalpha() and word.lower() not in self.custom_stopwords]
                processed_lyrics.append(' '.join(words))
            else:
                processed_lyrics.append('')  # Keep a placeholder for empty lyrics
        print("Pre-Process Finished")
        return processed_lyrics

if __name__ == "__main__":
    preprocessor = Preprocessor()
    
    lyric1 = "What people do for money Anything for that green What people do for money It's a scam"
    lyric2 = "Her smile was filled with a feeling of warmth and happiness. She feels happy"
    processed_lyrics = preprocessor.preprocess_lyrics([lyric1, lyric2])
    print(processed_lyrics)