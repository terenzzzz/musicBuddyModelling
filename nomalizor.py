import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# 下载必要的nltk数据
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# 定义函数读取自定义停用词列表
def load_custom_stopwords(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        stopwords = [line.strip() for line in file]
    return set(stopwords)

def preprocess_lyrics(lyrics):
    custom_stopwords = load_custom_stopwords('stopwords-en.txt')
    lemmatizer = WordNetLemmatizer()
    processed_lyrics = []

    for lyric in lyrics:
        if lyric.strip():  # If lyric is not empty
            words = word_tokenize(lyric)
            words = [lemmatizer.lemmatize(word.lower()) for word in words if word.isalpha() and word.lower() not in custom_stopwords]
            processed_lyrics.append(' '.join(words))
        else:
            processed_lyrics.append('')  # Keep a placeholder for empty lyrics

    print("Pre-Process Finished")
    return processed_lyrics









