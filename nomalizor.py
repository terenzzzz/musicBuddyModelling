import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# 下载必要的nltk数据
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')


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





if __name__ == "__main__":
    
    lyric1 = "What people do for money Anything for that green What people do for money It's a scam, it's a scheme What people do for money Anything for that green"
    processed_lyrics = preprocess_lyrics([lyric1,lyric1,lyric1,lyric1])
    print(processed_lyrics)



# 加载停用词列表：从文件中读取自定义停用词，并将其存储在一个集合中。
# 分词：将歌词文本分解为单词。
# 词形还原：将单词还原为其基本形式，以减少词汇量的多样性。
# 去停用词和去非字母词：去除不重要的停用词和非字母字符，以减少噪音。
# 处理空歌词：保持空歌词的占位符，以确保歌词列表的长度不变。