from flask import Flask, request, jsonify
from flask_cors import CORS
from lda import LDAModelManager
from word2vec import Word2VecManager
from tfidf import TFIDFManager
from WeightedManager import WeightedManager
import os

app = Flask(__name__)
CORS(app)  # 这将为所有路由启用 CORS


# Load tfidf model
tfidf_manager = TFIDFManager()
tfidf_manager.load_from_file("tfidf")

# Load word2vec model
w2v_manager = Word2VecManager()
w2v_manager.load_from_file("word2vec")

# Load lda model
lda_manager = LDAModelManager()
lda_manager.load_from_file("lda")


# Load Weighted similarity
weighted_manager = WeightedManager()




@app.route('/recommend', methods=['GET'])
def recommend():
    return jsonify("Hello World")

@app.route('/getTrackTopic', methods=['GET'])
def getTrackTopic():
    song_id = request.args.get('track')
    if not song_id:
        return jsonify({"error": "Missing 'track' parameter"}), 400
    
    try:
        response = lda_manager.get_song_topics(song_id)
        if response is None:
            return jsonify({"error": "Song not found"}), 404
        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='localhost', port=5002)