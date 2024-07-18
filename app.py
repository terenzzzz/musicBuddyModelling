from flask import Flask, request, jsonify
from flask_cors import CORS
# from lda import LDAModelManager
# from word2vec import Word2VecManager
# from tfidf import TFIDFManager
# from WeightedManager import WeightedManager
from pymongo import MongoClient
import json
from weightedManager import weightedManager

app = Flask(__name__)
CORS(app)  # 这将为所有路由启用 CORS


client = MongoClient('mongodb://localhost:27017/')
db = client['MusicBuddyVue']
tracks_collection = db['tracks']
        


# # Load tfidf model
# tfidf_manager = TFIDFManager()
# tfidf_manager.load_from_file("tfidf")

# # # Load word2vec model
# w2v_manager = Word2VecManager()
# w2v_manager.load_from_file("word2vec")

# # Load lda model
# lda_manager = LDAModelManager()
# lda_manager.load_from_file("lda")


# # Load Weighted similarity
default_tfidf_weight = 0.33
default_w2v_weight = 0.33
default_lda_weight = 0.34
weighted_manager = weightedManager(default_tfidf_weight, default_w2v_weight, 
                                      default_lda_weight,"tfidf/doc_id_to_index_map.json")


@app.route('/getLyricTopWordsByLyric', methods=['POST'])
def getLyricTopWordsByLyric():
    data = request.get_json()
    lyric = data['lyric']
    if not lyric:
        return jsonify({"error": "Missing 'lyric' parameter"}), 400
    
    try:
        response = weighted_manager.tfidf_manager.get_top_words_by_lyric(lyric)
        if response is None:
            return jsonify({"error": "Song not found"}), 404
        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/getTrackTopic', methods=['GET'])
def getTrackTopic():
    song_id = request.args.get('track')
    if not song_id:
        return jsonify({"error": "Missing 'track' parameter"}), 400
    
    try:
        response = weighted_manager.lda_manager.get_song_topics(song_id)
        if response is None:
            return jsonify({"error": "Song not found"}), 404
        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/getTrackTopicByLyric', methods=['POST'])
def getTrackTopicByLyric():
    data = request.get_json()
    lyric = data['lyric']
    if not lyric:
        return jsonify({"error": "Missing 'lyric' parameter"}), 400
    
    try:
        response = weighted_manager.lda_manager.get_song_topics_by_lyric(lyric)
        if response is None:
            return jsonify({"error": "Song not found"}), 404
        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@app.route('/getTfidfRecommendByLyrics', methods=['POST'])
def getTfidfRecommendByLyrics():
    try:
        # 从请求体中获取数组
        data = request.get_json()
        lyric = data['lyric']
        if not lyric:
            return jsonify({"error": "Missing 'lyric' parameter"}), 400
        
        try:   
            response = weighted_manager.tfidf_manager.get_similar_documents_for_lyrics(lyric)
            return jsonify(response), 200
        except json.JSONDecodeError:
            return jsonify({"error": "Invalid 'lyric' format. Expected a JSON array."}), 400
    except Exception as e:
        print(f"Error: {str(e)}")  # Print detailed error information
        return jsonify({"error": str(e)}), 500
    
@app.route('/getW2VRecommendByLyrics', methods=['POST'])
def getW2VRecommendByLyrics():
    try:
        # 从请求体中获取数组
        data = request.get_json()
        lyric = data['lyric']
        if not lyric:
            return jsonify({"error": "Missing 'lyric' parameter"}), 400
        
        try:   
            response = weighted_manager.w2v_manager.get_similar_documents_for_lyrics(lyric)
            return jsonify(response), 200
        except json.JSONDecodeError:
            return jsonify({"error": "Invalid 'lyric' format. Expected a JSON array."}), 400
    except Exception as e:
        print(f"Error: {str(e)}")  # Print detailed error information
        return jsonify({"error": str(e)}), 500
    
@app.route('/getLDARecommendByLyrics', methods=['POST'])
def getLDARecommendByLyrics():
    try:
        # 从请求体中获取数组
        data = request.get_json()
        lyric = data['lyric']
        if not lyric:
            return jsonify({"error": "Missing 'lyric' parameter"}), 400
        
        try:   
            response = weighted_manager.lda_manager.get_similar_documents_for_lyrics(lyric)
            return jsonify(response), 200
        except json.JSONDecodeError:
            return jsonify({"error": "Invalid 'lyric' format. Expected a JSON array."}), 400
    except Exception as e:
        print(f"Error: {str(e)}")  # Print detailed error information
        return jsonify({"error": str(e)}), 500
    
    
@app.route('/getWeightedRecommendByLyrics', methods=['POST'])
def getWeightedRecommendByLyrics():
    try:
        # 从请求体中获取数组
        data = request.get_json()
        lyric = data['lyric']
        tfidf_weight = data['tfidf_weight']
        w2v_weight = data['w2v_weight']
        lda_weight = data['lda_weight']
        if not lyric:
            return jsonify({"error": "Missing 'lyric' parameter"}), 400
        
        try:   
            response = weighted_manager.get_similar_documents_for_lyrics(lyric, 
                                                                         tfidf_weight, 
                                                                         w2v_weight, 
                                                                         lda_weight)
            return jsonify(response), 200
        except json.JSONDecodeError:
            return jsonify({"error": "Invalid 'lyric' format. Expected a JSON array."}), 400
    except Exception as e:
        print(f"Error: {str(e)}")  # Print detailed error information
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='localhost', port=5002)