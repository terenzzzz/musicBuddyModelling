from flask import Flask, request, jsonify
from flask_cors import CORS
from lda import LDAModelManager
import os

app = Flask(__name__)
CORS(app)  # 这将为所有路由启用 CORS

lda_manager = LDAModelManager()
lda_manager.load_files("lda")
lda_manager.get_song_topics('65ffc183c1ab936c978f29a8')

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