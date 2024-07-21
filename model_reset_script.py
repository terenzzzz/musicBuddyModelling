import os
import shutil

def delete_file(file_path):
    try:
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"File '{file_path}' has been deleted successfully.")
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
            print(f"Directory '{file_path}' has been deleted successfully.")
        else:
            print(f"File or directory '{file_path}' does not exist.")
    except Exception as e:
        print(f"Error occurred while deleting file or directory '{file_path}': {e}")

if __name__ == "__main__":
    paths_to_delete = [
        "processed_lyrics.txt",
        "artists/",
        "lda/",
        "word2vec/",
        "tfidf/",
        "weighted_similarity.npz",
        "lda_similarity.npz",
        "w2v_similarity.npz",
        "tfidf_similarity.npz"
    ]
    
    for path in paths_to_delete:
        delete_file(path)