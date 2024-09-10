## MusicBuddy Recommendation system
This project trained TF-IDF Word2Vec and LDA models for input lyrics dataset to give music recommendation service base on lyrics. 

## Database
This projec using mongodb as database to import lyrics in model training, set in each model files.

## Library
| 库名称        | 版本号   |
|---------------|----------|
| numpy         | 1.26.4   |
| scipy         | 1.10.1   |
| gensim        | 4.3.2    |
| python        | 3.11.4   |
| pymongo       | 4.8.0    |
| langdetect    | 1.0.9    |
| flask         | 3.0.3    |
| flask_cors    | 4.0.1    |
| nltk          | 3.8.1    |
| scikit-learn  | 1.5.0    |
```sh
pip install numpy==1.26.4 scipy==1.10.1 gensim==4.3.2 pymongo==4.8.0

python nltkDownloader.py
```

## Python Version
```sh
Python 3.11.4
```
## Train Model
```sh
python train_script.py
```
## Reset Model
```sh
python model_reset_script.py
```
## Start Service
```sh
python app.py
```
