# movie_recommendation_using_w2v

So, this is an updated version of my previous project ([older version](https://github.com/himalaya-singh-sheoran/movie_recommendation_using_w2v/tree/main)).
I've made this using app using [streamlit](https://streamlit.io/). In short it is a great framework for those who don't know langauges for frontend development. You can code everything in python.
I have used the average word2vec embeddings of the words in movie content as the vector for the movie. Learn more about word2vec embeddings [here](https://www.youtube.com/watch?v=LSS_bos_TPI).
Then I trained a nearest neighbours (metric as cosine) model to get movie recommendations.Learn more about knn [here](https://www.youtube.com/watch?v=HVXime0nQeI).
Learn more about cosine similarity [here](https://www.youtube.com/watch?v=ieMjGVYw9ag).
Working application is available [here](https://share.streamlit.io/himalaya-singh-sheoran/movie_recommendation_using_w2v/new_1/recomm.py) (create a new user or use username = qwe,password=asd).

NOTE : To store the user's details host a database online and connect that to streamlit learn more [here](https://docs.streamlit.io/en/latest/tutorial/databases.html)

To use the app in local runtime. Donwload all the files.

```python
pip install -U -r requirements.txt
```

Then,
```python
streamlit run recomm.py
```
About the contents of this repo:
|File name | Description |
|:--------:|:-----------:|
|KNNC.pkl| model|
|embedsData.txt|File of movie embeddings|
|id2title.pkl|Dictonary of movie id and title|
|id2val.pkl|Dictonary of movie id and index|
|val2id.pkl|opposite of id2val|
|title2id.pkl|opposite of id2title|
|new_final.csv|movies data|
|users.db|users database sqlite3|


