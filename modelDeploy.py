import streamlit as st
import pickle
import requests
import pandas as pd


# import base64
# def add_bg_from_local(image_file):
#     with open(image_file, "rb") as image_file:
#         encoded_string = base64.b64encode(image_file.read())
#     st.markdown(
#     f"""
#     <style>
#     .stApp {{
#         background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
#         background-size: cover
#     }}
#     </style>
#     """,
#     unsafe_allow_html=True
#     )
# add_bg_from_local('backg3.jpg')

st.title("Movie Review Analysis/Recommendation Model")


st.header('Movie Review System')
reviewInput = st.text_input('***Enter the Review***')
save_cv = pickle.load(open("count-vectorizer.pkl","rb"))
model = pickle.load(open("MovieReview.pkl","rb"))

if st.button('**Predict**'):
    sen=save_cv.transform([reviewInput]).toarray()
    res = model.predict(sen)[0]
    if res==1:
        st.write("##Positive")
    else:
        st.write('##Negative')


def load_poster(movieID):
    response = requests.get('https://api.themoviedb.org/3/movie/{}?api_key=d2a67a993186b66bfb2de3aa20f2d1cf&language=en-US'.format(movieID))
    data = response.json()
    return "https://image.tmdb.org/t/p/w500/"+data['poster_path']


def RecommendMovie(favMovieName):
    findIndex = list_of_movies[list_of_movies['title']==favMovieName].index[0]
    distance = simMat[findIndex]
    movieList = sorted(list(enumerate(distance)),reverse = True , key = lambda x:x[1])[1:6]
    RM = []
    poster = []
    for i in movieList:
        #Fetching poster from API using movieID
        movieID = list_of_movies.iloc[i[0]].movie_id

        RM.append(list_of_movies.iloc[i[0]].title)
        poster.append(load_poster(movieID))
    return RM,poster

simMat = pickle.load(open('simi.pkl','rb'))
st.header('Movie Recommendation System')
list_of_movies = pickle.load(open('movi.pkl','rb'))
movie_list = list_of_movies['title'].values
favMovieName = st.selectbox('Select a Movie:',movie_list)
if st.button('**Recommend**'):
    modelRecommend,poster = RecommendMovie(favMovieName)
    col1,col2,col3,col4,col5 = st.columns(5)
    with col1:
        st.write(modelRecommend[0])
        st.image(poster[0])
    with col2:
        st.write(modelRecommend[1])
        st.image(poster[1])
    with col3:
        st.write(modelRecommend[2])
        st.image(poster[2])
    with col4:
        st.write(modelRecommend[3])
        st.image(poster[3])
    with col5:
        st.write(modelRecommend[4])
        st.image(poster[4])