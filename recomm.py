# importing libraries
import streamlit as st
from streamlit.hashing import _CodeHasher
from streamlit.report_thread import get_report_ctx
from streamlit.server.server import Server

from ast import literal_eval
import pandas as pd 
from PIL import Image
import requests

from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser
from gensim.models import KeyedVectors
from gensim.models.callbacks import CallbackAny2Vec



# Will load css style.css

st.markdown(""" <style>
body {
color: #F5F5F5;
background-color:#00BFFF
;
}
.css-145kmo2 {
    font-size: 0.8rem;
    color: rgb(38, 39, 48);
    margin-bottom: 0.01rem;
}
.st-cc {
    color : black;
    background-color: #FFD700;
}
.css-1syfshr {
    display: flex;
    -webkit-box-align: center;
    align-items: center;
    padding: 1rem;
    background-color: #FFD700;
    border-radius: 0.25rem;
    color: rgb(38, 39, 48);
}
.css-3ftxgc {
    display: inline-flex;
    -webkit-box-align: center;
    align-items: center;
    -webkit-box-pack: center;
    justify-content: center;
    font-weight: 400;
    padding: 0.25rem 0.5rem;
    border-radius: 0.25rem;
    margin: 0px;
    line-height: 1.6;
    color: inherit;
    width: 100%;
    font-size: 0.8rem;
    background-color: #40E0D0;
    border: 1px solid rgb(230, 234, 241);
}
.css-x0vgxp {
    position: fixed;
    inset: 0px;
    background: #00BFFF;
    z-index: 1000050;
    padding: 3.5rem 0.75rem 0.75rem;
    overflow: auto;
    display: flex;
}
.css-2trqyj {
    display: inline-flex;
    -webkit-box-align: center;
    align-items: center;
    -webkit-box-pack: center;
    justify-content: center;
    font-weight: 400;
    padding: 0.25rem 0.75rem;
    border-radius: 0.25rem;
    margin: 0px;
    line-height: 1.6;
    color: white;
    width: auto;
    background-color: #40E0D0;
    border: #7CFC00;
}
.css-3xqji8 {
    display: inline-flex;
    -webkit-box-align: center;
    align-items: center;
    -webkit-box-pack: center;
    justify-content: center;
    font-weight: 400;
    border-radius: 0.25rem;
    margin: 0px;
    line-height: 1.6;
    color: white;
    width: auto;
    background-color: #FFD700;
    border: 1px solid transparent;
    padding: 0.75rem;
}
.reportview-container 
[class^="st-b"]  {
    color: black;
    font-family: monospace;
}
.st-bb {
    background-color: #FFD700;
    font-size:18px;
}

footer {
    color : red;
    font-family: monospace;
    background-color: #FFD700;
}
</style>""", unsafe_allow_html = True)

try:
    import streamlit.ReportThread as ReportThread
except Exception:
    # Streamlit >= 0.65.0
    import streamlit.report_thread as ReportThread


class _SessionState:

    def __init__(self, session, hash_funcs):
        """Initialize SessionState instance."""
        self.__dict__["_state"] = {
            "data": {},
            "embeddings" : [],
            "new_" : False,
            "model" : None,
            "title" : None,
            "ls" : [],
            "id_ls" : [],
            "new_title" : None,
            'id' : None,
            "hash": None,
            "hasher": _CodeHasher(hash_funcs),
            "is_rerun": False,
            "session": session,
        }

    def __call__(self, **kwargs):
        """Initialize state data once."""
        for item, value in kwargs.items():
            if item not in self._state["data"]:
                self._state["data"][item] = value

    def __getitem__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)
        
    def __getattr__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)

    def __setitem__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value

    def __setattr__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value
    
    def clear(self):
        """Clear session state and request a rerun."""
        self._state["data"].clear()
        self._state["session"].request_rerun()
    
    def sync(self):
        """Rerun the app with all state values up to date from the beginning to fix rollbacks."""

        # Ensure to rerun only once to avoid infinite loops
        # caused by a constantly changing state value at each run.
        #
        # Example: state.value += 1
        if self._state["is_rerun"]:
            self._state["is_rerun"] = False
        
        elif self._state["hash"] is not None:
            if self._state["hash"] != self._state["hasher"].to_bytes(self._state["data"], None):
                self._state["is_rerun"] = True
                self._state["session"].request_rerun()

        self._state["hash"] = self._state["hasher"].to_bytes(self._state["data"], None)


def _get_session():
    session_id = get_report_ctx().session_id
    session_info = Server.get_current()._get_session_info(session_id)

    if session_info is None:
        raise RuntimeError("Couldn't get your Streamlit Session object.")
    
    return session_info.session


def _get_state(hash_funcs=None):
    session = _get_session()

    if not hasattr(session, "_custom_session_state"):
        session._custom_session_state = _SessionState(session, hash_funcs)

    return session._custom_session_state


def _get_session():
    session_id = get_report_ctx().session_id
    session_info = Server.get_current()._get_session_info(session_id)

    if session_info is None:
        raise RuntimeError("Couldn't get your Streamlit Session object.")

    return session_info.session


def get_state(hash_funcs=None):
    session = _get_session()

    if not hasattr(session, "_custom_session_state"):
        session._custom_session_state = _SessionState(session, hash_funcs)

    return session._custom_session_state



@st.cache
def load_data():
    df = pd.read_csv('new_final.csv')
    titles = ['']
    titles.extend((df['title']))
    return df,titles
@st.cache
def load_model():
    model = Word2Vec.load('model_w2v.model')
    return model

from tmdbv3api import TMDb
import json
import requests
tmdb = TMDb()
tmdb.api_key = '3500eeb1590388069824646275773f36'
from tmdbv3api import Movie
tmdb_movie = Movie()

def return_data(x):
    try:
        movie_id = x
        response = requests.get('https://api.themoviedb.org/3/movie/{}?api_key={}'.format(movie_id,tmdb.api_key))
        data_json = response.json()
        return data_json
    except:
        return -1
    

def display_data(_id):
    data = return_data(_id)
    if data == -1:
        st.markdown("### !Data not found we're sorry")
    else :
        try:
            poster_path = data['poster_path']
            path = "https://image.tmdb.org/t/p/original/{}".format(poster_path)
            im = Image.open(requests.get(path, stream=True).raw)
            im = im.resize((400, 600))
            st.image(im, width=None)
        except :
            st.markdown("## POSTER NOT FOUND")
        slice_ = pd.DataFrame(state.data[state.data['id'] == _id])
        st.markdown("####  Genres : ")
        ls = list(slice_['genres'])
        ls = literal_eval(ls[0])
        _out =  ""
        for i in ls:
            _out += i + ", "
        _out = _out[:-2]
        st.write(_out)
        st.markdown("####  CAST : ")
        ls = list(slice_['cast'])
        ls = literal_eval(ls[0])
        _out =  ""
        for i in ls:
            _out += i + ", "
        _out = _out[:-2]
        st.write(_out)
        st.markdown("####  Director : ")
        st.write(slice_.iloc[0]['director'])
        st.markdown("####  Overview : ")
        st.write(slice_.iloc[0]['overview'])
        st.markdown("####  Rating : ")
        st.write(data['vote_average'])
        st.markdown("####  Spoken Language(s) : ")
        _out =  ""
        for i in literal_eval(slice_.iloc[0]['spoken_languages']):
            _out += i + ", "
        _out = _out[:-2]
        st.write(_out)
        st.markdown("####  Runtime : ")
        st.write(data['runtime'],"minutes")

        
global embeddings

@st.cache
def get_vectors():
    embeddings = []
    model =  Word2Vec.load('model_w2v.model')
    for line in state.data['content']:
        w2v = count = 0
        for word in line.split():
            if word in model.wv.vocab:
                count +=1
                if w2v is None:
                    w2v = model[word]
                else :
                    w2v +=  model[word]
        if w2v is not None:
            w2v = w2v/count
            embeddings.append(w2v)
    return embeddings
    
global state
state = _get_state()


@st.cache
def get_recomm(title):
    cosine_similarities = cosine_similarity(state.embeddings,state.embeddings)
    titles_ = state.data[['title']]
    indices = pd.Series(state.data.index,index = state.data['title']).drop_duplicates()
    ix = indices[title]
    cosine_sim = list(enumerate(cosine_similarities[ix]))
    cosine_sim = sorted(cosine_sim,key = lambda x: x[1],reverse = True)
    cosine_sim = cosine_sim[1:6]
    indx_ = [i[0] for i in cosine_sim]
    watch_next = titles_.iloc[indx_]
    ls = []
    id_ls = []
    for index,row in watch_next.iterrows():
      ls.append((row['title']))
      id_ls.append(index)
    for i,x in enumerate(id_ls):
        id_ls[i] = int(state.data.iloc[x]['id'])
    return ls,id_ls,True

def get_small_image(_id):
    data_ = return_data(_id)
    if data_ == -1:
        return -1
    else :
        poster_path = data_['poster_path']
        path = "https://image.tmdb.org/t/p/original/{}".format(poster_path)
        im = Image.open(requests.get(path, stream=True).raw)
        im = im.resize((100, 150))
        return im

def display_recomm(ls,id_ls):
    if st.button(ls[0],key = 'c1'):
            state.new_ = True
            state.new_title = ls[0]
            state.id = int(state.data[state.data['title'] == ls[0]]['id'])
            rerun()
    im1 = get_small_image(id_ls[0])
    
    if im1 != -1:
        try :
            st.image(im1, width=None)
        except:
            pass
        
    if st.button(ls[1],key = 'c2'):
        state.new_ = True
        state.new_title = ls[1]
        state.id = int(state.data[state.data['title'] == ls[1]]['id'])
        rerun()
    im2 = get_small_image(id_ls[1])
    if im2 != -1:
        try :
            st.image(im2, width=None)
        except:
            pass
    if st.button(ls[2],key = 'c3'):
        state.new_ = True
        state.new_title = ls[2]
        state.id = int(state.data[state.data['title'] == ls[2]]['id'])
        rerun()
    im3 = get_small_image(id_ls[2])
    if im3 != -1:
        try :
            st.image(im3, width=None)
        except:
            pass
            
    if st.button(ls[3],key = 'c4'):
        state.new_ = True
        state.new_title = ls[3]
        state.id = int(state.data[state.data['title'] == ls[3]]['id'])
        rerun()
    im4 = get_small_image(id_ls[3])
    if im4 != -1:
        try :
            st.image(im4, width=None)
        except:
            pass
            
    if st.button(ls[4],key = 'c5'):
        state.new_ = True
        state.new_title = ls[4]
        state.id = int(state.data[state.data['title'] == ls[4]]['id'])
        rerun()
    im5 = get_small_image(id_ls[4])
    if im5 != -1:
        try :
            st.image(im5, width=None)
        except:
            pass
def rerun():
    raise st.script_runner.RerunException(st.script_request_queue.RerunData(None))
    
def main():
    """Movie recommendation using word2vec"""
    html1 = '''<body style>
    <h1 style="font-family:georgia;color:#FFD700;font-size:60px;">MOVIE RECOMMENDATION SYSTEM<i style="color:white;"> (Using Word2Vec)</i></h1>
    </body>'''
    
    df,titles = load_data()
    state.data = df
    st.markdown(html1,unsafe_allow_html = True)
    activities = ["How to use","Search Movie","About"]  
    choice = st.sidebar.selectbox("Select Activitiy",activities)
    
    if choice == 'Search Movie':
        state.embeddings = get_vectors()
        new_selected = st.selectbox("Enter Movie Title",titles,index = 0,key = 's1')
        pressed =  st.button("Search",key = 'b1')
        if pressed:
            if new_selected == "":
                state.title = None
                state.new_ = False
                st.markdown('### Please enter a valid title')
            else :
                state.title = new_selected
                state.new_title = state.title
                state.id = int(state.data[state.data['title'] == new_selected]['id'])
                state.new_ = False
            rerun()
        if state.title is not None:
            if state.new_ == True:
                state.title = state.new_title
                state.id = int(state.data[state.data['title'] == state.new_title]['id'])
            display_data(state.id)
            st.markdown('''<body style>
        <h1 style="font-family:georgia;color:#FFD700;font-size:20px;">You may also like -</h1>
        </body>''',unsafe_allow_html= True)
            ls,id_ls,run = get_recomm(state.title)
            if run == True:
                state.ls = ls
                state.id_ls = id_ls
                run = False
                display_recomm(ls,id_ls)
            elif run == False:
                display_recomm(state.ls,state.id_ls)
         
    
    elif choice == "How to use":
        st.markdown("## How to use")
        st.markdown("Hi welcome to this application.")
        st.markdown("* Search your movie in the search bar provided in the 'Search Movie' option.")
        st.markdown("* To select a recommendation simply click on the button which have the recommendation name on it.")
    elif choice == "About":
        st.write("Hi! My name is Himalaya Sheoran. I created this movie recommendation system using word2vec.")
        st.write("Please e-mail me your feedback.")
        st.write("Also if you have ideas about some more features. Please mail them to me. I'll be happy to add them :)")
        st.write("email - sheoran26800@gmail.com")
        st.write("Linkedin - https://www.linkedin.com/in/himalaya-singh-5747061b0/")

if __name__ == '__main__':
    main()
            

        
