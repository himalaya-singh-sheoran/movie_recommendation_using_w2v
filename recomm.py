import streamlit as st
from streamlit.hashing import _CodeHasher
from streamlit.report_thread import get_report_ctx
from streamlit.server.server import Server


from ast import literal_eval
import pandas as pd 
import numpy as np
from PIL import Image
import requests
import time

import pickle5 as pickle
import sqlite3

import streamlit.report_thread as ReportThread

@st.cache(persist=True,show_spinner=False)
def load_data():
	df = pd.read_csv('embedsData.txt')
	df.set_index("movieID",drop=True,inplace=True)
	dft = pd.read_csv('new_final.csv')
	return df,dft,list(st.session_state.title2id.keys())
@st.cache(persist=True)
def load_model():
	pickle_in = open('KNNC.pkl','rb')
	model = pickle.load(pickle_in)
	pickle_in.close()

	with open('id2val.pkl', 'rb') as handle:
		id2val = pickle.load(handle)
	with open('val2id.pkl', 'rb') as handle:
		val2id = pickle.load(handle)
	with open('id2title.pkl', 'rb') as handle:
		id2title = pickle.load(handle)
	with open('title2id.pkl', 'rb') as handle:
		title2id = pickle.load(handle)
		
	return model,id2val,val2id,id2title,title2id

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
		slice_ = pd.DataFrame(st.session_state.df1[st.session_state.df1['id'] == _id])
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
		st.markdown(f"####  Rating : \n{str(data['vote_average'])}")

		_out =  ""
		for i in literal_eval(slice_.iloc[0]['spoken_languages']):
			_out += i + ", "
		_out = _out[:-2]
		st.markdown(f"####  Spoken Language(s) : \n{_out}")
		st.markdown(f"####  Runtime : \n {data['runtime']} minutes")

			

@st.cache(persist=True,show_spinner=False)
def get_recomm(ID):
	users_idx= st.session_state.df2.loc[st.session_state.id2val[ID]].values.reshape(1,300)
	distances , indices = st.session_state.model.kneighbors(users_idx,n_neighbors=5+1)
	ls = []
	id_ls = []
	for i in range(1,len(indices[0])):
		id_ls.append(st.session_state.val2id[indices[0][i]])
		ls.append(st.session_state.id2title[id_ls[-1]])
	return ls,id_ls,True
	
@st.cache(persist=True,show_spinner=False)
def get_small_image(_id):
	data_ = return_data(_id)
	if data_ == -1:
		return -1
	else :
		try:
			poster_path = data_['poster_path']
			path = "https://image.tmdb.org/t/p/original/{}".format(poster_path)
			im = Image.open(requests.get(path, stream=True).raw)
			im = im.resize((100, 150))
		except:
			return -1
		return im


def display_recomm(ls,id_ls):
	col1, col2, col3, col4, col5 = st.beta_columns(5)
	with col1:
		im1 = get_small_image(id_ls[0])
		try:
			st.image(im1, width=None)
		except:
			st.image(Image.new('RGB', (100, 150), color = (0, 0, 0)))
		if st.button(ls[0],key = 'c1'):
				st.session_state.new_ = True
				st.session_state.new_title = ls[0]
				st.session_state.id = int(id_ls[0])
				rerun()

	with col2:	
		im2 = get_small_image(id_ls[1])
		try:
			st.image(im2, width=None)
		except:
			st.image(Image.new('RGB', (100, 150), color = (0, 0, 0)))
		if st.button(ls[1],key = 'c2'):
			st.session_state.new_ = True
			st.session_state.new_title = ls[1]
			st.session_state.id = int(id_ls[1])
			rerun()

	with col3:
		im3 = get_small_image(id_ls[2])
		try:
			st.image(im3, width=None)
		except:
			st.image(Image.new('RGB', (100, 150), color = (0, 0, 0)))
		if st.button(ls[2],key = 'c3'):
			st.session_state.new_ = True
			st.session_state.new_title = ls[2]
			st.session_state.id = int(id_ls[2])
			rerun()
		
	with col4:	
		im4 = get_small_image(id_ls[3])	
		try:
			st.image(im4, width=None)
		except:
			st.image(Image.new('RGB', (100, 150), color = (0, 0, 0)))
		if st.button(ls[3],key = 'c4'):
			st.session_state.new_ = True
			st.session_state.new_title = ls[3]
			st.session_state.id = int(id_ls[3])
			rerun()
		
		
	with col5:	
		im5 = get_small_image(id_ls[4])	
		try:
			if im5 != -1:
				try :
					st.image(im5, width=None)
				except:
					st.markdown("#### Sorry Poster not available ")
		except Exception as ex:
			st.write(ex)
		if st.button(ls[4],key = 'c5'):
			st.session_state.new_ = True
			st.session_state.new_title = ls[4]
			st.session_state.id = int(id_ls[4])
			rerun()

		
		
def rerun():
	raise st.script_runner.RerunException(st.script_request_queue.RerunData(None))

def search_movie():
	st.session_state.searchMovieIdx = 1
	new_selected = st.selectbox("Enter Movie Title",st.session_state.titles,index = 0,key = 's1')
	pressed =  st.button("Search",key = 'b1')
	if pressed:
		if new_selected == "":
			st.session_state.title = None
			st.session_state.new_ = False
			st.markdown('### Please enter a valid title')
		else :
			st.session_state.title = new_selected
			st.session_state.new_title = st.session_state.title
			st.session_state.id = int(st.session_state.title2id[new_selected])
			st.session_state.new_ = False
			rerun()
	if st.session_state.title is not None:
		if st.session_state.new_ == True:
			st.session_state.title = st.session_state.new_title
			st.session_state.id = int(st.session_state.title2id[st.session_state.new_title])
		display_data(st.session_state.id)
		st.markdown('''<body style>
	<h1 style="font-family:georgia;color:#FFD700;font-size:20px;">You may also like -</h1>
	</body>''',unsafe_allow_html= True)
		with st.spinner("Getting recommendations "):
			ls,id_ls,run = get_recomm(st.session_state.id)
			st.session_state.ls = ls
			st.session_state.id_ls = id_ls
			run = False
			display_recomm(ls,id_ls)
	

def gotAPP():
	if st.session_state.get_model is None:
		st.session_state.get_model = 1
		st.session_state.model,st.session_state.id2val,st.session_state.val2id,st.session_state.id2title,st.session_state.title2id= load_model()

		df,dft,titles = load_data()
		st.session_state.df2 = df
		st.session_state.df1 = dft
		st.session_state.titles = titles
		
	html1 = '''<body style>
	<h1 style="font-family:georgia;color:#FFD700;font-size:60px;">MOVIE RECOMMENDATION SYSTEM<i style="color:white;"> (Using Word2Vec)</i></h1>
	</body>'''
	st.markdown(html1,unsafe_allow_html = True)
	activities = ["How to use","Search Movie","About","Return to Home"] 
	choice = st.sidebar.selectbox("Select Activitiy",activities,index=0)
	if choice == "How to use":
		st.session_state.searchMovieIdx = 0
		st.markdown("## How to use")
		st.markdown("Hi welcome to this application.")
		st.markdown("* Search your movie in the search bar provided in the 'Search Movie' option.")
		st.markdown("* To select a recommendation simply click on the button which have the recommendation name on it.")
	elif choice == 'Search Movie':
		search_movie()
	
	elif choice == "About":
		st.session_state.searchMovieIdx = 2
		st.write("Hi! My name is Himalaya Singh Sheoran. I created this movie recommendation system using word2vec.")
		st.write("Please e-mail me your feedback.")
		st.write("Also if you have ideas about some more features. Please mail them to me. I'll be happy to add them :)")
		st.write("email - sheoran26800@gmail.com")
		st.write("Linkedin - https://www.linkedin.com/in/himalaya-singh-5747061b0/")

	elif choice == "Return to Home":
		st.session_state.searchMovieIdx = 1
		st.session_state.return_home = True
		rerun()

def authentication():
	if st.session_state.user_id is not None and st.session_state.return_home==False:
		gotAPP()
	else:
		if st.session_state.logedin == True:
			menu = ["Login/Logout","SignUp","Back to App"]
		else:
			menu = ["Login/Logout","SignUp"]
		choice = st.sidebar.selectbox("Menu",menu,index=0)

		if choice == "Login/Logout":
			st.session_state.authenticationMenuIdx = 0
			st.markdown("Login/Logout Section")
			username = st.text_input("User Name",key ='ti1')
			password = st.text_input("Password",type='password',key = 'ti2')

			if st.button("Login",key = "b2"):
				st.session_state.dbc.c.execute("SELECT * FROM users WHERE username = :username",{"username":username})
				data = st.session_state.dbc.c.fetchone()
				passWORD = data[1]
				if st.session_state.logedin == True:
						st.warning("You are already logged in please logout first")
				elif len(data) != 0:
					if password == passWORD:
						st.success("Logged In as {}".format(username))
						st.balloons()
						st.session_state.return_home = False
						st.session_state.user_id = 1
						st.session_state.logedin = True
						time.sleep(1)
						rerun()
					else:
						st.warning("Incorrect Username/Password")
				else:
					st.warning("Incorrect Username not found")

			if st.session_state.logedin == True:
				if st.checkbox("Select to logout",key="c1"):
					if st.button("Confirm",key = '1239'):
						st.session_state['loadStateVars'] = True
						rerun()

		elif choice == "SignUp":
			st.session_state.authenticationMenuIdx = 1
			if st.session_state.logedin == True:
				st.warning("You are already logged in please logout first")

			username = st.text_input("User Name",key ='ti3')
			password = st.text_input("Password",type='password',key = 'ti6')
			confirm_password = st.text_input("Confirm Password",type='password',key = 'ti7')

			if password != confirm_password:
				st.warning("Passwords don't match")
			if st.button("Signup"):
				if st.session_state.logedin == True:
					st.warning("You are already logged in please logout first")
				elif password != confirm_password:
					st.warning("Passwords don't match")
				else:
					st.session_state.dbc.c.execute("SELECT * FROM users WHERE username = :username",{"username":username})
					data = st.session_state.dbc.c.fetchone()
					if data is None:
						st.session_state.dbc.c.execute("INSERT INTO users VALUES (:username,:password)",{"username":username,"password":password})
						st.session_state.dbc.conn.commit()
						st.success("You have successfully created a valid Account")
						st.info("Go to Login Menu to login")
					else:
						st.warning("Username already exists")
						

		elif choice == "Back to App":
			st.session_state.authenticationMenuIdx = 0
			st.session_state.return_home = False
			rerun()

def setStateVars():
	stateVarsF = ["return_home","logedin","new_","is_rerun"]
	for i in stateVarsF:
		st.session_state[i] = False

	stateVarsN = ["user_id","get_model","model","id2val","val2id","id2title","title2id","title","new_title","id"]
	for i in stateVarsN:
		st.session_state[i] = None

	stateVarsL = ["ls","id_ls","titles"]
	for i in stateVarsL:
		st.session_state[i] = []

	stateVarsL = ["df1","df2"]
	for i in stateVarsL:
		st.session_state[i] = {}
	stateVars0 = ["authenticationMenuIdx","searchMovieIdx"]
	for i in stateVars0:
		st.session_state[i] = 0
	st.session_state.dbc = connectDB()

class connectDB:
	def __init__(self):
		self.conn = sqlite3.connect("users.db",check_same_thread=False)
		self.c = self.conn.cursor()

def main():	
	if 'loadStateVars' not in st.session_state:
		st.session_state['loadStateVars'] = False
		setStateVars()
	elif st.session_state.loadStateVars:
		st.session_state['loadStateVars'] = False
		setStateVars()
	authentication()	

if __name__ == '__main__':
	main()
