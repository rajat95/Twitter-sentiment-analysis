#Contributors:
#Rajat Arora 2013A7PS104P : Implemented Maximum Entropy algorithm (file maxent.py) and file vector.py and training.py
#Anurag Prakash 2013A7PS061P  :Implemented Naive Bayes(file naive_bayes.py) and file preprocess.py 
#Gireek Bansal 2013A7PS094P : Implemented SVM(file svm_classifier.py) and file features.py

#code to extract features
#written by Gireek Bansal 2013A7PS094P
from preprocess import punctuations,detect_emoticon,preprocess
from vector import get_words,vectorize,get_word_vector
import pandas

#Reads csv data file
def get_data(csvfile,x = 0):
	data = pandas.read_csv(csvfile,sep = ',',index_col = False,header = 0)
	data.columns = ['sentiment','id','time','query','user','tweet']
	data = data.drop(data.columns[[1, 2,3,4]], axis=1)
	if x != 0:
		data['punctuation'] = data['tweet'].map(punctuations)
		data['emoticon'],data['sarcastic'] = zip(*data['tweet'].map(detect_emoticon))
	data['tweet'] = data['tweet'].apply(preprocess)
	return data
	
#Calculates sentiment_score based on word polarity obtained after initial classification
def sentiment_score(tweets,words,classifier):
	#input :
	#1. list of tweets
	#3. list of feature words
	#4. trained classifier as pickle file
	scores = []
	for tweet in tweets:
		vector = get_word_vector(words,tweet)[0]
		#print vector
		score = classifier.predict_proba(vector)[0][0]
		scores = scores+[score]
	return scores
	
def add_sentiment_score(data,words,classifier):
	data['score'] = sentiment_score(data['tweet'],words,classifier)
	return data
	
#print get_data('/home/rajat/dummy.csv')
