#Contributors:
#Rajat Arora 2013A7PS104P : Implemented Maximum Entropy algorithm (file maxent.py) and file vector.py and training.py
#Anurag Prakash 2013A7PS061P  :Implemented Naive Bayes(file naive_bayes.py) and file preprocess.py 
#Gireek Bansal 2013A7PS094P : Implemented SVM(file svm_classifier.py) and file features.py

#code to preprocess the tweet
#written by Anurag Prakash 2013A7PS061P
import nltk
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

#following function reoves non ascii characters
def remove_non_ascii_1(text):
	return ''.join(i for i in text if ord(i)<128)

#Code to remove punctuation
def remove_punctuation(text):
	for c in string.punctuation:
		if c == '@':
			continue
		text  = text.replace(c,'')
	text = text.lower()
	return text
	
#Code to count consecutive exclamations	
def punctuations(tweet):
	count = 0
	max_count = 0
	for i in range(0,len(tweet)):
		if tweet[i] == '!':
			while(i<len(tweet) and tweet[i] == '!'):
				count = count+1
				i = i+1
			if(count>max_count):
				max_count = count
			i = i-1
			count = 0
	#print tweet
	#print max_count
	return max_count

#+1 for happy emoticon, -1 for sad, also returns number of sarcastic emoticons 			
def detect_emoticon(tweet):
	happy_emoticons = [':-)',':-D',':)',';-D','X-D','<3','^_^']
	sarcastic_emoticon = [';-p',':-p',';p',':p',':-|']
	sad_emoticon = [':-(',';-(',':(']
	num = 0
	cnt = 0
	for emoji in happy_emoticons:
		if emoji in tweet:
			num = num+1
	for emoji in sad_emoticon:
		if emoji in tweet:
			num = num-1
	for emoji in sarcastic_emoticon:
		if emoji in tweet:
			cnt = cnt+1
			
	return num,cnt

#code to split tweet into list of words and then stem those words and convert to lower case and remove urls,usernames and stopwords
def preprocess(tweet):
		elim=[]
		#print tweet
		#tweet = remove_punctuation(tweet)
		tweet = re.sub("[^a-zA-z@]"," ",tweet)
		tweet = tweet.lower()
        	word_list=tweet.split()
        	for word in word_list:
        		remove_non_ascii_1(word)
        	#removes username
        	for word in word_list:
        	    if word[0]=="@" :
        	        elim.append(word)
        	for word in elim:
        	    word_list.remove(word)
        	#removes url    
        	for word in word_list:
        	    if "http" in word:
        	        word_list.remove(word)
        	#stop word removal
        	stop = stopwords.words('english')
        	stem = nltk.PorterStemmer()
        	for word in word_list:
        			if word in stop:
        				word_list.remove(word)
        	#removing repeated letters
        	for i in range(0,len(word_list)):
        		temp = ""
        		count = 0
        		for j in range(0,len(word_list[i])-1):
        			if(word_list[i][j] == word_list[i][j+1]):
        				count = count+1
        			else:
        				count = 0
        			if(count<2):
        				temp = temp+word_list[i][j]
        		temp = temp+word_list[i][-1]
        		word_list[i] = temp
        	return word_list
        	
#a = preprocess("i loooooovvvveeeeeeeee artificial INTELLIGENCE")
#print a
