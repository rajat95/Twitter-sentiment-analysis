#Contributors:
#Rajat Arora 2013A7PS104P : Implemented Maximum Entropy algorithm (file maxent.py) and file vector.py and training.py
#Anurag Prakash 2013A7PS061P  :Implemented Naive Bayes(file naive_bayes.py) and file preprocess.py 
#Gireek Bansal 2013A7PS094P : Implemented SVM(file svm_classifier.py) and file features.py

#code to vectorize tweets
#written by Rajat Arora 2013A7PS104P

#Create a list of all words and then choose those occuring in at least .2% tweets
def get_word_features(tweets):
	words = {}
	features = []
	for tweet in tweets:
		for word in tweet:
			if word in words:
				words[word]+=1
			else:
				words[word] = 1
			
	#arrange words by frequency
	
	for word in words:
			if words[word]>5:
				features = features+[word]
	f = open('features.txt','w')
	for item in features:
  		f.write("%s\n" % item)
	return features

#convert list containing words of tweets to vector form having true if a word obtained from above function is present and false otherwise
def get_word_vector(words,tweet,label = 2):
	vector = []
	for word in words:
		if word in tweet:
			vector.append(True)
		else:
			vector.append(False)
	return (vector,label)

#vectorize whole dataset	
def vectorize(words,tweets,labels):
	vectors = []
	for i in range(0,len(tweets)):
		vectors = vectors+[get_word_vector(words,tweets[i],labels[i])]
  	return vectors

#create file with words
def get_words(wordfile):
	features = [];
	f = open(wordfile,'r')
	features = f.readlines()
	for i in range(0,len(features)):
		features[i] = features[i][:-1]
	#print features
	return features

#Vector required for naive bayes
def naive_bayes_vector(words,tweets,sentiment):
	i=0
	ans=[]
	for tweet in tweets:
    		lis={}
    		for word in words:
        		if word in tweet:
            			lis[word]=True
        		else:
            			lis[word]=False
    		tweet_vector=[]
		tweet_vector.append(lis)
		tweet_vector.append(sentiment[i])
		tweet_vector = tuple(tweet_vector)
    		ans.append(tweet_vector)
    		i+=1
	return ans
	
#naive_bayes_vector(['a','b'],[["a",'b','c'],["a",'b']])

