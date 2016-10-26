#Contributors:
#Contributors:
#Rajat Arora 2013A7PS104P : Implemented Maximum Entropy algorithm (file maxent.py) and file vector.py and training.py
#Anurag Prakash 2013A7PS061P  :Implemented Naive Bayes(file naive_bayes.py) and file preprocess.py
#Gireek Bansal 2013A7PS094P : Implemented SVM(file svm_classifier.py) and file features.py

#code to train the algorithms
#written by Rajat Arora 2013A7PS104P
from features import get_data,sentiment_score,add_sentiment_score
from vector import get_word_features,vectorize,get_words,naive_bayes_vector
import cPickle
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from maxent import Maxent
import numpy
from naive_bayes import NaiveBayesClassifier
from svm_classifier import SVM
data = get_data('shortdatabase.csv')
word_features = get_word_features(data['tweet'])
word_features = sorted(word_features)
word_features = sorted(word_features)
word_vector = vectorize(word_features,data['tweet'],data['sentiment'])

vector = []
labels = []
for example in word_vector:
	vector = vector+[example[0]]
	labels = labels+[example[1]]
print "Stage 1: Word Polarity"
print "training bayesian network"

words = get_words("features.txt")
bayes_vector = naive_bayes_vector(words,data['tweet'],data['sentiment'])
#print bayes_vector
NaiveBayesClassifier.train(bayes_vector)

#gnb =  BernoulliNB()
#gnb.fit(vector,labels)
#with open('classifier_bayes.pkl', 'wb') as fid:
#    cPickle.dump(gnb, fid)

print "training svm"

svc = SVM()
svm_vector = []
for v in vector:
	for i in range(0,len(v)):
		if(v[i] == True):
			v[i] = 1.0
		else:
			v[i] = 0.0
	svm_vector = svm_vector+[v]

svm_labels = map(lambda x:float(x),labels)
svc.fit(numpy.vstack(numpy.array(svm_vector)),numpy.hstack(svm_labels))

#svc = SVC()
#svc.fit(vector,labels)
#with open('classifier_svm.pkl','wb') as fid:
#	cPickle.dump(svc,fid)

print "training maximum entropy classifier"
max_entropy = Maxent()
#max_entropy = BernoulliNB()
#max_entropy = DecisionTreeClassifier()
max_entropy.fit(numpy.array(vector),numpy.array(labels))
with open('classifier.pkl', 'wb') as fid:
    cPickle.dump(max_entropy, fid)

print "beginning with second phase consisting of training based on emoticons, word polarity and punctuations"
new_data = get_data("sampledataset.csv",1)
with open('classifier.pkl', 'rb') as fid:
    classifier = cPickle.load(fid)
new_data = add_sentiment_score(new_data,words,classifier)

new_vector = []
new_labels = []
i = 0
new_list = new_data.values.tolist()
for example in new_list:
	temp = []
	temp = temp+[example[2:]]
	new_vector = new_vector+temp
	new_labels = new_labels+[example[0]]
#print new_vector
final = Maxent()
final.fit(numpy.array(new_vector),numpy.array(new_labels))
print "training finished"
#print new_data


data = get_data('testdata.csv')
test_word_features = get_word_features(data['tweet'])
test_word_features = sorted(word_features)
test_word_features = sorted(word_features)
test_word_vector = vectorize(word_features,data['tweet'],data['sentiment'])

test_vector = []
test_label = []
for example in test_word_vector:
	test_vector = test_vector+[example[0]]
	test_label = test_label+[example[1]]
#print len(test_label)  
correct = 0
incorrect = 0
for i in range(0,len(test_vector)):
	if max_entropy.predict(test_vector[i])[0] == test_label[i]:
		correct = correct+1
	else:
		incorrect = incorrect+1
print correct, incorrect
