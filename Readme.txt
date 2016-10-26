Project Title: Sentiment Analysis in Twitter feeds (Mood Detection)

#Contributors:
#Rajat Arora 2013A7PS104P : Implemented Maximum Entropy algorithm (file maxent.py) and file vector.py and training.py
#Anurag Prakash 2013A7PS061P  :Implemented Naive Bayes(file naive_bayes.py) and file preprocess.py
#Gireek Bansal 2013A7PS094P : Implemented SVM(file svm_classifier.py) and file features.py

Following Python ibraries must be installed to run the system
1.NLTK with modules stopwords, PorterStemmer
2.SCikit
3.SNUmpy
4.Cvxopt
5.Scipy
6.Pandas

Following files are included
1. preprocess.py - Code to preprocess
2. vector.py - Code to generate vectors
3. features.py - Code to generate features
4. naive_bayes.py - Code to train Naive Bayes
5. maxent.py - Code to train Maximum Entropy Classifier
6. svm_classifier.py - Code to train SVM
7. training.py - To train all above mentioned classifier
8. features.txt - List of important words extracted (Automatically generated)
9  Dummy.csv - Small dummy dataset consisting of 200 tweets classifier into positive and negative
10. sampledatabase.csv - Dummy databae consisting of tweets labelled positive, negative and neutral used in second phase 
11. .pkl files - Pretrained model used for testing so that we don'y need to train again
