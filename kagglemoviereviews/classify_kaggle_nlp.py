import os
import sys
import random
import nltk
from nltk.corpus import stopwords
import re
nltk.download('stopwords')
import sentiment_read_subjectivity
import sentiment_read_LIWC_pos_neg_words
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score
import crossval_nlp


#BAG OF WORDS feature:
#getting most common words in phrases:
def bow(phrase, most_freq_words):
  phrase = nltk.FreqDist(phrase)
  features = [word for (word, count) in phrase.most_common(most_freq_words)]
  return features


#Unigram features:
#This function generates dictionary of single word features
def unigram( phrase, word_features):
  phrase_words = set(phrase)
  features = {}

  for word in word_features:
    features['contains({})'.format(word)] = (word in phrase_words)
  return features


#Tokenized the phrases, turned to lowercase and removed stop words
def tokenize(phraselist):
  phrasedocs = []

  for phrase in phraselist:
    #Tokenization of phrases
    phrase_tokens = nltk.word_tokenize(phrase[0])
    #turning every word to lower case
    phrase_tokens = [w.lower() for w in phrase_tokens]
    #removing stopwords. Here we added some of our own stopwords which were not included in nltk
    nltkstopwords = nltk.corpus.stopwords.words('english')
    morestopwords = ['could', 'would', 'might', 'must', 'need', 'sha', 'wo', 'y', "'s", "'d", "'ll", "'t", "'m", "'re", "'ve"]
    #Final stopwords list
    stopwords = nltkstopwords + morestopwords
    phrase_tokens = [word for word in phrase_tokens if not word in stopwords]
    phrasedocs.append((phrase_tokens, int(phrase[1])))
  return phrasedocs


#substituting the short form of words with full form with the help of regex
def regex_clean(doc):
    regex = []

    for review_text in doc:
        review_text = re.sub(r"that 's","that is",review_text)
        review_text = re.sub(r"\bcannot\b", "can not", review_text)
        review_text = re.sub(r"\bain't\b", "am not", review_text)
        review_text = re.sub(r"\'ve","have",  review_text)
        review_text = re.sub(r"\bno\b", "not",review_text)
        review_text = re.sub(r"wo n't","will not", review_text)
        review_text = re.sub(r"do n't","do not", review_text)
        review_text = re.sub(r"\bdoesn't\b", "does not", review_text)
        review_text = re.sub(r"n\'t","not", review_text)
        review_text = re.sub(r"\'re","are",review_text)
        review_text = re.sub(r"\'d", "would", review_text)
        review_text = re.sub(r"\'ll", "will", review_text)
        review_text = re.sub(r"\'m", "am",review_text)
        review_text = re.sub(r"it 's","it is", review_text)
        regex.append(review_text)
    return regex


#removing non alphabetic characters
def alpha_filter(l):
    # filter out non-alphabetic words which don't have relevance

    for i in l:
      pattern = re.compile('^[^A-Za-z]+$')
      if pattern.match(i):
          continue
      return i


#using the poter stemming for normalizing
def stemmer(phraselist):
    phrase_stemmed = []

    if phraselist is not None:
        for phrase in phraselist:
            if phrase is not None and phrase[0] is not None:
                porter = nltk.PorterStemmer()
                stemmed_words = [porter.stem(t) for t in phrase[0]]
                phrase_stemmed.extend([(stemmed_word, int(phrase[1])) for stemmed_word in stemmed_words])
    return phrase_stemmed


#The sl_features function generates a dictionary of sentiment-related features including the count of positive and negative sentiment-related words 
def sl_features(phrase, processed_token, Sentiment_Lexicon):
  doc_words = set(phrase)
  features = {'positive':0, 'negative':0}

  for w in processed_token:
    features['V_{}'.format(w)] = (w in doc_words)
  
  for w in doc_words:
    if w in Sentiment_Lexicon:
      strength, posTag, isStemmed, polarity = Sentiment_Lexicon[w]
      if strength == 'weaksubj' and polarity == 'positive':
        features['positive'] += 1
      elif strength == 'strongsubj' and polarity == 'positive':
        features['positive'] += 2
      elif strength == 'weaksubj' and polarity == 'negative':
        features['negative'] += 1
      elif strength == 'strongsubj' and polarity == 'negative':
        features['negative'] += 2
  return features


#The pos function generates a dictionary of part-of-speech (POS)-related features in the phrase and the count of different POS categories (nouns, verbs, adjectives, and adverbs) in it
def pos(phrase, processed_token):
  unique_words = set(phrase)
  tagged_words = nltk.pos_tag(phrase)
  features = {}

  for word in processed_token:
    features['contains({})'.format(word)] = (word in unique_words)
  numNoun = 0
  numVerb = 0
  numAdj = 0
  numAdverb = 0
  
  for (word, tag) in tagged_words:
    if tag.startswith('N'): numNoun += 1
    if tag.startswith('V'): numVerb += 1
    if tag.startswith('J'): numAdj += 1
    if tag.startswith('R'): numAdverb += 1
  
  features['nouns'] = numNoun
  features['verbs'] = numVerb
  features['adjectives'] = numAdj
  features['adverbs'] = numAdverb
  return features


#The liwc function generates a dictionary of LIWC (Linguistic Inquiry and Word Count) related features based on the given phrase as well as the count of positive and negative words from the LIWC lists present
def liwc(phrase, processed_token,poslist,neglist):
  phrase_words = set(phrase)
  features = {'positive':0, 'negative':0}

  for word in processed_token:   
    features['contains({})'.format(word)] = (word in phrase_words)
  
  for word in phrase_words:
    if sentiment_read_LIWC_pos_neg_words.isPresent(word,poslist):
      features['positive']  += 1
    if sentiment_read_LIWC_pos_neg_words.isPresent(word,neglist):
      features['negative'] += 1
  return features


#The liwc_sl function generates a dictionary of LIWC (Linguistic Inquiry and Word Count) and sentiment-related features also it keep count of positive and negative words from the LIWC and sentiment lexicon present in the phrase
#basically it is combination of sl_features() and liwc().
def liwc_sl(phrase, processed_token, Sentiment_Lexicon, poslist, neglist):
  phrase_words = set(phrase)
  features = {'positive':0, 'negative':0}

  for w in processed_token:
    features['contains({})'.format(w)] = (w in phrase_words)
  
  for w in phrase_words:
    if sentiment_read_LIWC_pos_neg_words.isPresent(w, poslist):
      features['positive'] += 2
    elif sentiment_read_LIWC_pos_neg_words.isPresent(w, neglist):
      features['negative'] += 2
    elif w in Sentiment_Lexicon:
      strength, posTag, isStemmed, polarity = Sentiment_Lexicon[w]
      if strength == 'weaksubj' and polarity == 'positive':
        features['positive'] += 1
      elif strength == 'strongsubj' and polarity == 'positive':
        features['positive'] += 2
      elif strength == 'weaksubj' and polarity == 'negative':
        features['negative'] += 1
      elif strength == 'strongsubj' and polarity == 'negative':
        features['negative'] += 2
  return features

#path to the sentiment lexicon file
path = "C:\Drive D\Study\Syracuse\May- NLP\FinalProject_Data\FinalProjectData\kagglemoviereviews\SentimentLexicons\subjclueslen1-HLTEMNLP05.tff"
Sentiment_Lexicon = sentiment_read_subjectivity.readSubjectivity(path)

poslist,neglist = sentiment_read_LIWC_pos_neg_words.read_words()
poslist = poslist + regex_clean(poslist)  #using regex_clean() to convert the abbrevation to full form.
neglist = neglist + regex_clean(neglist)


# using linear regression classifier and random forest clasifier from sklearn to train our model and compare it with nltk naive bayes results
def sklearn(features,percent):
  training_size = int(percent*len(features))

  #Split the features into a training set and a test set
  train_set, test_set = features[training_size:], features[:training_size]

  #Train a Linear Regression classifier using the SklearnClassifier wrapper
  classifier1 = SklearnClassifier(SVC(kernel='linear'))
  classifier1.train(train_set)

  print("SVM Classification sklearn")
  accuracy, precision, recall, F1 = evaluation_metrics_SVM(train_set, test_set)
  print("Evaluation Metric: Accuracy={:.4f}, Precision={:.4f}, Recall={:.4f}, F1 Score={:.4f}".format(accuracy, precision, recall, F1))
  
  #Train a Random Forest classifier using the SklearnClassifier wrapper
  classifier2 = SklearnClassifier(RandomForestClassifier())
  classifier2.train(train_set)

  print("Random Forest sklearn")
  accuracy, precision, recall, F1 = evaluation_metrics_RF(train_set, test_set)
  print("Evaluation Metric: Accuracy={:.4f}, Precision={:.4f}, Recall={:.4f}, F1 Score={:.4f}".format(accuracy, precision, recall, F1))


#This function trains a Naive Bayes classifier on the given training set and computes evaluation metrics (accuracy, precision, recall, and F1 score) 
def evaluation_metrics(train_set, test_set):
  classifier = nltk.NaiveBayesClassifier.train(train_set)
  true_labels = []
  predicted_labels = []

  for features, label in test_set:
      predicted_label = classifier.classify(features)
      true_labels.append(label)
      predicted_labels.append(predicted_label)
  
  accuracy = nltk.classify.accuracy(classifier, test_set)
  precision = precision_score(true_labels, predicted_labels,average='weighted', zero_division=1)
  recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=1)
  F1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=1)
  return accuracy, precision, recall, F1


#This function trains a Random Forest classifier on the given training set and computes evaluation metrics (accuracy, precision, recall, and F1 score) 
def evaluation_metrics_RF(train_set, test_set):
  classifier = SklearnClassifier(RandomForestClassifier())
  classifier.train(train_set)
  true_labels = []
  predicted_labels = []

  for features, label in test_set:
      predicted_label = classifier.classify(features)
      true_labels.append(label)
      predicted_labels.append(predicted_label)
  
  accuracy = nltk.classify.accuracy(classifier, test_set)
  precision = precision_score(true_labels, predicted_labels,average='weighted', zero_division=1)
  recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=1)
  F1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=1)
  return accuracy, precision, recall, F1


#This function trains a SK learn  classifier on the given training set and computes evaluation metrics (accuracy, precision, recall, and F1 score)
def evaluation_metrics_SVM(train_set, test_set):
  classifier = SklearnClassifier(SVC(kernel='linear'))
  classifier.train(train_set)
  true_labels = []
  predicted_labels = []

  for features, label in test_set:
      predicted_label = classifier.classify(features)
      true_labels.append(label)
      predicted_labels.append(predicted_label)
  
  accuracy = nltk.classify.accuracy(classifier, test_set)
  precision = precision_score(true_labels, predicted_labels,average='weighted', zero_division=1)
  recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=1)
  F1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=1)
  return accuracy, precision, recall, F1


# our main function- processkaggle
def processkaggle(dirPath,limitStr):

  limit = int(limitStr)
  os.chdir(dirPath)
  f = open('./train.tsv', 'r')
  phrasedata = []
  
  for line in f:
    # ignore the first line starting with Phrase and read all lines
    if (not line.startswith('Phrase')):
      # remove final end of line character
      line = line.strip()
      # each line has 4 items separated by tabs
      # ignore the phrase and sentence ids, and keep the phrase and sentiment
      phrasedata.append(line.split('\t')[2:4])
  
  # pick a random sample of length limit because of phrase overlapping sequences
  random.shuffle(phrasedata)
  phraselist = phrasedata[:limit]
  print('Read', len(phrasedata), 'phrases, using', len(phraselist), 'random phrases')

  phrasedocs = []
  
  for phrase in phraselist:
    tokens = nltk.word_tokenize(phrase[0])
    phrasedocs.append((tokens, int(phrase[1])))
  

  #tokenize lower case phrases without stopwords ~ pwos:
  pwos = tokenize(phraselist)
  #adding alpha filters to our tokenized phrases
  final_phrases = [(alpha_filter(l), s) for l, s in pwos]
  #stemming the final phrasses to get our final output
  final_phrases = stemmer(final_phrases)
  #most common filtered features with complete preprocessing
  processed_features = bow(final_phrases, 8000)


  #unfiltered tokenized phrases:
  only_tokened = []
  for(w, s) in phrasedocs:
    only_tokened.extend(w)
  #most common unfiltered features which have only been tokenized
  only_tokenized_features = bow(only_tokened, 8000)


  #filtered featured sets:
  preprocessed_fs = [(unigram(token, processed_features), sentiment) for (token, sentiment) in phrasedocs]
  sl_preprocessed = [(sl_features(token, processed_features, Sentiment_Lexicon), sentiment) for (token, sentiment) in phrasedocs]
  pos_preprocessed = [(pos(token, processed_features), sentiment) for (token, sentiment) in phrasedocs]
  liwc_preprocessed = [(liwc(token, processed_features,poslist,neglist), sentiment) for (token, sentiment) in phrasedocs]
  sl_liwc_preprocessed = [(liwc_sl(token, processed_features,Sentiment_Lexicon, poslist, neglist), sentiment) for (token, sentiment) in phrasedocs]


  #unfiltered featured sets:
  tokenized_fs = [(unigram(token, only_tokenized_features), sentiment) for (token, sentiment) in phrasedocs]
  sl_tokenized = [(sl_features(token, only_tokenized_features, Sentiment_Lexicon), sentiment) for (token, sentiment) in phrasedocs]
  pos_tokenized = [(pos(token, only_tokenized_features), sentiment) for (token, sentiment) in phrasedocs]
  liwc_tokenized = [(liwc(token, only_tokenized_features,poslist,neglist), sentiment) for (token, sentiment) in phrasedocs]
  sl_liwc_tokenized = [(liwc_sl(token, only_tokenized_features,Sentiment_Lexicon, poslist, neglist), sentiment) for (token, sentiment) in phrasedocs]
  


  #Preprocessed_fs naive bayes classification
  print("_______________________ Filtered Feature Set Naive Bayes Classification _______________________")
  train_set_fs = preprocessed_fs[300:]
  test_set_fs =  preprocessed_fs[:300]
  print()
  print("Preprocessed_fs naive bayes:")
  accuracy, precision, recall, F1 = evaluation_metrics(train_set_fs, test_set_fs)
  print("Evaluation Metric: Accuracy={:.4f}, Precision={:.4f}, Recall={:.4f}, F1 Score={:.4f}".format(accuracy, precision, recall, F1))
 
  # sl_preprocessed naive bayes classification
  train_set_slp =  sl_preprocessed[300:]
  test_set_slp =  sl_preprocessed[:300]
  print()
  print("sl_preprocessed naive bayes:")
  accuracy, precision, recall, F1 = evaluation_metrics(train_set_slp, test_set_slp)
  print("Evaluation Metric: Accuracy={:.4f}, Precision={:.4f}, Recall={:.4f}, F1 Score={:.4f}".format(accuracy, precision, recall, F1))

  # pos_preprocessed naive bayes classification
  train_set_pos =  pos_preprocessed[300:]
  test_set_pos =  pos_preprocessed[:300]
  print()
  print("pos_preprocessed naive bayes:")
  accuracy, precision, recall, F1 = evaluation_metrics(train_set_pos, test_set_pos)
  print("Evaluation Metric: Accuracy={:.4f}, Precision={:.4f}, Recall={:.4f}, F1 Score={:.4f}".format(accuracy, precision, recall, F1))

  # liwc_preprocessed naive bayes classification
  train_set_liwc =  liwc_preprocessed[300:]
  test_set_liwc =  liwc_preprocessed[:300]
  print()
  print("liwc_preprocessed naive bayes:")
  accuracy, precision, recall, F1 = evaluation_metrics(train_set_liwc, test_set_liwc)
  print("Evaluation Metric: Accuracy={:.4f}, Precision={:.4f}, Recall={:.4f}, F1 Score={:.4f}".format(accuracy, precision, recall, F1))

  #  sl_liwc_preprocessed naive bayes classification
  train_set_slliwc =   sl_liwc_preprocessed[300:]
  test_set_slliwc =   sl_liwc_preprocessed[:300]
  print()
  print(" sl_liwc_preprocessed naive bayes:")
  accuracy, precision, recall, F1 = evaluation_metrics(train_set_slliwc, test_set_slliwc)
  print("Evaluation Metric: Accuracy={:.4f}, Precision={:.4f}, Recall={:.4f}, F1 Score={:.4f}".format(accuracy, precision, recall, F1))
  print()

  #UNFILTERED NAIVE BAYES:
  print("_______________________Unfiltered Feature Set Naive Bayes Classification_______________________")

  #tokenized_fs naive bayes:
  train_set_tfs =   tokenized_fs[300:]
  test_set_tfs=   tokenized_fs[:300]
  print()
  print(" tokenized_fs naive bayes:")
  accuracy, precision, recall, F1 = evaluation_metrics(train_set_tfs, test_set_tfs)
  print("Evaluation Metric: Accuracy={:.4f}, Precision={:.4f}, Recall={:.4f}, F1 Score={:.4f}".format(accuracy, precision, recall, F1))

  #sl_tokenized naive bayes:
  train_set_slt =   sl_tokenized[300:]
  test_set_slt =   sl_tokenized[:300]
  print()
  print(" sl_tokenized naive bayes:")
  accuracy, precision, recall, F1 = evaluation_metrics(train_set_slt, test_set_slt)
  print("Evaluation Metric: Accuracy={:.4f}, Precision={:.4f}, Recall={:.4f}, F1 Score={:.4f}".format(accuracy, precision, recall, F1))

  #pos_tokenized naive bayes:
  train_set_post =   pos_tokenized[300:]
  test_set_post =   pos_tokenized[:300]
  print()
  print(" pos_tokenized naive bayes:")
  accuracy, precision, recall, F1 = evaluation_metrics(train_set_post, test_set_post)
  print("Evaluation Metric: Accuracy={:.4f}, Precision={:.4f}, Recall={:.4f}, F1 Score={:.4f}".format(accuracy, precision, recall, F1))

  #liwc_tokenized naive bayes:
  train_set_liwct =   liwc_tokenized[300:]
  test_set_liwct =   liwc_tokenized[:300]
  print()
  print(" liwc_tokenized naive bayes:")
  accuracy, precision, recall, F1 = evaluation_metrics(train_set_liwct, test_set_liwct)
  print("Evaluation Metric: Accuracy={:.4f}, Precision={:.4f}, Recall={:.4f}, F1 Score={:.4f}".format(accuracy, precision, recall, F1))

  #sl_liwc_tokenized naive bayes:
  train_set_slliwct =   sl_liwc_tokenized[300:]
  test_set_slliwct =   sl_liwc_tokenized[:300]
  print()
  print(" sl_liwc_tokenized naive bayes:")
  accuracy, precision, recall, F1 = evaluation_metrics(train_set_slliwct, test_set_slliwct)
  print("Evaluation Metric: Accuracy={:.4f}, Precision={:.4f}, Recall={:.4f}, F1 Score={:.4f}".format(accuracy, precision, recall, F1))
  print()

  # sklearn
  print("_______________________Filtered Feature Set SkLearn Classification_______________________")
  print()
  print("preprocessed_fs Sklearn Evaluation Metrics : ")
  sklearn(preprocessed_fs, 0.3)
  print()
  print("sl_preprocessed Sklearn Evaluation Metrics : ")
  sklearn(sl_preprocessed, 0.3)
  print()
  print("pos_preprocessed Sklearn Evaluation Metrics : ")
  sklearn(pos_preprocessed, 0.3)
  print()
  print("liwc_preprocessed Sklearn Evaluation Metrics : ")
  sklearn(liwc_preprocessed, 0.3)
  print()
  print("sl_liwc_preprocessed Sklearn Evaluation Metrics : ")
  sklearn(sl_liwc_preprocessed, 0.3)
  print()

  print("_______________________Unfiltered Feature Set SkLearn Classification_______________________")
  print()
  print("tokenized_fs Sklearn Evaluation Metrics : ")
  sklearn(tokenized_fs, 0.3)
  print()
  print("sl_tokenized Sklearn Evaluation Metrics : ")
  sklearn(sl_tokenized, 0.3)
  print()
  print(" pos_tokenized Sklearn Evaluation Metrics : ")
  sklearn( pos_tokenized, 0.3)
  print()
  print("liwc_tokenized  Sklearn Evaluation Metrics : ")
  sklearn(liwc_tokenized , 0.3)
  print()
  print("sl_liwc_tokenized  Sklearn Evaluation Metrics : ")
  sklearn(sl_liwc_tokenized , 0.3)
  print()

  #CROSS VAL
  # labels = [0,1,2]
  labels = [c for (d,c) in phrasedocs]
  # calling the crossval_nlp file into this python file
  crossval_nlp.processkaggle('c:\Drive D\Study\Syracuse\May- NLP\FinalProject_Data\FinalProjectData\kagglemoviereviews\corpus' ,1000)


  print("_______________________CROSS-VAL Filtered random_forest_classification SkLearn Classification_______________________")
  print()
  # Processed RF
  print("preprocessed_fs Sklearn Evaluation Metrics : ")
  crossval_nlp.random_forest_classification(5, preprocessed_fs, labels) 
  print()
  print("sl_preprocessed Sklearn Evaluation Metrics : ")
  crossval_nlp.random_forest_classification(5, sl_preprocessed, labels)
  print()
  print("pos_preprocessed Sklearn Evaluation Metrics : ")
  crossval_nlp.random_forest_classification(5, pos_preprocessed, labels)
  print()
  print("liwc_preprocessed Sklearn Evaluation Metrics : ")
  crossval_nlp.random_forest_classification(5, liwc_preprocessed, labels)
  print()
  print("sl_liwc_preprocessed Sklearn Evaluation Metrics : ")
  crossval_nlp.random_forest_classification(5, sl_liwc_preprocessed, labels)
  print()

  print("_______________________CROSS-VAL Unfiltered random_forest_classification SkLearn Classification_______________________")
  print()
  # Un-processed RF
  print("tokenized_fs Sklearn Evaluation Metrics : ")
  crossval_nlp.random_forest_classification(5, tokenized_fs, labels)
  print()
  print("sl_tokenized Sklearn Evaluation Metrics : ")
  crossval_nlp.random_forest_classification(5, sl_tokenized, labels)
  print()
  print(" pos_tokenized Sklearn Evaluation Metrics : ")
  crossval_nlp.random_forest_classification(5, pos_tokenized, labels)
  print()
  print("liwc_tokenized  Sklearn Evaluation Metrics : ")
  crossval_nlp.random_forest_classification(5, liwc_tokenized, labels)
  print()
  print("sl_liwc_tokenized  Sklearn Evaluation Metrics : ")
  crossval_nlp.random_forest_classification(5, sl_liwc_tokenized, labels)
  

  
"""
commandline interface takes a directory name with kaggle subdirectory for train.tsv
   and a limit to the number of kaggle phrases to use
It then processes the files and trains a kaggle movie review sentiment classifier.

"""
if __name__ == '__main__':
    if (len(sys.argv) != 3):
        print ('usage: classifyKaggle.py <corpus-dir> <limit>')
        sys.exit(0)
    processkaggle(sys.argv[1], sys.argv[2])