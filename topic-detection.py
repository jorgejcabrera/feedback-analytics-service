import pandas as pd
import nltk
import re 
import unidecode

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import ngrams

stopwords = set(stopwords.words('spanish'))
stemmer = PorterStemmer()

def loadReviewData(filepath):
	return pd.read_csv(filepath, sep='\t')

def removeSpecialCharacters(sentence):
	return re.sub('[\W_]+', ' ', sentence)

def normalizeSentence(sentence):
	sentence = unidecode.unidecode(sentence)
	return removeSpecialCharacters(sentence)

def tokenizeSentence(sentence):
	return nltk.word_tokenize(sentence)

def removeStopWords(sentence):
	tokenized_text = tokenizeSentence(sentence)
	filtered_tokenized_text = [word for word in tokenized_text if word not in stopwords]
	return " ".join(filtered_tokenized_text)

def cleanReview(sentence):
	sentenceNormalized = normalizeSentence(sentence)
	return removeStopWords(sentenceNormalized)

def createReviews(rows):
	allReviews = ""
	for index, row in rows.iterrows():
		reviewCleaned = cleanReview(str(row[2]).lower())
		allReviews += replaceByWordsStemming(reviewCleaned)
	return allReviews

def replaceByWordsStemming(sentence):
	words = sentence.split()
	for word in words:
		wordStemming = stemmer.stem(word)
		sentence = sentence.replace(word, wordStemming)
	return sentence

def countWordFrequency(reviews):
	tokens = [t for t in reviews.split()]
	return nltk.FreqDist(tokens)

def countNGramFrequency(reviews, n):
	ngram = createNGram(reviews, n)
	return nltk.FreqDist(ngram)

def show(fdist, terms):
	fdist.plot(terms,rotation='horizontal')

def createNGram(sentence, n):
	return ngrams(sentence.split(), n)


reviewData = loadReviewData('reviews.tsv')
reviews = createReviews(reviewData)
#wordFrequency = countWordFrequency(reviews)
#show(wordFrequency, 15)

ngramFrequency = countNGramFrequency(reviews, 4)
show(ngramFrequency, 15)