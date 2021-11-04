import nltk
from nltk import *
from nltk.corpus import brown
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from poem import poem1
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from preprocess import preprocess
import pandas as pd
import csv


#Used for assignment 1 and 2, a pre-processed version of the brown corpus.
processedBrown = preprocess(brown.words())
#Used for assignment 5
brown_bigrams = list(bigrams(processedBrown))

processedPoem = preprocess(word_tokenize(poem1))


def assignment_5(corpus, poem):
	corpus_bigrams = []
	corpus_count = []
	poem_bigrams_list = []
	poem_count = []


	#get the frequency distribution
	brown_bigrams = list(bigrams(corpus))         #tokenize all bigrams
	poem_bigrams = list(bigrams(poem))

	fdist_corpus = FreqDist(brown_bigrams)
	fdist_poem = FreqDist(poem_bigrams)

	#differentiating the bigrams and frequencies for excel
	for bigram, number in fdist_corpus.items():
		corpus_bigrams.append(bigram)
		corpus_count.append(number)

	for bigramP, numberP in fdist_poem.items():
		poem_bigrams_list.append(bigramP)
		poem_count.append(numberP)


	#save to csv and sort in decreasing order
	corpusFrame = pd.DataFrame({'Bigram': corpus_bigrams, 'Frequency': corpus_count})
	corpusFrame = corpusFrame.sort_values(by= ['Frequency'], ascending=False)
	#corpusFrame.to_csv("assignment5_brown_data.csv")

	poemFrame = pd.DataFrame({'Bigram': poem_bigrams_list, 'Frequency': poem_count})
	poemFrame = poemFrame.sort_values(by= ['Frequency'], ascending=False)
	#poemFrame.to_csv("assignment5_poem_data.csv")

	#matching bigrams and histogram: 
	percentages = []
	print("going in loop")
	for i in range(0, 30):
		slice1 = corpusFrame.iloc[(i*5):((i+1)*5)]
		slice2 = poemFrame.iloc[(i*5):((i+1)*5)]
		slice1 = slice1["Bigram"]
		slice2 = slice2["Bigram"]

		merged = slice1.append(slice2)
		merged = merged.duplicated()
		true_count = (merged.sum()/5)*100

		percentages.append(true_count)

	print(percentages)



#assigment_1(processedBrown)
#assignment_2(processedPoem)
#assignment_3()
#assignment_4()
#assignment_5(brown_bigrams)
#print(brown.words())