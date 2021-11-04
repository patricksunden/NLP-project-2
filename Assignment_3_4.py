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


def assignment_3():
	#CORPUS
	corpus_word = []
	corpus_word_count = []
	freq_count_pair = []

	#POEM
	poem_word = []
	poem_word_count = []
	poem_freq_count_pair = []
	
	#CORPUS
	with open('/Users/Patrick/Desktop/koulu/Python/NLP/assignment1_filtered_corpus_data.csv') as corpus_filtered_word_freq:
		corpus_csv_reader = csv.reader(corpus_filtered_word_freq, delimiter = ',')
		for i, j in corpus_csv_reader:
			freq_count_pair.append((i, j))
			corpus_word.append(i)
			corpus_word_count.append(j)

		#print("corpus_word: corpus_word_count: ", corpus_word, corpus_word_count)

	#POEM
	with open('/Users/Patrick/Desktop/koulu/Python/NLP/assignment2_lemmatized_words.csv') as poem_filtered_word_freq:
		poem_csv_reader = csv.reader(poem_filtered_word_freq, delimiter = ',')
		for i, j in poem_csv_reader:
			poem_freq_count_pair.append((i, j))
			poem_word.append(i)
			poem_word_count.append(j)

		#print("poem_word: poem_word_count: ", poem_word, poem_word_count)
	
	#CORPUS
	int_corpus_word_count_map = map(int, corpus_word_count)
	int_corpus_word_count = list(int_corpus_word_count_map)
	max_corpus_word_count = max(int_corpus_word_count)

	for i in range(0, len(corpus_word_count)):
		int_corpus_word_count[i] = int_corpus_word_count[i]/ max_corpus_word_count

	#print(int_corpus_word_count)

	#POEM
	int_poem_word_count_map = map(int, poem_word_count)
	int_poem_word_count = list(int_poem_word_count_map)
	max_poem_word_count = max(int_poem_word_count)


	for i in range(len(poem_word_count)):
		int_poem_word_count[i] = int_poem_word_count[i]/max_poem_word_count

	#print(int_poem_word_count)

	
	l=0
	p=0
	b=1
	n=5
	for k in range(6):
		print("Corpus top word frequencies {} - {}".format(b, n))
		for l in range(l, (l+5)):
			print("Corpus Word | Count: ", corpus_word[l], "|" , int_corpus_word_count[l])
		print("Poem top word frequencies {} - {}".format(b, n))
		for p in range(p, (p+5)):
			print("Poem Word | Count: ", poem_word[p], "|" , int_poem_word_count[p])
		l+=5
		b+=5
		n+=5
		print("")
	


	#save to csv and sort in decreasing order
	corpusFrame = pd.DataFrame({'Word': corpus_word, 'Frequency': int_corpus_word_count})
	corpusFrame = corpusFrame.sort_values(by= ['Frequency'], ascending=False)
	#corpusFrame.to_csv("assignment5_brown_data.csv")

	poemFrame = pd.DataFrame({'Word': poem_word, 'Frequency': int_poem_word_count})
	poemFrame = poemFrame.sort_values(by= ['Frequency'], ascending=False)
	#poemFrame.to_csv("assignment5_poem_data.csv")

	#matching bigrams and histogram: 
	percentages = []
	print("going in loop")
	for i in range(0, 30):
		slice1 = corpusFrame.iloc[(i*5):((i+1)*5)]
		slice2 = poemFrame.iloc[(i*5):((i+1)*5)]
		slice1 = slice1["Word"]
		slice2 = slice2["Word"]

		merged = slice1.append(slice2)
		merged = merged.duplicated()
		true_count = (merged.sum()/5)*100

		percentages.append(true_count)


	print(percentages)

	percentages_plot = []

	for i in range(0, len(percentages)):
		if len(percentages_plot) > 4:
			break
		percentages_plot.append(percentages[i])

	print(percentages_plot)
	y = [1, 2, 3, 4, 5]
	fig, ax = plt.subplots()

	plt.bar([1, 2, 3, 4, 5], percentages_plot)
	
	ax = plt.gca()
	ax.set_ylim([0, 100])
	ax.set_title("Histogram of similarities between Corpus and Poem words")
	ax.set_xticks(y)
	ax.set_xticklabels(labels = ["WORDS 1-5", "WORDS 6-10", "WORDS 11-15", "WORDS 16-20", "WORDS 21-25"])
	ax.set_xlabel("Top 1-25 words")
	ax.set_ylabel("Similarity of words (%)")


	plt.show()
	


def assignment_4():
	#CORPUS
	corpus_tag = []
	corpus_tag_count = []
	corpus_tag_count_pair = []

	#POEM
	poem_tag = []
	poem_tag_count = []
	poem_tag_freq_count_pair = []
	
	#CORPUS
	with open('/Users/Patrick/Desktop/koulu/Python/NLP/assignment1_POS_filtered_data.csv') as corpus_filtered_tag_freq:
		corpus_csv_reader = csv.reader(corpus_filtered_tag_freq, delimiter = ',')
		for i, j in corpus_csv_reader:
			corpus_tag_count_pair.append((i, j))
			corpus_tag.append(i)
			corpus_tag_count.append(j)

		#print("corpus_tag: corpus_tag_count: ", corpus_tag, corpus_tag_count)

	#POEM
	with open('/Users/Patrick/Desktop/koulu/Python/NLP/assignment2_lemmatized_tags.csv') as poem_filtered_tag_freq:
		poem_csv_reader = csv.reader(poem_filtered_tag_freq, delimiter = ',')
		for i, j in poem_csv_reader:
			poem_tag_freq_count_pair.append((i, j))
			poem_tag.append(i)
			poem_tag_count.append(j)

	
	#CORPUS
	int_corpus_tag_count_map = map(int, corpus_tag_count)
	int_corpus_tag_count = list(int_corpus_tag_count_map)
	max_corpus_tag_count = max(int_corpus_tag_count)

	for i in range(0, len(corpus_tag_count)):
		int_corpus_tag_count[i] = int_corpus_tag_count[i] / max_corpus_tag_count

	#POEM
	int_poem_tag_count_map = map(int, poem_tag_count)
	int_poem_tag_count = list(int_poem_tag_count_map)
	max_poem_tag_count = max(int_poem_tag_count)


	for i in range(len(poem_tag_count)):
		int_poem_tag_count[i] = int_poem_tag_count[i] / max_poem_tag_count

	
	l=0
	p=0
	b=1
	n=5
	for k in range(6):
		try:
			print("Corpus top tag frequencies {} - {}".format(b, n))
			for l in range(l, (l+5)):
				print("Corpus tag | Count: ", corpus_tag[l], "|" , int_corpus_tag_count[l])
				
			print("Poem top tag frequencies {} - {}".format(b, n))
			for p in range(p, (p+5)):
				print("Poem tag | Count: ", poem_tag[p], "|" , int_poem_tag_count[p])
				
			l+=5
			b+=5
			n+=5
			print("")
		except IndexError:
			continue
	


	#save to csv and sort in decreasing order
	corpusFrame = pd.DataFrame({'tag': corpus_tag, 'Frequency': int_corpus_tag_count})
	corpusFrame = corpusFrame.sort_values(by= ['Frequency'], ascending=False)
	#corpusFrame.to_csv("assignment5_brown_data.csv")

	poemFrame = pd.DataFrame({'tag': poem_tag, 'Frequency': int_poem_tag_count})
	poemFrame = poemFrame.sort_values(by= ['Frequency'], ascending=False)
	#poemFrame.to_csv("assignment5_poem_data.csv")

	#matching bigrams and histogram: 
	percentages = []
	print("going in loop")
	for i in range(0, 30):
		slice1 = corpusFrame.iloc[(i*5):((i+1)*5)]
		slice2 = poemFrame.iloc[(i*5):((i+1)*5)]
		slice1 = slice1["tag"]
		slice2 = slice2["tag"]

		merged = slice1.append(slice2)
		merged = merged.duplicated()
		true_count = (merged.sum()/5)*100

		percentages.append(true_count)


	print(percentages)

	percentages_plot = []

	for i in range(0, len(percentages)):
		if percentages[i] > 0:
			percentages_plot.append(percentages[i])



	y = [1, 2, 3, 4, 5]
	fig, ax = plt.subplots()

	plt.bar([1, 2, 3, 4, 5], percentages_plot)
	
	ax = plt.gca()
	ax.set_ylim([0, 100])
	ax.set_title("Histogram of similarities between Corpus and Poem POS-tags")
	ax.set_xticks(y)
	ax.set_xticklabels(labels = ["TAGS 1-5", "TAGS 6-10", "TAGS 11-15", "TAGS 16-20", "TAGS 21-25"])
	ax.set_xlabel("Top 1-25 POS tags")
	ax.set_ylabel("Similarity of POS tags (%)")


	plt.show()


#assigment_1(processedBrown)
#assignment_2(processedPoem)
#assignment_3()
#assignment_4()
#assignment_5(brown_bigrams)
#print(brown.words())