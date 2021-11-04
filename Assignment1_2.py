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




def assigment_1(processedBrown):
	#assignment 1 graph of words in the corpus, (1)save the frequency of words in the corpus to a database
	# (2) Plot a graph about frequency of words in the corpus
	# (3) PLot a graph about the frequency of POS tags in the corpus
	#*****************************************
	#Word part of assignment 1
	word = []
	count = []
	tag_list = []
	tag_x = []
	tag_count = []
	
	fdist = FreqDist(processedBrown)
	print(fdist)
	words = fdist.most_common(50)
	print(words)
	#fdist.plot(50) #Would plot the 50 most common words


	for i in range(0, len(words)):
		word.append(words[i][0])
		count.append(words[i][1])
	
	#Plotting
	plt.plot(word, count)
	plt.title("Frequency of words in pre-processed corpus")
	plt.xticks(word, labels = word, rotation = 'vertical')
	plt.ylabel('Count of words')
	plt.xlabel('Word')
	plt.show()
	

	np.savetxt('assignment1_filtered_corpus_data.csv', np.c_[word, count], delimiter = ',', fmt = '%s')
	#*****************************************


	#Tag part of assignment 1
	#*****************************************
	tags = nltk.pos_tag(processedBrown)
	fdist_tag = FreqDist(tags)
	fdist_tag_most_common = fdist_tag.most_common()

	"""
	for i in range(0, len(fdist_tag_most_common)):
		tag_x.append(fdist_tag_most_common[i][0])
		
		tag_count.append(fdist_tag_most_common[i][1])
	"""

	tag_counts = Counter(tag for word, tag in tags)

	for i, j in tag_counts.items():
		tag_list.append((i, j))
	tag_list.sort(key=lambda x: (-x[1], x[0]))
	[t[1] for t in tag_list]

	#Appending the tags and counts into two different lists
	for i in range(0, len(tag_list)):
		tag_x.append(tag_list[i][0])
		tag_count.append(tag_list[i][1])


	np.savetxt('assignment1_POS_filtered_data.csv', np.c_[tag_x, tag_count], delimiter = ',', fmt = '%s')

	plt.plot(tag_x, tag_count)
	plt.title("Frequency of POS-tags in pre-processed corpus")
	plt.xticks(tag_x, labels = tag_x, rotation = 'vertical')
	plt.ylabel('Count of tags')
	plt.xlabel('Tag')
	plt.show()	
	#*****************************************

	def assignment_2(processedPoem):

	word_list = []
	count_list = []
	tag_list = []
	tag_x = []
	tag_count = []
	lemmatized_list = []

	wnl = WordNetLemmatizer()
	non_processed_poem = word_tokenize(poem1)


	for word in processedPoem:
		lemmatized_list.append(wnl.lemmatize(word))

	lemmatized_pos_tags = nltk.pos_tag(lemmatized_list)
	print("lemmatized_pos_tags: ", lemmatized_pos_tags)
	tag_counts = Counter(tag for lemmatized_list, tag in lemmatized_pos_tags)

	for i, j in tag_counts.items():
		tag_list.append((i, j))
	tag_list.sort(key=lambda x: (-x[1], x[0]))
	[t[1] for t in tag_list]

	for i in range(0, len(tag_list)):
		tag_x.append(tag_list[i][0])
		tag_count.append(tag_list[i][1])

	#Plotting the tag count of the poem
	"""
	plt.plot(tag_x, tag_count)
	plt.title("Frequency of POS-tags in unprocessed poem")
	plt.xticks(tag_x, labels = tag_x, rotation = 'vertical')
	plt.ylabel('Count of tags')
	plt.xlabel('Tag')
	plt.show()	
	"""
	
	#
	fdist = FreqDist(lemmatized_list)
	most_common_lemmatized = fdist.most_common(50)


	#SAVING THE LEMMATIZED WORDS IN A DATABASE
	
	for i in range(0, len(most_common_lemmatized)):
		word_list.append(most_common_lemmatized[i][0])
		count_list.append(most_common_lemmatized[i][1])
		
	np.savetxt('assignment2_lemmatized_words.csv', np.c_[word_list, count_list], delimiter = ',', fmt = '%s')
	
	#Plotting the word frequency of the poem
	
	plt.plot(word_list, count_list)
	plt.title("Frequency of words in processed poem")
	plt.xticks(word_list, labels = word_list, rotation = 'vertical')
	plt.ylabel('Count of words')
	plt.xlabel('Word')
	plt.show()
	
	
	#LAST 45 WORDS THAT ARE MENTIONED ONLY ONCE
	last_words = FreqDist(dict(fdist.most_common()[-45:]))
	print(last_words)
	last_words.plot() #Plots last 45 words that are mentioned once.
	

		
	#POS-tag part of the assignment#
	
	#SAVING THE POS TAGS INTO A DATABASE
	for i, j in tag_counts.items():
		tag_list.append((i, j))

	tag_list.sort(key=lambda x: (-x[1], x[0]))
	[t[1] for t in tag_list]

	for i in range(0, len(tag_list)):
		tag_x.append(tag_list[i][0])
		tag_count.append(tag_list[i][1])

	np.savetxt('assignment2_lemmatized_tags.csv', np.c_[tag_x, tag_count], delimiter = ',', fmt = '%s')
	


	"""
	#THE PLOT FOR MOST COMMON POS TAGS
	print("Tag_counts: ", tag_counts)
	plt.plot(tag_x, tag_count)
	plt.xticks(tag_x, labels = tag_x, rotation = 'vertical')
	plt.ylabel('Count of tags')
	plt.xlabel('Tag')
	plt.show()
	"""


#assigment_1(processedBrown)
#assignment_2(processedPoem)
#assignment_3()
#assignment_4()
#assignment_5(brown_bigrams)
#print(brown.words())