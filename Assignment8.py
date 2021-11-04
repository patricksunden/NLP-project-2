
#tags in the NLTK POS are found here: https://www.guru99.com/pos-tagging-chunking-nltk.html
#part-of-speech = sanaluokka

#In corpus linguistics, part-of-speech tagging (POS tagging or PoS tagging or POST),
#also called grammatical tagging is the process of marking up a word in a text (corpus)
#as corresponding to a particular part of speech, based on both its definition and its context.

#a text corpus is a large body of text.

#Many corpora are designed to contain a careful balance of material in one or more genres.
#Brown Corpus	Francis, Kucera	15 genres, 1.15M words, tagged, categorized
#Brown corpus source: http://korpus.uib.no/icame/brown/bcm.html
#NLTK data: https://www.nltk.org/nltk_data/
#NLTK corpus howto: https://www.nltk.org/howto/corpus.html

import nltk
from nltk import *
from nltk.corpus import brown
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from poem import poem1
import numpy as np
from collections import Counter
from preprocess import preprocess, lemmatize
import pandas as pd
import matplotlib.pyplot as plt
from ast import literal_eval
from scipy.optimize import curve_fit

processedBrown = preprocess(brown.words())
processedPoem = preprocess(word_tokenize(poem1))



def assignment_6(poem):
    '''average sense per line'''
    averageSenses = []
    wordSenseDataFrame = pd.DataFrame(columns = ['Word', 'Senses'])
    
    #split the unprocessed poem into lines
    lines = poem.split('\\')
    
    #number of senses
    lineList = []
    for line in lines:
        senses = []
        words = []
        line = preprocess(word_tokenize(line))      #tokenize and preprocess
        line = lemmatize(line)                      #lemmatized list
        lineList.append(line)
        for word in line:
            senses.append(countSenses(word))            #senses has senses for ONE LINE
            words.append(word)
            
        df = pd.DataFrame({'Word': words , 'Senses' : senses})
        wordSenseDataFrame = wordSenseDataFrame.append(df)
        try:
            averageSenses.append(sum(senses)/len(senses))   #averageSenses has avg for EVERY LINE
        except ZeroDivisionError:
            averageSenses.append(0)
        
    lineSenseDataFrame = pd.DataFrame({'Line' : lineList, 'Avg senses' : averageSenses})
    
    wordSenseDataFrame = wordSenseDataFrame.drop_duplicates(subset='Word')
    
    lineSenseDataFrame.to_excel("assignment6_lineSense.xlsx")
    wordSenseDataFrame.to_excel("assignment6_wordSense.xlsx")

def objective(x, a, b, c, d, e):
    return (a * x) + (b * x**2) + (c * x**3) + (d * x**4) + e
    
def assignment_7():
    '''code for curve fitting https://machinelearningmastery.com/curve-fitting-with-python/'''
    #read excel file and axis values
    dataframe = pd.read_excel("assignment6_wordSense.xlsx", usecols = [2])
    freq = dataframe['Senses'].value_counts()
    freq = freq.sort_index(ascending=True)
    y_values = freq.tolist()
    x_values = np.arange(0,len(y_values),1)
    
    #curve fit
    popt, _ = curve_fit(objective, x_values, y_values)
    a, b, c, d, e = popt
    plt.scatter(x_values, y_values)
    x_line = np.arange(min(x_values), max(x_values), 1)
    y_line = objective(x_line, a, b, c, d, e)
    
    #plot curve fit
    plt.plot(x_line, y_line, '--', color='red')
    plt.title("Polynomial curve fitting")
    plt.show()
    
    #plot sense frequency
    freq.plot(title="Sense Frequency of individual words", xlabel="Sense count", ylabel="Frequency")
    plt.show()
    
def assignment_8(poem):
    '''lexical diversity in the poem. LD = (adjectives+adverbs)/verbs'''
    lines = poem.split('\\')    #list of strings
    ld = []
    tags = []
    
    for line in lines:
        line = preprocess(word_tokenize(line))
        adj = 0
        adv = 0
        verb = 0
        #calculate adjectives, adverbs and verbs
        tags = nltk.pos_tag(line)
        for item in tags:
            if item[1] == 'JJ' or item[1] == 'JJR' or item[1] == 'JJS':
                adj = adj+ 1
            elif item[1] == 'RB' or item[1] == 'RBR' or item[1]=='RBS':
                adv =adv + 1
            elif item[1] == 'VBG' or item[1] == 'VBD' or item[1]=='VBN' or item[1] == 'VBP' or item[1]=='VBZ':
                verb = verb+ 1
        try:
            ld.append((adj+adv)/verb)
        except ZeroDivisionError:
            ld.append((adj+adv)) #assume that every sentence has at least one verb
    lexDiv = pd.DataFrame({'Line': lines, 'Lexical diversity' : ld})
    lexDiv.to_excel("assignment8_lexdiv.xlsx")
    lexDiv.plot()
    plt.title("Lexical Diversity by lines")
    plt.show()
    
    #bins
    bins = []
    binsize = (lexDiv['Lexical diversity'].max() - lexDiv['Lexical diversity'].min())/10
    binsize = round(binsize, 2)
    bins.append(lexDiv['Lexical diversity'].min())
    
    for i in range(0,10):
        roundedBin = round((bins[i] + binsize) , 2)
        bins.append(roundedBin)
    
    #histogram
    plt.hist(lexDiv['Lexical diversity'].tolist() , bins)
    plt.title('Histogram of lexical diversity')
    plt.xlabel('Lexical diversity')
    plt.ylabel('Frequency')
    plt.show()
    
    
    
    
#assigment_1()
#assignment_2()
#assignment_5(processedBrown, processedPoem)
#assignment_6(poem1)
#assignment_7()
#assignment_8(poem1)

#print(brown.words())