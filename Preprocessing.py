import nltk
import re
from nltk import word_tokenize
import numpy as np
import pandas as pd
from nltk.util import ngrams
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import math

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer 
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import wordnet as wn
from nltk import word_tokenize, pos_tag
from collections import defaultdict

c = 'A' # Helper variable
f = open('wiki_05','r', encoding="utf8") #add appropriate path for the file
docs = f.read()
f.close()
Parsed_Text = BeautifulSoup(docs, 'html.parser').get_text()
raw=Parsed_Text.casefold()

 
# Function to generate n-grams from sentences.
def extract_ngrams(data, num):
    # Num : n for n gram
    # data : a list of data

    n_grams = ngrams(data, num)
    return list(n_grams)

# funtion to get X% of corpus limit
def required_num(ngram_list,num):
    # ngram_list: a list of n-grams
    # num : the required number of corpus coverage

    freqcount_ngrams = nltk.FreqDist(ngram_list)
    freqch=freqcount_ngrams.most_common(freqcount_ngrams.N())
    sum=0
    k=0
    for i in range(freqcount_ngrams.N()):
        sum+=freqch[i][1]
        if 100*sum/freqcount_ngrams.N()>=num:
            k=i
            break
    assert((100*sum/freqcount_ngrams.N()) >= num )
    return k+1

# function to generate graphs
def generate_graph(ngram_list,name):
    # ngram_list: A list of n-grams
    # name: used for graph naming
    #global c

    freqcount_ngrams = nltk.FreqDist(ngram_list)
    a = dict(freqcount_ngrams.items())
    b = [(i,j) for i,j in a.items()]
    b.sort(reverse=True, key = lambda x:x[1])
    
    x, y = zip(*b) # unpack a list of pairs into two tuples
    #print(len(x))
    x = range(len(x))
    x = [math.log10(y+1) for y in x] # scaling the ranks to log scale
    y = [math.log10(l) for l in y]
    plt.plot(x, y)
    plt.xlabel('Log Scaled Rank')
    plt.ylabel('Log Scaled Frequency of '+ name)
    plt.title('Frequency Distribution Graph')
    #plt.savefig(str(c)+'.png')  # Used for generating image files
    plt.show()
    #c = chr(ord(c)+1)

# function used for stemming
def stemming(ngram_list):

    # ngram_list = the original list of unigrams created. 'unigrams'
    ps = PorterStemmer() 
    post_stem_list = []
    for w in ngram_list: 
        post_stem_list.append(ps.stem(w))

    return post_stem_list # new stemmed unigram list is returned

def lemmatizer(ngram_list):

    # ngram_list = the original list of unigrams created. 'unigrams'
    lmtzr = WordNetLemmatizer()
    tagged = nltk.pos_tag(ngram_list)
    t_map = defaultdict(lambda : wn.NOUN)
    t_map['J'] = wn.ADJ
    t_map['V'] = wn.VERB
    t_map['R'] = wn.ADV

    post_lem_list=[]
    for token, tag in tagged:
        lemma = lmtzr.lemmatize(token, t_map[tag[0]])
        post_lem_list.append(lemma)

    return post_lem_list # new lemmatized unigram list is returned


# COLLOCATIONS

def calx2(bigram):
    # Calculates chi-square value for the bigram supplied as 'bigram'
    # helper1 = frequency dictionary for bigrams
    # helper2 = frequency distribution of unigrams
    o11 = helper1[bigram]
    o12 = (helper2[bigram[0]]-o11)
    o21 = (helper2[bigram[1]]-o11)
    o22 = n - o21 - o12 - o11
    x2 = n*(o11*o22-o12*o21)*(o11*o22-o12*o21)
    k = (o11+o12)*(o11+o21)*(o12+o22)*(o21+o22)

    return x2/k

def getdict(bi):
    k = {}
    for i in bi:
        a = calx2(i)
        if i not in k:
            k[i] = a
    return k # a dictionary with bigrams:chi-square value


# MAIN

tokenizer = RegexpTokenizer(r"[a-z0-9_A-Z]+\w'[a-z0-9_A-Z]+|[0-9]+[.][0-9]+|[\w]+")
unigram_list = tokenizer.tokenize(raw) # this is the main list

print("Without any processing (Q1,2,3)")

# These are not unique lists.
unigrams = unigram_list
bigrams = extract_ngrams(unigram_list,2)
trigrams = extract_ngrams(unigram_list,3)

print("No. of Unique Unigrams: ", len(set(unigrams)))
generate_graph(unigrams,'unigrams')
print("No. of most frequent unigrams required to cover 90% of corpus: ",required_num(unigrams,90))

print("No. of Unique bigrams: ", len(set(bigrams)))
generate_graph(bigrams,'bigrams')
print("No. of most frequent bigrams required to cover 80% of corpus: ",required_num(bigrams,80))

print("No. of Unique trigrams: ", len(set(trigrams)))
generate_graph(trigrams,'trigrams')
print("No. of most frequent trigrams required to cover 70% of corpus: ",required_num(trigrams,70))

print('\n')
print("Post Stemming Q4.")
unigrams_post_stem = stemming(unigrams)
bigrams_post_stem = extract_ngrams(unigrams_post_stem,2)
trigrams_post_stem = extract_ngrams(unigrams_post_stem,3)


print("No. of Unique Unigrams: ", len(set(unigrams_post_stem)))
generate_graph(unigrams_post_stem,'unigrams_post_stem')
print("No. of most frequent unigrams required to cover 90% of corpus: ",required_num(unigrams_post_stem,90))

print("No. of Unique bigrams: ", len(set(bigrams_post_stem)))
generate_graph(bigrams_post_stem,'bigrams_post_stem')
print("No. of most frequent bigrams required to cover 80% of corpus: ",required_num(bigrams_post_stem,80))

print("No. of Unique trigrams: ", len(set(trigrams_post_stem)))
generate_graph(trigrams_post_stem,'trigrams_post_stem')
print("No. of most frequent trigrams required to cover 70% of corpus: ",required_num(trigrams_post_stem,70))

print('\n')
print("Post Lemmatization Q5.")
unigrams_post_lem = lemmatizer(unigrams)
bigrams_post_lem = extract_ngrams(unigrams_post_lem,2)
trigrams_post_lem = extract_ngrams(unigrams_post_lem,3)


print("No. of Unique Unigrams: ", len(set(unigrams_post_lem)))
generate_graph(unigrams_post_lem,'unigrams_post_lem')
print("No. of most frequent unigrams required to cover 90% of corpus: ",required_num(unigrams_post_lem,90))

print("No. of Unique bigrams: ", len(set(bigrams_post_lem)))
generate_graph(bigrams_post_lem,'bigrams_post_lem')
print("No. of most frequent bigrams required to cover 80% of corpus: ",required_num(bigrams_post_lem,80))

print("No. of Unique trigrams: ", len(set(trigrams_post_lem)))
generate_graph(trigrams_post_lem,'trigrams_post_lem')
print("No. of most frequent trigrams required to cover 70% of corpus: ",required_num(trigrams_post_lem,70))

helper1 = dict(nltk.FreqDist(bigrams).items())
helper2 = dict(nltk.FreqDist(unigrams).items())

#print(helper1[('space', 'syntax')])
n = len(unigrams)
#print(n)
collocations = getdict(bigrams)
collocations1= [(k, v) for k, v in collocations.items()]

collocations1.sort(reverse=True,key=lambda x:x[1])

print(collocations1[:20])
