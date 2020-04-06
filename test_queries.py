# test_query.py
# Group 17
# Aviral Sethi 		- 2016B3A70532P
# Aditya P. Nahata 	- 2016B3A30502P
# Wikipedia file used: AH/wiki_05
import nltk
import math
import pickle
import sys 
import time
from keras import preprocessing
# command line arguments are stored in the form 
# of list in sys.argv 
argumentList = sys.argv 

path = sys.argv[2]
input_query = sys.argv[1]

''' Loading Data From Disc '''
with open(path,'rb') as f:
    metadata = pickle.load(f)
doc_list = metadata[0]  # A list of all docs
posting_list = metadata[1] # Initial Index
doc_vec = metadata[2] # A list of vectors of all docs in the corpus based on lnc scheme
bigram_pos = metadata[3] # bi-gram character level index on the corpus
m_posting_list = metadata[4] # modified posting list based on the term frequency for champion list creation

# Spelling Correction
def expand_query(query):
    # query is the token list of input query
    #bigram_pos = get_2grams() # bigram_pos is a mapping. 2gram --> a list of words/tokens which have the corresponding 2gram
    query_list=[]
    for wd in query:
        ng1_chars = set(nltk.ngrams(wd, n=2))
        #print(ng1_chars)
        top5={}
        for i in ng1_chars:
            if i in bigram_pos.keys():
                for sugg in bigram_pos[i]:
                    #print(bigram_pos[i])
                    ng2_chars = set(nltk.ngrams(sugg, n=2))
                    jc = float(len(ng1_chars & ng2_chars)/len(ng1_chars | ng2_chars))
                    if sugg not in top5:
                        top5[sugg] = jc
        #top 5 of this dict will be possible suggestions
        #python 3.6 required
        m = [(k,v) for k, v in sorted(top5.items(), key=lambda item: item[1],reverse=True)]
        suggestion=wd
        globalmin = len(wd)
        #print(m)
        for i in range(min(5,len(m))):
            score = nltk.edit_distance(wd,m[i][0])
            #print(score)
            if m[i][1]==1:
                #print("here")
                break
            if score < globalmin:
                globalmin=score
                suggestion = m[i][0]
        #replace the current processing word wd of query with suggestion
        query_list.append(suggestion)
    return query_list # best suggestion for every token of the query

def get_result(query_vec): # Returns cosine scores of docs w.r.t query vect
    score_mod=[] # Comparing with all docs ----> Calculating Cosine Score using champion list index
    for word in query_vec:
        if word[0] in posting_list:
            k = posting_list[word[0]][0] # First 5 entries of champion list index 
            for i in range(k):
                score_mod.append(posting_list[word[0]][1][i])
    score_mod = list(set(score_mod))
    score=[]
    for i in score_mod:
        vec = doc_vec[i]
        doc_score=0
        for word in query_vec:
            if word[0] in vec:
                doc_score+= word[1]*vec[word[0]]
        score.append((doc_score,i))
        i=i+1
    score.sort(reverse=True)
    return score

def print_result(score): # score is a list with doc id and score
    m=[]
    for k in range(min(10,len(score))):
        m.append(score[k][1])
        print("ID="+doc_list[score[k][1]][0]+" Title="+str(doc_list[score[k][1]][1])+  " Score="+str(score[k][0]))
    if len(score)<10:
        k = [i for i in range(len(doc_list)) if i not in m]
        print("The original retrieval system only returned ",len(score)," docs. This means that all other documents are equivalent and either have a score of 0.0 or were not selected because of champion list")
        j=0
        for i in range(10-len(score)):
            print("ID="+doc_list[k[j]][0]+" Title="+str(doc_list[k[j]][1])+  " Score= 0.0")
            j+=1

# Returns a dict with key as a token and value as the corresponding frequency as per logarithmic scheme (1+log(tfd))
def term_freq(token_list):
    counts = dict()
    for word in token_list:
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1
    for word,freq in counts.items():
        counts[word]=1+math.log10(counts[word])
    return counts

def get_vector(token_list,tag=0): # 0 for query
    tfw = term_freq(token_list)
    v=[] # vector of given list. Each item is a list with token and its score based on lnc.ltc scheme
    sum1=0 # if tag==1
    for i in tfw:
        z=tfw[i]
        v.append([i,z])
        sum1 = sum1+z*z
    sum1 = math.sqrt(sum1)
  
    if tag==0:
        sum2=0
        c=0
        for word in v:
            if word[0] in posting_list:
                word[1] = word[1]*(math.log10(len(doc_list)/posting_list[word[0]][0]))
            v[c][1] = word[1]
            sum2 = sum2 + v[c][1]*v[c][1]
            c+=1
        sum2 = math.sqrt(sum2)
        for i in range(len(v)):
            v[i][1]/=sum2
        return v
    else:
        for i in range(len(v)):
            v[i][1]/=sum1
        return v
        
def mergeDict(dict1, dict2):
   ''' Merge dictionaries and keep max values of common keys in list'''
   dict3 = {**dict1, **dict2}
   for key, value in dict3.items():
       if key in dict1 and key in dict2:
               dict3[key] = max(value,dict1[key])
   return dict3


# QUERY level Optimizations
## Spelling Check and Correction ---> Algorithm given in detail in report

# System level improvements
## Champion List Creation to save time

def get_results_after_improvements(new_query_vec,query_vec):
    score_mod=[] # Comparing with all docs ----> Calculating Cosine Score using champion list index
    for word in query_vec:
        if word[0] in m_posting_list:
            k = min(5,m_posting_list[word[0]][0]) # First 5 entries of champion list index 
            for i in range(k):
                score_mod.append(m_posting_list[word[0]][1][i][0])
    for word in new_query_vec:
        if word[0] in m_posting_list:
            k = min(5,m_posting_list[word[0]][0])
            for i in range(k):
                score_mod.append(m_posting_list[word[0]][1][i][0])
    score_mod = list(set(score_mod))
    score=[]
    for i in score_mod:
        vec = doc_vec[i]
        doc_score=0
        for word in query_vec:
            if word[0] in vec:
                doc_score+= word[1]*vec[word[0]]
        score.append((doc_score,i))
        i=i+1
    score.sort(reverse=True)
    return score

''' QUERY PROCESSING '''

query = input_query
print("\nInput Query= ",query)
query.casefold()
query_token_list = preprocessing.text.text_to_word_sequence(query, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ')
new_query_token_list = expand_query(query_token_list)
print("Before improvements... Top 10 Retrieved doc IDs and title for the query")

start_noimp = time.time() # To check performance
query_vec = get_vector(query_token_list,0)
result = get_result(query_vec)
end_noimp = time.time()
print_result(result)
print("Time Taken = {0:.8f}".format(end_noimp-start_noimp))

print("\nAfter improvements... Top 10 Retrieved doc IDs and title for the query")

start_imp = time.time()
new_query_vec = get_vector(new_query_token_list,0)
result_mod = get_results_after_improvements(new_query_vec,query_vec)
end_imp=time.time()
print_result(result_mod)
print("Time Taken = {0:.8f}".format(end_imp-start_imp))