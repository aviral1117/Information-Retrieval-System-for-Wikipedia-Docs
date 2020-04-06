import nltk
import re
from bs4 import BeautifulSoup
import math
import pickle
from collections import defaultdict
from keras import preprocessing

store_ob = []

# parsing
f = open('wiki_05','r', encoding="utf8") #add appropriate path for the file
html_doc = f.read()
soup = BeautifulSoup(html_doc, 'html.parser')
to_be_parsed = soup.find_all('doc')

doc_list=[] # Final list of all docs in the file. Each element is (id,title,token_list)
for i in to_be_parsed:
    title = i.get('title')
    data = i.get_text().casefold()
    data = preprocessing.text.text_to_word_sequence(data, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ')
    id_ = i.get('id')
    one_doc = [id_,title,data]
    doc_list.append(one_doc)
store_ob.append(doc_list)

# Func to create posting list
def get_postings(doc_list):
    # Doc_list : All docs in the file as created above
    posting=dict()
    i=0
    for doc in doc_list:
        words=set(doc[2])
        for wd in words:
            if wd in posting:
                posting[wd][0]=posting[wd][0]+1
                posting[wd][1].append(i)
            else:
                posting[wd]=[1,[i]]
        i=i+1
    return posting
posting_list = get_postings(doc_list)
store_ob.append(posting_list)

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
    v=[] # vector of given token_list. Each item is a list with token and its score based on lnc.ltc scheme
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
doc_vec=[] # vector for all docs

for doc in doc_list:
  doc_vec.append(dict(get_vector(doc[2],1)))
store_ob.append(doc_vec)

# Spelling Correction
# Building 2-gram character level index
def get_2grams():
    bigram_index=dict() # Key: 2-gram , Value: All tokens that have that two gram
    i=0
    for doc in doc_list:
        words=set(doc[2])
        for wd in words:
            ng1_chars = set(nltk.ngrams(wd, n=2))
            for bi in ng1_chars:
                if bi in bigram_index:
                    bigram_index[bi].append(wd)
                else:
                    bigram_index[bi] = [wd]
    return bigram_index
store_ob.append(get_2grams())

# Champion List modified index/posting_list creation
def modified_get_postings(doc_list):
    posting=dict() # Key: token , value: A list of lists with each sub list having doc id and corresponding lnc score
    i=0
    for doc in doc_list:
        tfw = term_freq(doc[2]) # tf weight = 1+log(tfd)
        words=set(doc[2])
        for wd in words:
            if wd in posting:
                posting[wd][0]=posting[wd][0]+1
                posting[wd][1].append([i,tfw[wd]])
            else:
                posting[wd]=[1,[[i,tfw[wd]]]]
        i=i+1
    for posting_i in posting.values():
        #print(posting_i[1])
        posting_i[1].sort(reverse=True,key = lambda x: x[1])
    return posting 
store_ob.append(modified_get_postings(doc_list))

# Storing the required indexes and other data structures on disc
with open('index_data.pickle', 'wb') as f:
    pickle.dump(store_ob, f)