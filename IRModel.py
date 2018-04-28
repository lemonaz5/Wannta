
# coding: utf-8

# # Implementing IR Model

# ### Setup commonly used function and constraint

# In[1]:

import pickle
import time
import numpy as np
import os
import sys
import subprocess
from pythainlp.tokenize import word_tokenize

def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    r = {}
    with open(filename, 'rb') as f:
        r = pickle.load(f)
    return r

sentenceVectorFile = 'senVec.txt'
sentenceDictFile = 'sen2vec.pkl'
tokenizeFile = 'tokenized_out.txt'
sentencesFile = 'sentences.txt'


# ### Prepare memory

# In[2]:

def getSentenceVector(file = None, text = None, Format = True):
    args = ["/data2/fasttext/fasttext", "print-sentence-vectors", "/data2/cc.th.300.bin"]
    
    if(text != None):
        popen = subprocess.Popen(args,stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        output = popen.communicate(text.encode())[0]
        popen.kill()
        return np.array([line.split(' ')[:-1] for line in output.decode('utf8').split('\n')[:-1]], dtype = np.float)
    
    elif(file != None):
        f = open(file)
        o = open(sentenceVectorFile, 'w')
        popen = subprocess.Popen(args,stdin=f, stdout=o)   
        popen.wait()
#         output = popen.stdout.read()
        f.close()
        o.close()
        popen.kill()
#         if(Format):
#             return np.array([line.split(' ')[:-1] for line in output.decode('utf8').split('\n')[:-1]])
#         else:
#             return output


# In[9]:



def sentenceTokenize(inputSentence):
    # Tokenize
    tokenized = word_tokenize(inputSentence)
    newTokenize = []
    for w in tokenized:
        newTokenize += word_tokenize(w, engine='newmm')
    return " ".join(newTokenize)


# In[4]:

def prepare_memory():
    sentences = []
    with open(tokenizeFile, 'r') as fp:
        for idx, line in enumerate(fp):
            sentences.append(" ".join(line.strip().split('|')))

    with open(sentencesFile, 'w') as fp:
        for idx, sen in enumerate(sentences):
            if idx%2 != 0:
                fp.write("{}\n".format(sen))
  
    getSentenceVector(file = sentencesFile)


# In[5]:

prepare_memory()


# ### Define prepare_model function
# By checking if file at 'sentenceDictFile' variable exist or not. If not it will create such file, otherwise it will load from file. The output of this function is sentenceDatabase which is a numpy array with dimension of length of all sentence in database by 300, each row is a sentence vector from precompiled fastText. Next is sen2vec is like sentenceDatabase but instead of index of number, the index is the sentence itself. And lastly idx2sen which connected the gap between the two previous mentioned variables.

# In[6]:

def prepare_model():
    sentencesTokenized = []
    with open(tokenizeFile, 'r') as file:
        for line in file:
            sentencesTokenized.append("".join(line.strip().split("|")))
    
    if os.path.isfile(sentenceDictFile):
        sen2vec = load_object(sentenceDictFile)
        idx2sen = {}
        for idx, sen in enumerate(sen2vec):
            idx2sen[idx] = sen
        
        sentenceDatabase = np.zeros((len(sen2vec), 300))
        for i in range(len(sen2vec)):
            for j in range(300):
                sentenceDatabase[i][j] = sen2vec[idx2sen[i]][j]
        return sentenceDatabase, idx2sen, sen2vec
    else:
        sen2vec = {}
        with open(sentenceVectorFile, 'r') as file:
            for idx, line in enumerate(file):
                vector = line.strip().split(' ')[-300:]
                sentence = sentencesTokenized[idx]
                sen2vec[sentence] = list(map(lambda x: float(x), vector))
        save_object(sen2vec, sentenceDictFile)
        idx2sen = {}
        for idx, sen in enumerate(sen2vec):
            idx2sen[idx] = sen
        
        sentenceDatabase = np.zeros((len(sen2vec), 300))
        for i in range(len(sen2vec)):
            for j in range(300):
                sentenceDatabase[i][j] = sen2vec[idx2sen[i]][j]
        return sentenceDatabase, idx2sen, sen2vec


# ### Define talkVec function
# This function compare inputSentenceVector (expected to be np array with dimension of (300, )) with rest of sentenceDatabase using consine similarity, and output the cloest sentence in database.

# In[7]:

def talkVec(sentenceDatabase, idx2sen, sen2vec, inputSentenceVector):
    inputAb = np.linalg.norm(inputSentenceVector,ord=1)
    output = sentenceDatabase.dot(inputSentenceVector)
    for i in range(sentenceDatabase.shape[0]):
        output[i] /= (np.linalg.norm(sentenceDatabase[i], ord=1))*inputAb
    sumAll = np.sum(output)
    output = output/sumAll
    outIdx = np.argmax(output)
    return idx2sen[outIdx]


# ### Define main function
# As of right now. this is for testing only

# In[11]:

def main():
    sentenceDatabase, idx2sen, sen2vec = prepare_model()
    print("type something")
    while(True):
        text = input()
        if(text == ''):
            break
        start = time.time()
        #print('tokenizing...')
        text = sentenceTokenize(text)
        #print('gen vector...')
        sent_vec = getSentenceVector(text = text)[0]
        print(text)
        print(talkVec(sentenceDatabase, idx2sen, sen2vec, sent_vec))
        print("time {}".format(time.time()-start))
          
main()
