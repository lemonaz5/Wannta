import pickle
import numpy as np
import os
import subprocess
from pythainlp.tokenize import word_tokenize

# def word_tokenize(a, engine=''):
#     return []

class talkWithMe:
    sentenceVectorFile = '../senVec-30k.txt'
    sentenceDictFile = '../sen2vec-30k.pkl'
    tokenizeFile = '../tokenized_out-30k.txt'
    sentencesFile = '../sentences-30k.txt'

    def __init__(self):
        self.prepare_memory()
        self.sentenceDatabase, self.idx2sen, self.sen2vec = self.prepare_model()

    def save_object(self, obj, filename):
        with open(filename, 'wb') as output:
            pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

    def load_object(self, filename):
        r = {}
        with open(filename, 'rb') as f:
            r = pickle.load(f)
        return r
    
    def getSentenceVector(self, file = None, text = None, Format = True):
        args = ["/data2/fasttext/fasttext", "print-sentence-vectors", "/data2/cc.th.300.bin"]
        
        if text != None:
            popen = subprocess.Popen(args,stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            output = popen.communicate(text.encode())[0]
            popen.kill()
            return np.array([line.split(' ')[:-1] for line in output.decode('utf8').split('\n')[:-1]], dtype = np.float)
        
        elif file != None:
            f = open(file)
            o = open(self.sentenceVectorFile, 'w')
            popen = subprocess.Popen(args,stdin=f, stdout=o)   
            popen.wait()
            f.close()
            o.close()
            popen.kill()
            return 

    def sentenceTokenize(self, inputSentence):
        tokenized = word_tokenize(inputSentence)
        newTokenize = []
        for w in tokenized:
            newTokenize += word_tokenize(w, engine='newmm')
        return " ".join(newTokenize)
    
    def prepare_memory(self):
        sentences = []
        with open(self.tokenizeFile, 'r') as fp:
            for idx, line in enumerate(fp):
                sentences.append(" ".join(line.strip().split('|')))
        with open(self.sentencesFile, 'w') as fp:
            for idx, sen in enumerate(sentences):
                if idx % 2 != 0:
                    fp.write("{}\n".format(sen))
        self.getSentenceVector(file=self.sentencesFile)
    
    def prepare_model(self):
        sentencesTokenized = []
        with open(self.tokenizeFile, 'r') as file:
            for line in file:
                sentencesTokenized.append("".join(line.strip().split("|")))

        if os.path.isfile(self.sentenceDictFile):
            sen2vec = self.load_object(self.sentenceDictFile)
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
            with open(self.sentenceVectorFile, 'r') as file:
                for idx, line in enumerate(file):
                    vector = line.strip().split(' ')[-300:]
                    sentence = sentencesTokenized[idx]
                    sen2vec[sentence] = list(map(lambda x: float(x), vector))
            self.save_object(sen2vec, self.sentenceDictFile)
            idx2sen = {}
            for idx, sen in enumerate(sen2vec):
                idx2sen[idx] = sen

            sentenceDatabase = np.zeros((len(sen2vec), 300))
            for i in range(len(sen2vec)):
                for j in range(300):
                    sentenceDatabase[i][j] = sen2vec[idx2sen[i]][j]
            return sentenceDatabase, idx2sen, sen2vec
    
    def talkVec(self, inputSentenceVector):
        inputAb = np.linalg.norm(inputSentenceVector)
        output = self.sentenceDatabase.dot(inputSentenceVector)
        for i in range(self.sentenceDatabase.shape[0]):
            if ((np.linalg.norm(self.sentenceDatabase[i])) * inputAb) == 0:
                output[i] = 0
            else:
                output[i] /= ((np.linalg.norm(self.sentenceDatabase[i])) * inputAb)
        sumAll = np.sum(output)
        raw_output = np.argmax(output)
        output = output / sumAll
        outIdx = np.argmax(output)
        print( self.idx2sen[outIdx] + " = ",raw_output)
        if raw_output < 0.35:
            return "noMatch"
        else:
            return self.idx2sen[outIdx]
    
    def talk(self, inputText):
        inputText = self.sentenceTokenize(inputText)
        sent_vec = self.getSentenceVector(text=inputText)[0]
        #print(sent_vec)
        return self.talkVec(sent_vec)


bot = talkWithMe()
print(bot.talk("gwllo"))