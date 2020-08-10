import operator
import random
import renom as rm
import numpy as np
import nltk
import MeCab
rm.cuda.set_cuda_active(True)

mecab = MeCab.Tagger('-Owakati')

jpn_freq = {}
eng_freq = {}

def analyze_jpn(sentence):
    ret = mecab.parse(sentence).split(' ')
    ret.remove('\n')
    return ret
        
def analyze_eng(sentence):
    return nltk.word_tokenize(sentence.lower())
    
jpn_sentences, eng_sentences = {}, {}
lines = open('./dataset/sentences.csv').read().split('\n')
for i in range(len(lines)):
    if lines[i] == '':
        continue
    index, lang, sentence = lines[i].split('\t')
    if lang == 'jpn':
        tokens = analyze_jpn(sentence)
        for token in tokens:
            jpn_freq[token] = jpn_freq.get(token, 0) + 1
        jpn_sentences[index] = tokens
    elif lang == 'eng':
        tokens = analyze_eng(sentence)
lines = open('./dataset/jpn_indices.csv').read().split('\n')

ordered_sentences = []

def get_min_freq(sentence, token_freq):
    min_freq = 1000000000000
    for token in sentence:
        min_freq = min(min_freq, token_freq[token])
    return min_freq
        
for i in range(len(lines)):
    if lines[i] == '':
        continue
    jpn_index, eng_index, _ = lines[i].split('\t')
    if jpn_index not in jpn_sentences:
        continue
    if eng_index == -1:
        continue
    if eng_index not in eng_sentences:
        continue
    
    jpn_sentence = jpn_sentences[jpn_index]
    eng_sentence = eng_sentences[eng_index]
    
    jpn_min_freq = get_min_freq(jpn_sentence, jpn_freq)
    eng_min_freq = get_min_freq(eng_sentence, eng_freq)
    score = jpn_min_freq * eng_min_freq
    tup = (score, jpn_min_freq, eng_min_freq, jpn_sentence, eng_sentence)
    ordered_sentences.append(tup)

ordered_sentences = sorted(ordered_sentences, key=operator.itemgetter(0), reverse=True)
jpn_file = open('./dataset/train_ja.txt', 'w')
eng_file = open('./dataset/train_en.txt', 'w')

data_size = 3000
sentences = ordered_sentences[:data_size]
random.shuffle(sentences)

for _, _, _, jpn_sentence, eng_sentence in sentences[:data_size]:
    jpn_file.write(' '.join(jpn_sentence) + '\n')
    eng_file.write(' '.join(eng_sentence) + '\n')
    
jpn_file.close()
eng_file.close()
class Translator(rm.Model):
    def __init__(self, src_filedir, tar_filedir, hidden_size=100):
        
        self.src_i2w, self.src_w2i, self.X_train = self.process_dataset(src_filedir)
        self.tar_i2w, self.tar_w2i, self.Y_train = self.process_dataset(tar_filedir, True)
        
        self.src_vocab_size = len(self.src_w2i)
        self.tar_vocab_size = len(self.tar_w2i)
        self.hidden_size = hidden_size
        
        # encoder
        self.l1 = rm.Embedding(hidden_size, self.src_vocab_size)
        self.l2 = rm.Lstm(hidden_size)

        # decoder
        self.l3 = rm.Embedding(hidden_size, self.tar_vocab_size)
        self.l4 = rm.Lstm(hidden_size)
        self.l5 = rm.Dense(self.tar_vocab_size)
    
    def encode(self, x):
        h = self.l2(x)
        return h

    def decode(self, y):
        h = self.l4(y)
        h = self.l5(h)
        return h

    def src_word2onehot(self,word):
        v = np.zeros(shape=(self.src_vocab_size,))
        v[self.src_w2i(word)] = 1
        return v
        
    def tar_word2onehot(self, word):
        v = np.zeros(shape=(self.tar_vocab_size,))
        v[self.tar_w2i[word]] = 1
        return v

    def truncate(self):
        self.l2.truncate()
        self.l4.truncate()
        
    def forward(self, src_seq, tar_seq):
        src_seq = src_seq[::-1] # reverse
        xi = [self.src_w2i[word] for word in src_seq] # input word to index 
        xi = np.array(xi).reshape(len(xi),1)
        xe = self.l1(xi) # index to vector(embedding)
        # encode
        for x in xe:
            h = self.encode(x.reshape(1,-1))
            
        # Let the initial state of the decoder's LSTM be the final state of the encoder's LSTM.
        self.l4._z = h
        self.l4._state = self.l2._state
        
        yi = [self.tar_w2i[word] for word in tar_seq] # input word to index 
        yi = np.array(yi).reshape(len(yi),1)
        ye = self.l3(yi)
        loss = 0
        # decode
        for i in range(len(ye) - 1):
            y = ye[i].reshape(1,-1)
            yy = self.decode(y)
            d = self.tar_word2onehot(tar_seq[i+1])
            loss += rm.softmax_cross_entropy(yy.reshape(1, -1), d.reshape(1, -1))
        return loss
    
    def learn(self, optimizer=rm.optimizer.Adam(), epoch=100):
        
        predict_sentence = self.X_train[len(self.X_train) // 2]
        
        for i in range(epoch):
            total_loss = 0
            N = self.X_train.shape[0]
            perm = np.random.permutation(N)
            
            for j in range(N):
                index = perm[j]
                X, Y = self.X_train[index], self.Y_train[index]
                with self.train():
                    loss = self.forward(X, Y)
                loss.grad().update(optimizer)
                total_loss += loss.as_ndarray()
                self.truncate()
            print('Epoch %d - loss: %f' % (i, total_loss / N))
            print(predict_sentence)
            print(self.predict(predict_sentence))
                    
    def predict(self, src_seq, beam_width=10):
        src_seq = src_seq[::-1]
        xi = [self.src_w2i.get(word, self.src_w2i['<unk>']) for word in src_seq] # input word to index 
        xi = np.array(xi).reshape(len(xi),1)
        xe = self.l1(xi) # index to vector(embedding)
        # encode
        for x in xe:
            h = self.encode(x.reshape(1,-1))
            
        # decode
        cnt = 1
        limit = 100
        L = 0
        H = {}
        H['z'] = h
        H['state'] = self.l2._state
        word = '<bos>'
        sentence = [word]
        t = (L, sentence, H)
        Q = [t]
        is_all_eos = False
        while is_all_eos == False and cnt <= limit + 1: # limit + 1 for <'eos'>
            cand = list()
            is_all_eos = True
            for L, sentence, H in Q:
                self.l4._z = H['z']
                self.l4._state = H['state']
                word = sentence[-1]
                
                if word == '<eos>':
                    t = (L, sentence, H)
                    cand.append(t)
                else:
                    is_all_eos = False
                    yi = [self.tar_w2i[word]]
                    yi = np.array(yi).reshape(len(yi),-1)
                    ye = self.l3(yi)
                    y = ye.reshape(1,-1)
                    yy = self.decode(y)
                    p = rm.softmax(yy)
                    p = rm.log(p).as_ndarray()
                    p = p[0]
                    z = {}
                    z['z'] = self.l4._z
                    z['state'] = self.l4._state
                    for i in range(self.tar_vocab_size):
                        w = self.tar_i2w[i]
                        s = sentence + [w]
                        l = L + p[i]
                        t = (l, s, z)
                        cand.append(t)
                        
            cand = sorted(cand, key=lambda tup:tup[0], reverse=True)
            Q = cand[:beam_width]
            cnt += 1
        self.truncate()
        _, sentence, _ = Q[0]
        return sentence
    
    
    def process_dataset(self, filename, is_decorate=False):
        file = open(filename, 'r')
        vocabulary = []
        w2i = {}
        i2w = {}
        dataset = []
        for sentence in file.readlines():
            if is_decorate == True:
                words = ['<bos>'] + sentence.strip('\n').split(' ') + ['<eos>']
            else:
                words = sentence.strip('\n').split(' ')
            dataset.append(words)
            vocabulary += words
            
        vocabulary = sorted(set(vocabulary)) + ['<unk>']
        i2w = dict((i,c) for i,c in enumerate(vocabulary))
        w2i = dict((c,i) for i,c in enumerate(vocabulary))
        vocab_size=len(vocabulary)    
        return i2w, w2i, np.array(dataset)           
model = Translator('./dataset/train_ja.txt', './dataset/train_en.txt', 100)
model.learn(epoch=100)
for src in model.X_train[:10]:
    sentence = model.predict(src, beam_width=10)
    print(src)
    print(sentence)