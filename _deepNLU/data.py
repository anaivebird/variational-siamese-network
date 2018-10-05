import os
import re
import nltk
import json
import numpy as np
import pandas as pd
import gensim as gensim
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
dir_ = '..'
#dir_ = '/ldaphome/mdeudon/NIPS' #os.getcwd()+'/deep_NLU' #####


""" Preprocess text """
def clean_sent(sentence):
    sentence = str(sentence).lower() # convert to lower case
    sentence = re.sub(' - ','-',sentence) # refit dashes (single words)
    for p in '/+-^*÷#!"(),.:;<=>?@[\]_`{|}~\'¿€$%&£±': # clean punctuation
        sentence = sentence.replace(p,' '+p+' ') # good/bad to good / bad
    sentence = sentence.strip() # strip leading and trailing white space
    tokenized_sentence = nltk.tokenize.word_tokenize(sentence) # nltk tokenizer
    #tokenized_sentence = [w for w in tokenized_sentence if w in printable] # remove non ascii characters
    return tokenized_sentence


""" Load Language Model (word <-> int mapping + w2vec representations) """
def load_w2v(w2v_size=300):
    saving_path = dir_+'/w2v/'+'embeddings_'+str(w2v_size)+'d.p'
    vocab_path = dir_+'/w2v/'+'vocab_'+str(w2v_size)+'d.p'

    if not os.path.exists(saving_path): # init language model
        print("\n Creating word_embeddings...")
        df = pd.read_csv(dir_+"/data/quora_duplicate_questions.tsv",delimiter='\t') # load data
        df1 = df[['question1']].rename(index=str, columns={"question1": "question"}) # questions 1
        df2 = df[['question2']].rename(index=str, columns={"question2": "question"}) # questions 2
        unique_questions = pd.concat([df1,df2]).question.unique() # unique questions
        corpus = list(unique_questions)
        print(' Collected',len(corpus),'unique sentences.') # 808 580 sentences --> 537 362 unique sentences

        corpus = list(map(clean_sent, corpus)) # preprocess text [clean_sent(sent) for sent in corpus]
        corpus.append(['UNK','UNK', 'UNK', 'UNK', 'UNK']) # unknown word
        corpus.append(['EOS','EOS', 'EOS', 'EOS', 'EOS']) # padding

        my_model = gensim.models.word2vec.Word2Vec(size=w2v_size, min_count=2, sg=1) # initialize W2V model: collect 87 116 word types from a corpus of 7 161 626 (+10) raw words 
        my_model.build_vocab(corpus) # 48 096 (+2) not unique words (min_count=2). --> 55% of word types / 99% of raw words
        my_model.intersect_word2vec_format(dir_+'/w2v/' + 'glove.6B.'+str(w2v_size)+'d.txt',binary=False) # update with GloVe: 44 373 retrieved (84%)

        weights = my_model.wv.syn0 # word embeddings
        np.save(open(saving_path, 'wb'), weights)
        vocab = dict([(k, v.index) for k, v in my_model.wv.vocab.items()]) # word mapping (dictionary)
        with open(vocab_path, 'w') as f:
            f.write(json.dumps(vocab))

    with open(saving_path, 'rb') as f:
        weights = np.load(f)
    print('\n Loaded Word_embeddings Matrix', weights.shape)

    with open(vocab_path, 'r') as f:
        data = json.loads(f.read())
    word2idx = data
    idx2word = dict([(v, k) for k, v in data.items()])
    print(' Loaded Vocabulary Mapping (size {})'.format(len(word2idx)))

    return weights, word2idx, idx2word  # return w2v weights + vocabulary mapping


weights, word2idx, idx2word = load_w2v()
pad_tok = word2idx['EOS'] # pad token


""" Map list of tokens to list of word id (wid) """
def sent2ids(tokenized_sentence):
    sent_ids = []
    for word in tokenized_sentence:
        try:
            sent_ids.append(word2idx[word])
        except:
            sent_ids.append(word2idx['UNK']) # Unknown words
    return np.asarray(sent_ids)


""" Load UNLABELED data as [[w_id]] """
def load_all(maxlen=40):
    folder = dir_+'/corpus/'
    saving_path = 'Q_id.npy'
    if not os.path.exists(folder+saving_path):
        print('\n Creating wid corpus (all)...')
        df = pd.read_csv(dir_+"/data/quora_duplicate_questions.tsv",delimiter='\t') # load UNLABELED data
        df1 = df[['question1']].rename(index=str, columns={"question1": "question"})
        df2 = df[['question2']].rename(index=str, columns={"question2": "question"})
        unique_questions = pd.concat([df1,df2]).question.unique() # unique questions
        print('\n Loaded {} unique questions (corpus)'.format(len(unique_questions)))
        corpus = list(unique_questions)
        corpus = list(map(clean_sent, corpus)) # preprocess text
        corpus_ids = list(map(sent2ids, corpus)) # map to wid 
        corpus_ids = dict(zip(np.arange(len(corpus_ids)), corpus_ids)) # zip
        np.save(folder+saving_path, corpus_ids)

    corpus_ids = np.load(folder+saving_path).item()  # A question = a list of word_index
    print("\n Loaded wid corpus (all)")
    return corpus_ids


corpus_ids = load_all()


""" Load LABELED data as [[w_id]] """
def load_split(name='train', maxlen=40):
    folder = dir_+'/corpus/'
    saving_path = '_'+name+'_id.npy'
    if not os.path.exists(folder+'Xs'+saving_path):
        print('\n Creating wid corpus ({})...'.format(name))
        df = pd.read_csv(dir_+"/data/split/"+name+".csv", sep=',') # load LABELED data
        df = df[['question1','question2','is_duplicate']]
        print('\n Loading Quora '+name+' dataset (',len(df),'question pairs)')

        df_true_duplicate = df[df['is_duplicate']==1] # duplicate questions
        Xs = list(df_true_duplicate['question1'].values)  # Xs[k] and Ys[k] are duplicates
        Ys = list(df_true_duplicate['question2'].values)
        print('\n Duplicate questions (',len(Xs),'pairs):')
        print(Xs[0],'\n',Ys[0])

        Xs = list(map(clean_sent, Xs)) # preprocess text
        Ys = list(map(clean_sent, Ys))
        Xs_ids = list(map(sent2ids, Xs)) # map to wid
        Ys_ids = list(map(sent2ids, Ys))
        Xs_ids = dict(zip(np.arange(len(Xs_ids)), Xs_ids)) # zip
        Ys_ids = dict(zip(np.arange(len(Ys_ids)), Ys_ids))
        np.save(folder+'Xs'+saving_path, Xs_ids)
        np.save(folder+'Ys'+saving_path, Ys_ids)

        df_false_duplicate = df[df['is_duplicate']==0] # NOT duplicate questions
        Xa = list(df_false_duplicate['question1'].values)  # Xa[k] and Ya[k] are NOT duplicates
        Ya = list(df_false_duplicate['question2'].values)
        print('\n Not duplicate questions (',len(Xa),'pairs):')
        print(Xa[0],'\n',Ya[0])

        Xa = list(map(clean_sent, Xa)) # preprocess text
        Ya = list(map(clean_sent, Ya))
        Xa_ids = list(map(sent2ids, Xa)) # map to wid
        Ya_ids = list(map(sent2ids, Ya))
        Xa_ids = dict(zip(np.arange(len(Xa_ids)), Xa_ids)) # zip
        Ya_ids = dict(zip(np.arange(len(Ya_ids)), Ya_ids))
        np.save(folder+'Xa'+saving_path, Xa_ids)
        np.save(folder+'Ya'+saving_path, Ya_ids)

    Xs_ids = np.load(folder+'Xs'+saving_path).item()
    Ys_ids = np.load(folder+'Ys'+saving_path).item()
    Xa_ids = np.load(folder+'Xa'+saving_path).item()
    Ya_ids = np.load(folder+'Ya'+saving_path).item()
    print("\n Loaded wid corpus ({})".format(name))
    return Xs_ids, Ys_ids, Xa_ids, Ya_ids


Xs_train_ids, Ys_train_ids, Xa_train_ids, Ya_train_ids = load_split(name='train') # train       NB: Xs,Ys --> 1 et Xa,Ya --> 0
Xs_dev_ids, Ys_dev_ids, Xa_dev_ids, Ya_dev_ids = load_split(name='dev') # dev
Xs_test_ids, Ys_test_ids, Xa_test_ids, Ya_test_ids = load_split(name='test') # test


""" Pad sequence """
def pad_sequence(ids, padlen=40, target_offset=False):
  if target_offset == False:
    return ids[:padlen] + [pad_tok] * max(padlen - len(ids),0), min(len(ids), padlen)
  else:
    return [pad_tok] + ids[:padlen-1] + [pad_tok] * max(padlen-1 - len(ids),0), min(len(ids), padlen) # shift decoder's input


""" Load generative corpus (repeat, reformulate) """
def load_VAD_corpus(corpus_ids, Xs_ids, Ys_ids, mode='VAE', padlen=40):            ##### keep questions / len(q) <= maxlen (repeat, reformulate)
    d1, d2 = {}, {}
    n1, n2 = len(corpus_ids), 0
    if mode == 'VAD':
        n2 += len(Xs_ids)

    for i in range(n1):
        d1[i] = pad_sequence(list(corpus_ids[i]), padlen=padlen) 
        d2[i] = pad_sequence(list(corpus_ids[i]), padlen=padlen, target_offset=True) # REPEAT (VAE): q -> q
    for i in range(n2):
        d1[n1+i] = pad_sequence(list(Xs_ids[i]), padlen=padlen)
        d2[n1+i] = pad_sequence(list(Ys_ids[i]), padlen=padlen, target_offset=True)  # REFORMULATE (VAD) q -> q'
    for i in range(n2):
        d1[n1+n2+i] = pad_sequence(list(Ys_ids[i]), padlen=padlen)
        d2[n1+n2+i] = pad_sequence(list(Xs_ids[i]), padlen=padlen, target_offset=True) # REFORMULATE (VAD) q' -> q'''

    print('\n Loaded generative corpus: {} sentence pairs ({} VAE / {} VAD)'.format(n1+2*n2, n1, 2*n2)) # n1+2*n2 = 502 530 pairs (len 12)
    return d1, d2, n1+2*n2 # VAD corpus (q, q')


""" Load discriminative corpus (paraphrase identification) """
def load_CLF_corpus(Xs_ids, Ys_ids, Xa_ids, Ya_ids, padlen=40):
    n1, n2 = len(Xs_ids), len(Xa_ids)
    d1, d2, d3 = {}, {}, {}
    for i in range(n1): # Duplicate
        d1[i] = pad_sequence(list(Xs_ids[i]), padlen=padlen)  # tuple tokens, seq_length
        d2[i] = pad_sequence(list(Ys_ids[i]), padlen=padlen) 
        d3[i] = 1
    for i in range(n2): # Not Duplicate
        d1[n1+i] = pad_sequence(list(Xa_ids[i]), padlen=padlen) 
        d2[n1+i] = pad_sequence(list(Ya_ids[i]), padlen=padlen) 
        d3[n1+i] = 0
    print('\n Loaded discriminative corpus: {} sentence pairs ({} duplicate / {} non duplicate)'.format(n1+n2, n1, n2))
    return d1, d2, d3, n1+n2 # CLF corpus (q1, q2, label)


""" Create batches """
def create_batches(data_size, batch_size=64, shuffle=True):
    batches = [] # create batches by index
    ids = np.arange(data_size)
    if shuffle:
        np.random.shuffle(np.asarray(ids))
    for i in range(np.floor(data_size / batch_size).astype(int)):
        start = i * batch_size
        end = (i + 1) * batch_size
        batches.append(ids[start:end])
    return batches # batch by indices


""" Fetch generative batch (repeat, reformulate) """
def fetch_data_ids_VAD(d1, d2, idx_batch, padlen=40):
    batch_size = len(idx_batch)
    q1, q2 = np.zeros([batch_size, padlen]), np.zeros([batch_size, padlen])
    q1_len, q2_len = np.zeros([batch_size]), np.zeros([batch_size])
    for i, idx in enumerate(idx_batch):
        q1[i], q1_len[i] = d1[idx] # padded
        q2[i], q2_len[i] = d2[idx]
    return q1, q1_len, q2, q2_len  # q, q'


""" Fetch discriminative batch (paraphrase identification) """
def fetch_data_ids_CLF(d1, d2, d3, idx_batch, padlen=40):
    batch_size = len(idx_batch)
    q1, q2 = np.zeros([batch_size, padlen]), np.zeros([batch_size, padlen])
    q1_len, q2_len = np.zeros([batch_size]), np.zeros([batch_size])
    labels = np.zeros([batch_size])
    for i, idx in enumerate(idx_batch):
        q1[i], q1_len[i] = d1[idx]
        q2[i], q2_len[i] = d2[idx]
        labels[i] = d3[idx]
    q_stack = np.concatenate((q1, q2),axis=0)
    q_len_stack = np.concatenate((q1_len, q2_len),axis=0)
    return q_stack, q_len_stack, labels # q1, q2, label




if __name__ == '__main__':

    # W2v
    #print(word2idx['linear'],word2idx['algebra'])
    #print(idx2word[2619],idx2word[3718])

    # Corpus ids
    #print(corpus_ids[0])
    #print([idx2word[i] for i in corpus_ids[0]])

    # VAD corpus
    d1, d2, N = load_VAD_corpus(corpus_ids, Xs_train_ids, Ys_train_ids, mode='VAD', padlen=40)
    batches = create_batches(N, batch_size=64)
    q1, q1_len, q2, q2_len = fetch_data_ids_VAD(d1, d2, batches[0], padlen=40)
    #print([idx2word[i] for i in q1[0]])
    #print([idx2word[i] for i in q2[0]])
    #print(q1_len[0], q2_len[0])

    # CLF corpus
    d1, d2, d3, N = load_CLF_corpus(Xs_dev_ids, Ys_dev_ids, Xa_dev_ids, Ya_dev_ids, padlen=40)
    batches = create_batches(N, batch_size=64)
    q_stack, q_len_stack, labels = fetch_data_ids_CLF(d1, d2, d3, batches[0], padlen=40)