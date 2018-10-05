import os
import tensorflow as tf
from tqdm import tqdm

import sys
from data import *
from nnet import Model


        
def build_graph(seed=123, build_decoder=True, batch_size=256, padlen=40):
    print("\n Building graph...")
    np.random.seed(seed) 
    tf.set_random_seed(seed) # reproducibility
    tf.reset_default_graph()
    model = Model(embedding_weights=weights, build_decoder=build_decoder, batch_size=batch_size, padlen=padlen) # Build tensorflow graph from config

    variables_to_save1 = [v for v in tf.global_variables() if 'Adam' not in v.name and 'global_step' not in v.name and 'vad' not in v.name] # Saver to save & restore all the variables.
    variables_to_save2 = [v for v in tf.global_variables() if 'Adam' not in v.name and 'global_step' not in v.name and 'clf' not in v.name]
    saver1 = tf.train.Saver(var_list=variables_to_save1, keep_checkpoint_every_n_hours=1.0) # CLF saver
    saver2 = tf.train.Saver(var_list=variables_to_save2, keep_checkpoint_every_n_hours=1.0) # VAD saver
    return model, saver1, saver2


def train_VAD(model, saver2, d1, d2, N, pretrain='VAD', nepochs=5, batch_size=256, padlen=40):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer()) # run init op

        writer_VAD = tf.summary.FileWriter('{}/_deepNLU/summary/{}'.format(dir_, pretrain), sess.graph) # Summary writer
        for epoch in range(nepochs):
            batches = create_batches(N, batch_size=batch_size) # Random batch indices            
            for i, idx_batch in tqdm(enumerate(batches)): # train VAD
                q1, q1_len, q2, q2_len = fetch_data_ids_VAD(d1, d2, idx_batch, padlen=padlen)  # fetch data (paraphrases)
                feed = {model.q1: q1, model.len1: q1_len, model.q2: q2, model.len2: q2_len}
                _, summary_VAD, global_step_ = sess.run([model.optim1, model.merged_VAD, model.global_step], feed_dict=feed) # Forward pass & VAD step
                if i%100 == 0: writer_VAD.add_summary(summary_VAD, global_step_)

        print('{} Training COMPLETED !'.format(pretrain))
        if not os.path.exists('{}/_deepNLU/save/{}'.format(dir_,pretrain)):
            os.makedirs('{}/_deepNLU/save/{}'.format(dir_,pretrain))
        saver2.save(sess, '{}/_deepNLU/save/{}/actor.ckpt'.format(dir_,pretrain)) # Save the variables to disk


def train_CLF(model, saver1, saver2, d1, d2, d3, N, pretrain='VAD', nepochs=3, batch_size=256, padlen=40):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer()) # run init op
        saver2.restore(sess, '{}/_deepNLU/save/{}/actor.ckpt'.format(dir_,pretrain)) # Restore variables from disk.

        writer_SIAM = tf.summary.FileWriter('{}/_deepNLU/summary/{}SIAM'.format(dir_,pretrain), sess.graph) # Summary writer
        if not os.path.exists('{}/_deepNLU/save/{}SIAM{}'.format(dir_,pretrain,0)):
            os.makedirs('{}/_deepNLU/save/{}SIAM{}'.format(dir_,pretrain,0))
        saver1.save(sess, '{}/_deepNLU/save/{}SIAM{}/actor.ckpt'.format(dir_,pretrain,0)) # Save the variables to disk

        for epoch in range(nepochs): # match op nb epoch
            batches = create_batches(N, batch_size=batch_size) # Random batch indices
            for i, idx_batch in tqdm(enumerate(batches)): # train CLF 
                q_stack, q_len_stack, labels = fetch_data_ids_CLF(d1, d2, d3, idx_batch, padlen=padlen)  # fetch data (question pairs [stack] and labels)
                feed = {model.q1: q_stack, model.len1: q_len_stack, model.label: labels}
                _, summary_SIAM, global_step_ = sess.run([model.optim2, model.merged_SIAM, model.global_step], feed_dict=feed) # Forward pass & CLF step
                if i%100 == 0:
                    writer_SIAM.add_summary(summary_SIAM, global_step_)

        print('\n {}SIAM Training COMPLETED !'.format(pretrain))
        if not os.path.exists('{}/_deepNLU/save/{}SIAM{}'.format(dir_,pretrain,epoch+1)):
            os.makedirs('{}/_deepNLU/save/{}SIAM{}'.format(dir_,pretrain,epoch+1))
        saver1.save(sess, '{}/_deepNLU/save/{}SIAM{}/actor.ckpt'.format(dir_,pretrain,epoch+1)) # Save the variables to disk



if __name__== '__main__':

    # Config
    pretrain_ =  sys.argv[1] # run train.py VAD (or VAE)
    padlen_ = 40
    batch_size_ = 256

    # Generative pretraining (VAD session)
    model, _, saver2 = build_graph(seed=123, build_decoder=True, batch_size=batch_size_, padlen=padlen_)
    d1, d2, N = load_VAD_corpus(corpus_ids, Xs_train_ids, Ys_train_ids, mode=pretrain_, padlen=padlen_)
    train_VAD(model, saver2, d1, d2, N, pretrain=pretrain_, nepochs=5, batch_size=batch_size_, padlen=padlen_)   # Train VAD
    #del corpus_ids

    # Discriminative training (CLF session)
    model, saver1, saver2 = build_graph(seed=36, build_decoder=False, batch_size=batch_size_, padlen=padlen_)
    d1, d2, d3, N = load_CLF_corpus(Xs_train_ids, Ys_train_ids, Xa_train_ids, Ya_train_ids, padlen=padlen_)
    train_CLF(model, saver1, saver2, d1, d2, d3, N, pretrain=pretrain_, nepochs=3, batch_size=batch_size_, padlen=padlen_)  # Train CLF