import tensorflow as tf
import utils as utils


class Model(object):


    def __init__(self, embedding_weights, build_decoder=True, batch_size=256, padlen=40, w2vec_size=300, num_neurons=1000, hidden_size=1000, mlp_inner_dim=1000,
        lr_start=0.001, lr_decay_rate=0.96, lr_decay_step=5000, word_dropout_keep_prob=0.6, lb_decay_rate=0.002, lb_decay_step=2500, l1_regul=0.00001, l2_regul=0.0):

        # Data config
        self.batch_size = batch_size
        self.padlen = padlen # sentences length
        self.vocab_size = embedding_weights.shape[0] # number of tokens in dictionary
        self.word_embeddings = tf.Variable(embedding_weights, name="word_embeddings", dtype=tf.float32, trainable=True)

        # Network config
        self.w2v_size = w2vec_size # w2vec dimension
        self.num_neurons = num_neurons # encoder num_neurons/2 (LSTM cell)
        self.hidden_size = hidden_size # hidden dimension 
        self.mlp_inner_dim = mlp_inner_dim # MLP inner layer dimension
        self.initializer = tf.contrib.layers.xavier_initializer() # variables initializer

        # Training config
        self.global_step = tf.Variable(0, trainable=False, name='global_step') # global step
        self.lr_start = lr_start # initial learning rate
        self.lr_decay_rate= lr_decay_rate # learning rate decay rate
        self.lr_decay_step= lr_decay_step # learning rate decay step
        self.build_decoder = build_decoder

        # Regularization config
        self.lb_decay_step= lb_decay_step # spike step for a sigmoid kld annealing (0 to 1)
        self.lb_decay_rate= lb_decay_rate # spike rate (1/transition duration): kld_anneal=sigmoid( 0.002*(x-2500))
        self.word_dropout_keep_prob = word_dropout_keep_prob # decoder word keep prob (repeat, reformulate)
        self.regularizer = tf.contrib.layers.l1_l2_regularizer(scale_l1=l1_regul, scale_l2=l2_regul) # clf weights regularization

        # Placeholders
        self.q1 = tf.placeholder(tf.int32, shape=[None, self.padlen], name="question1")
        self.len1 = tf.placeholder(tf.int32, shape=[None], name="question1_length")
        self.q2 = tf.placeholder(tf.int32, shape=[None, self.padlen], name="question2")
        self.len2 = tf.placeholder(tf.int32, shape=[None], name="question2_length")

        self.encode()

        if self.build_decoder:
            with tf.variable_scope('vad'): self.decode()
            self.merged_VAD = tf.summary.merge_all()

        else:
            with tf.variable_scope('clf'): self.match()
            self.merged_SIAM = tf.summary.merge_all()


    def encode(self):

        encoded_output, encoded_state = utils.encode_seq(input_seq=self.q1, seq_len=self.len1, word_embeddings=self.word_embeddings, num_neurons=self.num_neurons) # [batch_size, 2*num_neurons]

        with tf.variable_scope("variational_inference"): # Variational inference
            mean = utils.linear(encoded_state, self.hidden_size, scope='mean') # [batch_size, n_hidden]
            logsigm = utils.linear(encoded_state, self.hidden_size, scope='logsigm') # [batch_size, n_hidden]
            self.mean, self.logsigm = mean, logsigm

            # Gaussian Multivariate kld(z,N(0,1)) = -0.5 * [ sum_d(logsigma) + d - sum_d(sigma) - mu_T*mu]
            klds = -0.5 * ( tf.reduce_sum(logsigm,1) + tf.cast(tf.shape(mean)[1], tf.float32) - tf.reduce_sum(tf.exp(logsigm),1) - tf.reduce_sum(tf.square(mean),1) )  # KLD(q(z|x), N(0,1))     tensor [batch_size]
            utils.variable_summaries('klds',klds) # posterior distribution close to prior N(0,1)      
            self.kld = tf.reduce_mean(klds, 0) # mean over batches: scalar

            h_ = tf.get_variable("GO",[1,self.hidden_size], initializer=self.initializer)
            h_ = tf.tile(h_, [self.batch_size,1]) # trainable tensor: decoder init_state[1]

            eps = tf.random_normal((self.batch_size, self.hidden_size), 0, 1)
            self.doc_vec = tf.multiply(tf.exp(logsigm), eps) + mean  # sample from latent intent space: decoder init_state[0]
            self.doc_vec = self.doc_vec, h_   # tuple state Z, h


    def decode(self):

        q2_embed = tf.nn.embedding_lookup(params=self.word_embeddings, ids=self.q2[:,:-1], name="q2_embedded") # Embed [batch_size, pad_len-1, w2v_dim]
        q2_bn = tf.transpose(q2_embed, [1,0,2]) # [pad_len-1, batch_size, w2v_dim]

        UNK = tf.tile(tf.get_variable("UNK",[1, 1, self.w2v_size], initializer=tf.constant_initializer(0)), [self.padlen-1, self.batch_size, 1])
        keep = tf.where(tf.concat([[0.], tf.random_uniform([ self.padlen-2])], axis=0) < self.word_dropout_keep_prob, tf.fill([self.padlen-1], True), tf.fill([self.padlen-1], False))
        masked_q2_bn = tf.where(keep, q2_bn, UNK) # [pad_len-1, batch_size, w2v_dim]
        masked_q2_bn = tf.unstack(masked_q2_bn, axis=0) # List [batch size, w2v_dim] of size pad_len-1

        logits = utils.decode_seq(decoder_inputs=masked_q2_bn, decoder_init_state=self.doc_vec, hidden_size=self.hidden_size, vocab_size=self.vocab_size)
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.q2[:,1:], logits=logits) # predict word i+1 given i, <i:    tensor [batch_size, pad_length-1]
        losses_mask = tf.sequence_mask(lengths=self.len2, maxlen=self.padlen-1, dtype=tf.float32)

        cross_entropies = tf.reduce_sum(losses * losses_mask, 1) # -sum_i log_p(wi | w<i, z) = -log_p(q'|z)  (sum over seq length):      tensor [batch_size]
        utils.variable_summaries('cross_entropies',cross_entropies)
        tf.summary.scalar('per_word_cross_entropy', tf.reduce_sum(losses * losses_mask) / tf.reduce_sum(losses_mask)) # per word cross entropy
        self.cross_entropy = tf.reduce_mean(cross_entropies, 0) # mean over batch: scalar

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):

            lr = tf.train.natural_exp_decay(learning_rate=self.lr_start, global_step=self.global_step, decay_steps=self.lr_decay_step, decay_rate=self.lr_decay_rate, staircase=False, name="learning_rate1") # learning rate
            tf.summary.scalar('lr',lr)
            opt = tf.train.AdamOptimizer(learning_rate=lr) # Optimizer
                
            lb = tf.sigmoid(self.lb_decay_rate*tf.cast(self.global_step-self.lb_decay_step,tf.float32)) # Kld cost annealing
            tf.summary.scalar('kld_anneal', lb)

            objective = self.cross_entropy + lb*self.kld # Objective
            tf.summary.scalar('objective',objective)
                
            gvs = opt.compute_gradients(objective)
            capped_gvs = [(tf.clip_by_norm(grad, 1.), var) for grad, var in gvs if grad is not None] # L2 clip
            self.optim1 = opt.apply_gradients(grads_and_vars=capped_gvs, global_step=self.global_step) # Minimize step


    def match(self):

        mu_batch_split = tf.split(self.mean, 2, axis=0) # split rpz(q) % batch 
        self.mu1, self.mu2 = mu_batch_split[0], mu_batch_split[1]

        logsigm_batch_split = tf.split(self.logsigm, 2, axis=0) 
        self.logs1, self.logs2 = logsigm_batch_split[0], logsigm_batch_split[1]

        ### Euclidean distance ###
        Sq_euclid = tf.square(self.mu1-self.mu2)
        
        ### Wasserstein2 distance (a.k.a root mean square bipartite matching distance) ###
        W2 = Sq_euclid + tf.square(tf.sqrt(tf.exp(self.logs1))-tf.sqrt(tf.exp(self.logs2)))
        self.W2_dist = tf.reduce_sum(W2, axis=1)
        utils.variable_summaries('W2', self.W2_dist)

        ### Mahalanobis distance ###
        Scale = 0.5*(tf.reciprocal(tf.exp(self.logs1))+tf.reciprocal(tf.exp(self.logs2)))
        Mah = tf.multiply(Scale, Sq_euclid)
        self.Mah_dist = tf.reduce_sum(Mah, axis=1)
        utils.variable_summaries('Mahalanobis', self.Mah_dist)

        features = tf.concat([W2, tf.multiply(self.mu1, self.mu2)], axis=-1) # [batch_size, 2*hidden_dim]
        logits_pred = utils.mlp(features,  mlp_hidden=[self.mlp_inner_dim, 2],  mlp_nonlinearity=tf.nn.relu, regularizer=self.regularizer, scope='mlp') # 2-layer MLP

        self.label = tf.placeholder(tf.int32, shape=[None], name="label") # true label (q and q' are duplicate or not)
        loss2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label, logits=logits_pred), 0) # loss
        tf.summary.scalar('cross_entropy', loss2)

        self.predict = tf.cast(tf.argmax(logits_pred, axis=1), tf.int32)
        self.accuracy = tf.reduce_mean(tf.cast(1-tf.abs(self.predict-self.label), tf.float32)) # accuracy
        tf.summary.scalar('accuracy',self.accuracy)

        # elastic regularization
        reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_term = tf.contrib.layers.apply_regularization(self.regularizer, reg_variables)
        objective2 = loss2 + reg_term
        tf.summary.scalar('objective', objective2)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):

            lr_ = tf.train.natural_exp_decay(learning_rate=self.lr_start, global_step=self.global_step, decay_steps=self.lr_decay_step, decay_rate=self.lr_decay_rate, staircase=False, name="learning_rate2") # learning rate
            tf.summary.scalar('lr_', lr_)
                
            opt_ = tf.train.AdamOptimizer(learning_rate=lr_) # Optimizer
            gvs_ = opt_.compute_gradients(objective2)
            capped_gvs_ = [(tf.clip_by_norm(grad, 1.), var) for grad, var in gvs_ if grad is not None] # L2 clip
            self.optim2 = opt_.apply_gradients(grads_and_vars=capped_gvs_, global_step=self.global_step) # Minimize step
