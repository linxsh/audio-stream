# coding: utf-8
import tensorflow as tf
import numpy as np

#对优化类进行一些自定义操作。
class MaxPropOptimizer(tf.train.Optimizer):
    def __init__(self, learning_rate=0.001, beta2=0.999, use_locking=False, name="MaxProp"):
        super(MaxPropOptimizer, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta2 = beta2
        self._lr_t = None
        self._beta2_t = None
    def _prepare(self):
        self._lr_t = tf.convert_to_tensor(self._lr, name="learning_rate")
        self._beta2_t = tf.convert_to_tensor(self._beta2, name="beta2")
    def _create_slots(self, var_list):
        for v in var_list:
            self._zeros_slot(v, "m", self._name)
    def _apply_dense(self, grad, var):
        lr_t = tf.cast(self._lr_t, var.dtype.base_dtype)
        beta2_t = tf.cast(self._beta2_t, var.dtype.base_dtype)
        if var.dtype.base_dtype == tf.float16:
            eps = 1e-7
        else:
            eps = 1e-8
        m = self.get_slot(var, "m")
        m_t = m.assign(tf.maximum(beta2_t * m + eps, tf.abs(grad)))
        g_t = grad / m_t
        var_update = tf.assign_sub(var, lr_t * g_t)
        return tf.group(*[var_update, m_t])
    def _apply_sparse(self, grad, var):
        return self._apply_dense(grad, var)

class AudioResNet(object):
    def __init__(self, batch_size = 16, input_size = 20, n_dim = 128, n_blocks = 3):
        self.batch_size = batch_size
        self.input_size = input_size
        self.n_dim = n_dim
        self.n_blocks = n_blocks

        tf.reset_default_graph()
        self.build_inputs()
        self.build_network()
        self.build_loss()
        self.build_optimizer()
        self.saver = tf.train.Saver(tf.global_variables())

    def build_inputs(self):
        with tf.name_scope('inputs'):
            self.inputs  = tf.placeholder(dtype = tf.float32, shape = [self.batch_size, None, self.input_size])#定义输入格式
            self.targets = tf.placeholder(dtype = tf.int32,   shape = [self.batch_size, None])#输出格式
            self.seq_len = tf.reduce_sum(tf.cast(tf.not_equal(tf.reduce_sum(self.inputs, reduction_indices = 2), 0.), tf.int32), reduction_indices = 1)

    def build_network(self): #定义神经网络
        conv1d_index = 0
        aconv1d_index = 0

        def conv1d_layer(input_tensor,size,dim,activation,scale,bias):
            with tf.variable_scope('conv1d_'+str(conv1d_index)):
                W= tf.get_variable('W', (size, input_tensor.get_shape().as_list()[-1], dim), dtype=tf.float32, initializer=tf.random_uniform_initializer(minval=-scale, maxval=scale))
                if bias:
                    b= tf.get_variable('b',[dim],dtype=tf.float32,initializer=tf.constant_initializer(0))
                out = tf.nn.conv1d(input_tensor,  W, stride=1, padding='SAME')#输出与输入同纬度
                if not bias:
                    beta = tf.get_variable('beta', dim, dtype=tf.float32, initializer=tf.constant_initializer(0))
                    gamma = tf.get_variable('gamma', dim, dtype=tf.float32, initializer=tf.constant_initializer(1))
                    mean_running = tf.get_variable('mean', dim, dtype=tf.float32, initializer=tf.constant_initializer(0))#均值
                    variance_running = tf.get_variable('variance', dim, dtype=tf.float32, initializer=tf.constant_initializer(1))#方差
                    mean, variance = tf.nn.moments(out, axes=range(len(out.get_shape()) - 1))
                    def update_running_stat(): #可以根据矩（均值和方差）来做normalize，见tf.nn.moments
                        decay =0.99 #mean_running、variance_running更新操作
                        update_op = [mean_running.assign(mean_running * decay + mean * (1 - decay)), variance_running.assign(variance_running * decay + variance * (1 - decay))]
                        with tf.control_dependencies(update_op):
                            return tf.identity(mean), tf.identity(variance) #返回mean,variance
                        m, v = tf.cond(tf.Variable(False, trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES]), update_running_stat, lambda: (mean_running, variance_running))
                        out = tf.nn.batch_normalization(out, m, v, beta, gamma, 1e-8)#batch_normalization
                if activation == 'tanh':
                    out = tf.nn.tanh(out)
                if activation == 'sigmoid':
                    out = tf.nn.sigmoid(out)

                conv1d_index += 1
                return out

        def aconv1d_layer(input_tensor, size, rate, activation, scale, bias):
            with tf.variable_scope('aconv1d_' + str(aconv1d_index)):
                shape = input_tensor.get_shape().as_list()#以list的形式返回tensor的shape
                W = tf.get_variable('W', (1, size, shape[-1], shape[-1]), dtype=tf.float32, initializer=tf.random_uniform_initializer(minval=-scale, maxval=scale))
                if bias:
                    b = tf.get_variable('b', [shape[-1]], dtype=tf.float32, initializer=tf.constant_initializer(0))
                out = tf.nn.atrous_conv2d(tf.expand_dims(input_tensor, dim=1), W, rate=rate, padding='SAME')
                #tf.expand_dims(input_tensor,dim=1)==>在第二维添加了一维，rate：采样率
                out = tf.squeeze(out, [1])#去掉第二维
                #同上
                if not bias:
                    beta = tf.get_variable('beta', shape[-1], dtype=tf.float32, initializer=tf.constant_initializer(0))
                    gamma = tf.get_variable('gamma', shape[-1], dtype=tf.float32, initializer=tf.constant_initializer(1))
                    mean_running = tf.get_variable('mean', shape[-1], dtype=tf.float32, initializer=tf.constant_initializer(0))
                    variance_running = tf.get_variable('variance', shape[-1], dtype=tf.float32, initializer=tf.constant_initializer(1))
                    mean, variance = tf.nn.moments(out, axes=range(len(out.get_shape()) - 1))

                    def update_running_stat():
                        decay = 0.99
                        update_op = [mean_running.assign(mean_running * decay + mean * (1 - decay)), variance_running.assign(variance_running * decay + variance * (1 - decay))]
                        with tf.control_dependencies(update_op):
                            return tf.identity(mean), tf.identity(variance)
                        m, v = tf.cond(tf.Variable(False, trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES]), update_running_stat, lambda: (mean_running, variance_running))
                        out = tf.nn.batch_normalization(out, m, v, beta, gamma, 1e-8)
                if activation == 'tanh':
                    out = tf.nn.tanh(out)
                if activation == 'sigmoid':
                    out = tf.nn.sigmoid(out)

                aconv1d_index += 1
                return out

        def residual_block(input_sensor, size, rate): # skip connections
            conv_filter = aconv1d_layer(input_sensor, size=size, rate=rate, activation='tanh', scale=0.03, bias=False)
            conv_gate = aconv1d_layer(input_sensor, size=size, rate=rate, activation='sigmoid', scale=0.03, bias=False)
            out = conv_filter * conv_gate
            out = conv1d_layer(out, size = 1, dim = n_dim, activation = 'tanh', scale = 0.08, bias = False)
            return out + input_sensor, out

        out = conv1d_layer(input_tensor = self.inputs, size = 1, dim = self.n_dim, activation = 'tanh', scale = 0.14, bias = False) #卷积层输出
        skip = 0
        for _ in range(n_blocks):
            for r in [1, 2, 4, 8, 16]:
                out, s = residual_block(out, size=7, rate=r)#根据采样频率发生变化
                skip += s

        #两层卷积
        self.logit = conv1d_layer(skip,  size = 1, dim = skip.get_shape().as_list()[-1], activation = 'tanh', scale = 0.08, bias = False)
        self.logit = conv1d_layer(self.logit, size = 1, dim = words_size, activation = None, scale = 0.04, bias = True)

    def build_loss(self): # CTC loss
        indices = tf.where(tf.not_equal(tf.cast(self.targets, tf.float32), 0.))
        target = tf.SparseTensor(indices = indices, values = tf.gather_nd(self.targets, indices) - 1, shape = tf.cast(tf.shape(self.targets), tf.int64))
        self.loss = tf.nn.ctc_loss(self.logit, target, self.seq_len, time_major = False)

    def build_optimizer(self):
        self.lr = tf.Variable(0.001, dtype = tf.float32, trainable = False)
        optimizer = MaxPropOptimizer(learning_rate = self.lr, beta2 = 0.99)
        var_list = [t for t in tf.trainable_variables()]
        gradient = optimizer.compute_gradients(self.loss, var_list = var_list)
        self.optimizer_op = optimizer.apply_gradients(gradient)

    def train(self, folder, text, batch_size):
        print "开始训练:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        audio_batch = AudioBatch(folder, text)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())#初始化变量
            for epoch in range(16):
                print "第%d次循环迭代: %s" % (epoch, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                sess.run(tf.assign(self.lr, 0.001 * (0.97 ** epoch)))
                audio_batch.update_batches()
                for batch in range(audio_batch.get_n_batch()):
                    batches_wavs, batches_labels = audio_batch.get_batches(batch_size)
                    train_loss, _ = sess.run([self.loss, self.optimizer_op], feed_dict = {self.inputs: batches_wavs, self.targets: batches_labels})
                    print epoch, batch, train_loss

                if epoch % 5 == 0:
                    self.saver.save(sess, './speech.module', global_step = epoch)
                    print "第%d次模型保存结果: %s" % (epoch, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        print "结束训练时刻:",time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    def sample(self, mfcc):
        mfcc = np.transpose(np.expand_dims(mfcc, axis=0), [0, 2, 1])

        with tf.Session() as sess:
            self.saver.restore(sess, tf.train.latest_checkpoint('.'))

            decoded = tf.transpose(self.logit, perm=[1, 0, 2])
            decoded, _ = tf.nn.ctc_beam_search_decoder(decoded, self.seq_len, merge_repeated = False)
            predict = tf.sparse_to_dense(decoded[0].indices, decoded[0].shape, decoded[0].values) + 1
            output = sess.run(decoded, feed_dict={self.inputs: mfcc})
            print(output)
