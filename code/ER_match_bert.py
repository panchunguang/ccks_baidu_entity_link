import pandas as pd
from sklearn.model_selection import KFold
from keras.layers import *
from keras.models import *
from keras_layers import *
from keras.optimizers import *
from keras.losses import *
from keras.callbacks import *
from keras.metrics import *
from keras_bert import load_trained_model_from_checkpoint
from keras_bert.layers import MaskedGlobalMaxPool1D

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import logging
import argparse

logging.basicConfig(level=logging.DEBUG,filename='model/match_bert.log',filemode='a',
                    format='%(asctime)s - %(levelname)s: %(message)s')
parser=argparse.ArgumentParser()
parser.add_argument('--learning_rate',type=float,default=0.001)

parser.add_argument('--spatial_dropout',type=float,default=0.3)
parser.add_argument('--dropout',type=float,default=0.2)
parser.add_argument('--batch_size',type=int,default=12)
parser.add_argument('--num_epochs',type=int,default=3)
parser.add_argument('--mode',type=int,default=1)

arg=parser.parse_args()

def metrics_f1(y_true, y_pred):
    '''
    tensorflow版 序列二分类f1
    :param y_true:
    :param y_pred:
    :return:
    '''
    y_pred = tf.where(y_pred < 0.5, x=tf.zeros_like(y_pred), y=tf.ones_like(y_pred))
    equal_num = tf.count_nonzero(tf.multiply(y_true, y_pred))
    true_sum=tf.count_nonzero(y_true)
    pred_sum=tf.count_nonzero(y_pred)
    precision = equal_num / pred_sum
    recall = equal_num / true_sum
    f1 = (2 * precision * recall) / (precision + recall)
    return f1
def match_f1(y_true, y_pred):
    '''
        numpy版 序列二分类f1
        :param y_true:
        :param y_pred:
        :return:
        '''
    threshold_valud = 0.5
    y_true = np.reshape(y_true, (-1))
    y_pred = [1 if p >= threshold_valud else 0 for p in np.reshape(y_pred, (-1))]
    equal_num = np.sum([1 for t, p in zip(y_true, y_pred) if t == p and t == 1 and p == 1])
    true_sum = np.sum(y_true)
    pred_sum = np.sum(y_pred)
    precision = equal_num / pred_sum
    recall = equal_num / true_sum
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1
class Evaluate(Callback):
    '''
    keras 评价类 继承与Callback类
    '''
    def __init__(self, validation_data, filepath, factor=0.5, patience=1, min_lr=1e-4, stop_patience=2):
        val_data, val_label = validation_data
        self.F1 = []
        self.val = val_data
        self.label = val_label
        self.best = 0.
        self.f1_raise = 1
        self.factor = factor
        self.min_lr = min_lr
        self.patience = patience
        self.wait = 0
        self.stop_patience = stop_patience
        self.filepath = filepath
        super(Evaluate, self).__init__()
    def on_epoch_end(self, epoch, logs=None):
        print('Evaluate:')
        precision, recall, f1, = self.evaluate()
        if f1 > self.best:
            self.best = f1
            self.model.save_weights(self.filepath)

        print(' precision: %.6f, recall: %.6f,f1: %.6f, best f1: %.6f\n' % (
            float(precision), float(recall), float(f1), float(self.best)))

        logging.debug(str(precision) + ' ' + str(recall) + ' ' + str(f1))

    def evaluate(self):
        pred = self.model.predict(self.val)
        return match_f1(self.label, pred)

    def stop_train(self, F1, best_f1, stop_patience):
        stop = True
        for f in F1[-stop_patience:]:
            if f >= best_f1:
                stop = False
        if stop == True:
            print('EarlyStopping!!!')
            self.model.stop_training = True


def seq_and_vec(x):
    """seq是[None, seq_len, s_size]的格式，
    vec是[None, v_size]的格式，将vec重复seq_len次，拼到seq上，
    得到[None, seq_len, s_size+v_size]的向量。
    """
    seq, vec = x
    vec = K.expand_dims(vec, 1)
    vec = K.zeros_like(seq[:, :, :1]) + vec
    return K.concatenate([seq, vec], 2)

def seq_maxpool(x):
    """seq是[None, seq_len, s_size]的格式，
    mask是[None, seq_len, 1]的格式，先除去mask部分，
    然后再做maxpooling。
    """
    seq, mask = x
    seq -= (1 - mask) * 1e10
    return K.max(seq, 1)

embedding_matrix_entity = np.load('data/ER_entity_embedding.npy')

def link_model():

    input_en= Input(shape=(13,), name='kb_en')
    input_begin = Input(shape=(13,), name='begin')
    input_end = Input(shape=(13,), name='end')
    bert_path = 'bert_model/'
    config_path = bert_path + 'bert_config.json'
    checkpoint_path = bert_path + 'bert_model.ckpt'
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, trainable=True, seq_len=52)
    entity_embedding = Embedding(input_dim=312452, output_dim=768, weights=[embedding_matrix_entity],trainable=True,name='entity_embedding')
    men_sen = bert_model.output
    mask_sen = Lambda(lambda x: K.cast(K.greater(x, 0), 'float32'))(bert_model.input[0])
    men_sen = Lambda(lambda x: x[0] * K.expand_dims(x[1], axis=-1))([men_sen, mask_sen])
    men_sen = SpatialDropout1D(0.15)(men_sen)
    [forward, backward] = Bidirectional(CuDNNGRU(128, return_sequences=True), merge_mode=None)(men_sen,mask=None)
    gru=concatenate([forward,backward],axis=-1)
    max_x=MaskedGlobalMaxPool1D()(gru)
    x=StateMix()([input_begin,input_end,forward,backward])
    t_dim = K.int_shape(x)[-1]
    x = Lambda(seq_and_vec, output_shape=(13, t_dim * 2))([x, max_x])
    mask = Lambda(lambda x: K.cast(K.greater(x, -1), 'float32'))(input_begin)
    kb_en = entity_embedding(input_en)
    x=concatenate([kb_en,x],axis=-1)
    x = Lambda(lambda x: x[0] * K.expand_dims(x[1], axis=-1))([x, mask])
    x = Dropout(0.1)(x)
    x = Conv1D(128, 1, activation='relu', padding='same')(x)
    # x = Dense(units=128, activation='relu')(x)
    x = TimeDistributed(Dropout(0.1))(x)
    x = Dense(units=1, activation='sigmoid')(x)
    model = Model(bert_model.inputs+[input_en,input_begin,input_end], x)
    model.compile(optimizer=adam(), loss=binary_crossentropy, metrics=[metrics_f1])
    return model

def get_input(input_file):
    data=pd.read_pickle(input_file)
    inputs=[data['ids'],data['seg'],data['entity_id'],data['begin'],data['end'],np.expand_dims(data['labels'],axis=-1)]
    return inputs

def step_decay(epoch):
    '''
    用来控制学习率
    :param epoch:
    :return:
    '''
    if epoch<2:
       lr=1e-5
    else:
       lr = 1e-6
    return lr

def train():
    '''
    模型训练，将训练集分为9份，9折交叉验证，分别按照loss和f1保存模型。
    :return:
    '''
    dataset = get_input('data/ER_input_train_match.pkl')
    train=dataset
    kfold = KFold(n_splits=9, shuffle=False)
    for i, (tra_index, val_index) in enumerate(kfold.split(train[0])):
        K.clear_session()
        input_train = [tra[tra_index] for tra in train]
        input_val = [tra[val_index] for tra in train]
        filepath_loss = "model/ER_match_bert_loss.h5_"+str(i)
        filepath_f1 = "model/ER_match_bert_f1.h5_"+str(i)
        evaluate = Evaluate((input_val[:-1], input_val[-1]), filepath_f1)
        checkpoint = ModelCheckpoint(filepath_loss, monitor='val_loss', verbose=2, save_best_only=True,save_weights_only=True, mode='min')
        earlystopping = EarlyStopping(monitor='val_loss', min_delta=0.000001, patience=2, verbose=3, mode='auto')
        lrate = LearningRateScheduler(step_decay, verbose=2)
        callbacks = [checkpoint,evaluate,lrate,earlystopping]
        model = link_model()
        print(model.summary())
        logging.debug(str(i))
        logging.debug(str(i)*30)
        model = model.fit(input_train[:-1], input_train[-1], batch_size=arg.batch_size, epochs=arg.num_epochs,
                          validation_data=(input_val[:-1], input_val[-1]), verbose=1, callbacks=callbacks)
        logging.debug(arg.__str__())
        logging.debug(model.history)
        logging.debug(np.min(model.history['val_loss']))

def predict(input_file,out_file):
    '''
    模型预测，加载按照F1保存的模型，9折交叉验证求平均
    :param input_file:
    :param out_file:
    :return:
    '''
    print('test predict')
    all_test=[]
    for i in range(9):
        # if not i==8:
        #     continue
        print(i)
        K.clear_session()
        model = link_model()
        filepath_loss = "model/ER_match_bert_f1.h5_" + str(i)
        test = get_input(input_file)
        model.load_weights(filepath_loss)
        y_pred_test = model.predict(test[:-1], batch_size=6, verbose=2)
        all_test.append(y_pred_test)

    all_test=np.mean(all_test,axis=0)
    np.save(out_file, all_test)


if __name__ == '__main__':
    train()
    # predict('data/input_dev_match_bert.pkl','data/pred_dev_match.npy')

    predict('data/input_eval_match_bert.pkl', 'data/pred_eval_match.npy')
    pass