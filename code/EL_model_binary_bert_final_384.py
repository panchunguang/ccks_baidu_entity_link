
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
from evaluate import *
import json
import logging
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

logging.basicConfig(level=logging.DEBUG,filename='model/ED_binary_model_bert.log',filemode='a',
                    format='%(asctime)s - %(levelname)s: %(message)s')
parser=argparse.ArgumentParser()
parser.add_argument('--learning_rate',type=float,default=0.001)

parser.add_argument('--spatial_dropout',type=float,default=0.3)
parser.add_argument('--dropout',type=float,default=0.15)
parser.add_argument('--gru',type=int,default=128)
parser.add_argument('--num_capsule',type=int,default=12)
parser.add_argument('--dim_capsule',type=int,default=10)
parser.add_argument('--routings',type=int,default=4)
parser.add_argument('--word_num',type=int,default=200000)

parser.add_argument('--type_dim',type=int,default=8)
parser.add_argument('--dense',type=int,default=100)
parser.add_argument('--batch_size',type=int,default=8)
parser.add_argument('--num_epochs',type=int,default=1)
parser.add_argument('--mode',type=int,default=1)

arg=parser.parse_args()
def metrics_f1(y_true, y_pred):
    y_pred = tf.where(y_pred < 0.5, x=tf.zeros_like(y_pred), y=tf.ones_like(y_pred))
    equal_num = tf.count_nonzero(tf.multiply(y_true, y_pred))
    true_sum=tf.count_nonzero(y_true)
    pred_sum=tf.count_nonzero(y_pred)
    precision = equal_num / pred_sum
    recall = equal_num / true_sum
    f1 = (2 * precision * recall) / (precision + recall)
    return f1
def link_f1(y_true, y_pred):
    threshold_valud = 0.5
    y_true = np.reshape(y_true, (-1))
    y_pred = [1 if p > threshold_valud else 0 for p in np.reshape(y_pred, (-1))]
    equal_num = np.sum([1 for t, p in zip(y_true, y_pred) if t == p and t == 1 and p == 1])
    true_sum = np.sum(y_true)
    pred_sum = np.sum(y_pred)
    precision = equal_num / pred_sum
    recall = equal_num / true_sum
    f1 = (2 * precision * recall) / (precision + recall)

    print('equal_num:', equal_num)
    print('true_sum:', true_sum)
    print('pred_sum:', pred_sum)
    print('precision:', precision)
    print('recall:', recall)
    print('f1:', f1)

    return precision, recall, f1

class Evaluate(Callback):
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
        return link_f1(self.label, pred)

    def stop_train(self, F1, best_f1, stop_patience):
        stop = True
        for f in F1[-stop_patience:]:
            if f >= best_f1:
                stop = False
        if stop == True:
            print('EarlyStopping!!!')
            self.model.stop_training = True

def mention_link_result(model,input_file,out_file):
    out_file=open(out_file,'w')
    with open(input_file, 'r') as f:
        for i,line in enumerate(f):
            if i%1000 == 0:
                print(i)
            temDict = json.loads(line)
            mention_data = temDict['mention_data']
            tem_men_data=[]
            for men in mention_data:
                if len(men['link_id'])>0:
                    pred=model.predict(get_input(men['link_data'],mode='test'))
                    pred=list(np.squeeze(pred, axis=-1))
                    arg_max=int(np.argmax(pred))
                    if pred[arg_max]>0.06:
                        men['kb_id'] = men['link_id'][arg_max]
                    else:
                        men['kb_id'] = 'NIL'
                else:
                    men['kb_id']='NIL'
                men.pop('link_id')
                men.pop('link_data')
            out_file.write(json.dumps(temDict, ensure_ascii=False))
            out_file.write('\n')


def seq_and_vec(x):
    seq, vec = x
    vec = K.expand_dims(vec, 1)
    vec = K.zeros_like(seq[:, :, :1]) + vec
    return K.concatenate([seq, vec], 2)

def seq_maxpool(x):
    seq, mask = x
    seq -= (1 - mask) * 1e10
    return K.max(seq, 1)

def link_model():

    input_begin= Input(shape=(1,),name='men_sen')
    input_end= Input(shape=(1,), name='men_pos')
    bert_model= 'bert_model/'
    config_path = bert_model+'bert_config.json'
    checkpoint_path = bert_model+'bert_model.ckpt'
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, trainable=True, seq_len=384)
    x = bert_model.output

    cls = CLSOut()(x)
    entity_em = StateMixOne()([input_begin,input_end,x,x])

    x=concatenate([cls,entity_em],axis=-1)
    x = Dense(units=128, activation='relu')(x)
    x = Dropout(arg.dropout)(x)
    x = Dense(units=1, activation='sigmoid')(x)
    model = Model(bert_model.inputs + [input_begin, input_end], x)
    model.compile(optimizer=adam(), loss=binary_crossentropy, metrics=[metrics_f1])
    return model


def get_input(input_file,mode='train'):
    if mode=='test':
        data=input_file
        return [data['ids'],data['seg'],data['begin'], data['end']]
    data=pd.read_pickle(input_file)
    inputs=[data['ids'],data['seg'],data['begin'], data['end'],np.expand_dims(data['labels'],axis=-1)]
    return inputs

def step_decay(epoch,initial_lrate=0.00001):
   if epoch<3:
       lr=1e-7
   else:
       lr = 1e-7
   return lr
def train():
    dataset = get_input('data/train_input_bert_final_384.pkl')
    train=dataset

    kfold = KFold(n_splits=7, shuffle=False)
    for i, (tra_index, val_index) in enumerate(kfold.split(train[0])):
        if not i==5:
            continue
        K.clear_session()
        input_train = [tra[tra_index] for tra in train]
        input_val = [tra[val_index] for tra in train]
        filepath_loss = "model/ED_binary_model_bert_loss_384.h5_"+str(i)
        filepath_f1 = "model/ED_binary_model_bert_f1_384.h5_" + str(i)

        evaluate = Evaluate((input_val[:-1], input_val[-1]), filepath_f1)
        checkpoint = ModelCheckpoint(filepath_loss, monitor='val_loss', verbose=2, save_weights_only=True,save_best_only=True, mode='min')
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.0001, verbose=2)
        earlystopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=2, verbose=2, mode='auto')
        lrate = LearningRateScheduler(step_decay, verbose=2)
        callbacks = [checkpoint,evaluate,lrate,earlystopping]
        model = link_model()
        print(model.summary())
        print(i)
        # #
        logging.debug(str(i)+'################')
        model = model.fit(input_train[:-1], input_train[-1], batch_size=arg.batch_size, epochs=arg.num_epochs,
                          validation_data=(input_val[:-1], input_val[-1]), verbose=1, callbacks=callbacks)

        logging.debug(arg.__str__())
        logging.debug(model.history)
        logging.debug(np.min(model.history['val_loss']))
        logging.debug(np.max(model.history['val_metrics_f1']))


def predict_loss(input_file,out_file,model_index=0):
    out_file=open(out_file,'w')
    result_list=[]
    with open(input_file, 'r') as f:
        for line in f:
            temDict = json.loads(line)
            re_dict = {'text_id': temDict['text_id'], 'text': temDict['text']}
            re_dict['mention_data'] = []
            mention_data = temDict['mention_data']
            for men in mention_data:
                men.pop('link_data')
                men['link_pred']=[0]*len(men['link_id'])
                re_dict['mention_data'].append(men)
            result_list.append(re_dict)
    for i in range(7):
        if not i==model_index:
            continue
        print(i)
        filepath_loss = "model/ED_binary_model_bert_loss_384.h5_" + str(i)
        model = link_model()
        model.load_weights(filepath_loss)
        with open(input_file,'r') as f:
            for j,line in enumerate(f):
                if j%1000==0:
                    print(j)
                temDict = json.loads(line)
                mention_data = temDict['mention_data']
                re_men_data = result_list[j]['mention_data']
                for men,re_men in zip(mention_data,re_men_data):
                    if len(men['link_id']) > 0:
                        pred=model.predict(get_input(men['link_data'],mode='test'))
                        re_men['link_pred']=list(np.sum([re_men['link_pred'],list(np.squeeze(pred,axis=-1))],axis=0))
                    else:
                        re_men['link_pred']=[]
    for r in result_list:
        out_file.write(json.dumps(r, ensure_ascii=False))
        out_file.write('\n')
def predict_f1(input_file,out_file,model_index=0):
    out_file=open(out_file,'w')
    result_list=[]
    with open(input_file, 'r') as f:
        for line in f:
            temDict = json.loads(line)
            re_dict = {'text_id': temDict['text_id'], 'text': temDict['text']}
            re_dict['mention_data'] = []
            mention_data = temDict['mention_data']
            for men in mention_data:
                men.pop('link_data')
                men['link_pred']=[0]*len(men['link_id'])
                re_dict['mention_data'].append(men)
            result_list.append(re_dict)
    for i in range(5):
        if not i==model_index:
            continue
        print(i)
        filepath_loss = "model/ED_binary_model_bert_f1_384.h5_" + str(i)
        model = link_model()
        model.load_weights(filepath_loss)
        with open(input_file,'r') as f:
            for j,line in enumerate(f):
                if j%1000==0:
                    print(j)
                temDict = json.loads(line)
                mention_data = temDict['mention_data']
                re_men_data = result_list[j]['mention_data']
                for men,re_men in zip(mention_data,re_men_data):
                    if len(men['link_id']) > 0:
                        pred=model.predict(get_input(men['link_data'],mode='test'))
                        re_men['link_pred']=list(np.sum([re_men['link_pred'],list(np.squeeze(pred,axis=-1))],axis=0))
                    else:
                        re_men['link_pred']=[]

    for r in result_list:
        out_file.write(json.dumps(r, ensure_ascii=False))
        out_file.write('\n')



if __name__ == '__main__':
    train()#当时只跑了三个

    predict_loss('data/eval_link_binary_bert_384.json', 'result/eval_bert_loss_384_final_0.json', model_index=0)
    predict_loss('data/eval_link_binary_bert_384.json', 'result/eval_bert_loss_384_final_1.json', model_index=1)
    predict_loss('data/eval_link_binary_bert_384.json', 'result/eval_bert_loss_384_final_6.json', model_index=5)
    predict_f1('data/eval_link_binary_bert_384.json', 'result/eval_bert_f1_384_final_0.json', model_index=0)
    predict_f1('data/eval_link_binary_bert_384.json', 'result/eval_bert_f1_384_final_1.json', model_index=1)
    predict_f1('data/eval_link_binary_bert_384.json', 'result/eval_bert_f1_384_final_6.json', model_index=5)

    pass
