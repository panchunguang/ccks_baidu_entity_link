import codecs
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from keras.models import *
from keras_layers import CLSOut
import pandas as pd
import os
bert_path='bert_model/'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def get_token_dict():
    dict_path = bert_path + 'vocab.txt'
    token_dict = {}
    with codecs.open(dict_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)
    return token_dict

def get_model(seq_len):
    '''
    bert模型，模型输出为CLS位置的向量
    :param seq_len:
    :return:
    '''
    config_path=bert_path + 'bert_config.json'
    checkpoint_path=bert_path + 'bert_model.ckpt'
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path,seq_len=seq_len)
    x=CLSOut()(bert_model.output)
    model = Model(bert_model.inputs, x)
    return model
def extract(max_len=512):
    '''
    :param max_len: 文本最大长度
    :return: 字典形式，key: kb_id  value: kb_id对应描述文本形成的向量
    '''
    model = get_model(max_len)
    token_dict = get_token_dict()
    tokenizer = Tokenizer(token_dict)
    id_text=pd.read_pickle('data/id_text.pkl')
    id_embedding={}
    for id in id_text:
        if int(id)%10000==0:
            print(id)
        text=id_text[id]
        indices, segments = tokenizer.encode(first=text,max_len=512)
        predicts = model.predict([[indices], [segments]], verbose=2)
        id_embedding[id]=predicts[0]
    pd.to_pickle(id_embedding,'data/id_embedding.pkl')

if __name__ == '__main__':
    extract()
    pass
