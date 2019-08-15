import pandas as pd
import  numpy as np
import json

def get_bio_dict():
    bio_dict={}
    bio_dict['O']=0
    bio_dict['B'] = 1
    bio_dict['I'] = 2
    bio_dict['TAG'] = 3
    return bio_dict

def get_index_bio_dict():
    bio_dict=get_bio_dict()
    index_bio={}
    for k in bio_dict:
        index_bio[bio_dict[k]]=k
    index_bio[3]='O'
    index_bio[4] = 'O'
    return index_bio
def get_entity(lable,index_bio):
    lable = [index_bio[i] for i in lable]
    entity=[]
    start = -1
    end = -1
    for i in range(len(lable)):
        if lable[i][0] == 'B':
            start = i
            end = i

            if i==len(lable)-1 and not start==-1:
                entity.append((start,end))
                start = -1
            elif i != len(lable)-1 and lable[i+1][0] == 'O' and not start==-1:
                entity.append((start, end))
                start = -1
            elif i != len(lable)-1 and lable[i+1][0] == 'B' and not start==-1:
                entity.append((start, end))
                start = -1
        elif lable[i][0] == 'I':
            end = i
            if i==len(lable)-1 and not start==-1:
                entity.append((start,end))
                start = -1
            elif i != len(lable)-1 and lable[i+1][0] == 'O' and not start==-1:
                entity.append((start, end))
                start = -1
            elif i != len(lable)-1 and lable[i+1][0] == 'B' and not start==-1:
                entity.append((start, end))
                start = -1
    return entity

def get_equal_num(y_true,y_pred,index_bio):
    entity_true=set(get_entity(y_true,index_bio))
    entity_pred = set(get_entity(y_pred, index_bio))
    return len(entity_true),len(entity_pred),len(entity_true&entity_pred)


def ner_eval(y_true,y_pred):
    index_bio = get_index_bio_dict()
    y_true = np.squeeze(y_true, 2)
    y_pred= np.argmax(y_pred, -1)
    equal_num=1e-10
    true_num=1e-10
    pred_num =1e-10
    for y_t,y_p in zip(y_true,y_pred):
        tn, pn, en=get_equal_num(y_t,y_p,index_bio)
        true_num+=tn
        pred_num+=pn
        equal_num+=en
    precision = equal_num / pred_num
    recall = equal_num / true_num
    f1 = (2 * precision * recall) / (precision + recall)
    print('equal_num:', equal_num)
    print('true_num:', true_num)
    print('pred_num:', pred_num)
    print('precision:', precision)
    print('recall:', recall)
    print('f1:', f1)
    return precision,recall,f1



def mention_evaluate(y_ture,y_pred=None,is_NIL=False):
    if y_pred==None:
        y_pred=y_ture
    true_num = 1e-10
    pred_num = 1e-10
    equal_num = 1e-10

    y_ture_file = open(y_ture,'r')
    y_pred_file = open(y_pred, 'r')
    for line_true,line_pred in zip(y_ture_file,y_pred_file):
        line_true = json.loads(line_true)
        line_pred = json.loads(line_pred)
        mention_data_true = line_true['mention_data']
        mention_data_pred = line_pred['mention_data']
        true_set = set()
        pred_set = set()
        for mention in mention_data_true:
            if is_NIL:
                if mention['kb_id'] != 'NIL':
                    true_set.add((mention['mention'],mention['offset']))
            else:
                true_set.add((mention['mention'], mention['offset']))
        for mention in mention_data_pred:
            pred_set.add((mention['mention'],mention['offset']))
        true_num += len(true_set)
        pred_num += len(pred_set)
        equal_num += len(true_set&pred_set)
    precision = equal_num / pred_num
    recall = equal_num / true_num
    f1 = (2 * precision * recall) / (precision + recall)
    print('equal_num:',equal_num)
    print('true_num:', true_num)
    print('pred_num:', pred_num)
    print()
    return precision, recall, f1


def link_eval(y_ture,y_pred):
    true_num = 1e-10
    pred_num = 1e-10
    equal_num = 1e-10

    y_ture_file = open(y_ture,'r')
    y_pred_file = open(y_pred, 'r')
    for line_true,line_pred in zip(y_ture_file,y_pred_file):
        line_true = json.loads(line_true)
        line_pred = json.loads(line_pred)

        mention_data_true = line_true['mention_data']
        mention_data_pred = line_pred['mention_data']
        true_set = set()
        pred_set = set()

        for mention in mention_data_true:
            if mention['kb_id'] != 'NIL':
                true_set.add((mention['kb_id'],mention['mention'],mention['offset']))
        for mention in mention_data_pred:
            if mention['kb_id'] != 'NIL':
                pred_set.add((mention['kb_id'],mention['mention'],mention['offset']))

        true_num += len(true_set)
        pred_num += len(pred_set)
        equal_num += len(true_set&pred_set)
    precision = equal_num / pred_num
    recall = equal_num / true_num
    f1 = (2 * precision * recall) / (precision + recall)

    print('equal_num:',equal_num)
    print('true_num:', true_num)
    print('pred_num:', pred_num)
    print('precision:',precision)
    print('recall:', recall)
    print('f1:', f1)
    return precision, recall, f1



if __name__ == '__main__':
    # link_eval('data/dev.json', 'data/tem_result.json')

    pass