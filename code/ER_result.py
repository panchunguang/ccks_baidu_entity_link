import numpy as np
import pandas as pd
import json
from evaluate import mention_evaluate
from trie import entity_link_prob

def filterEntity(ner_men,match_men):
    '''
    将NER模型与Match模型得到结果融合，具体融合思路：如果NER得到结果与Match得到结果有交叉，选择Match得到的结果
    :param ner_men:
    :param match_men:
    :return:
    '''
    info_del=[]
    for ner in ner_men:
        for match in match_men:
            offset_ner=int(ner['offset'])
            end_ner=offset_ner+len(ner['mention'])
            offset_match = int(match['offset'])
            end_match=offset_match+len(match['mention'])
            if offset_ner==offset_match and end_ner==end_match:
                match['ner_num'] = ner['ner_num']
                ner['m_pred'] = match['m_pred']
            if max(end_ner,end_match)-min(offset_ner,offset_match) <len(ner['mention'])+len(match['mention']):
                info_del.append(ner)

    ner_men=[ner for ner in ner_men if ner not in info_del]
    return ner_men+match_men

def get_match_entity(match_en_prob,entity_prob):
    match_set=set()
    for match in match_en_prob:
        p = 1
        if match[0] in entity_prob:
            p = entity_prob[match[0]]
        if match_en_prob[match]*p > 0.45:
            match_set.add((match[0], str(match[1]),match_en_prob[match]*p))
    return match_set
def get_bio_dict():
    bio_dict={}
    bio_dict['O']=0
    bio_dict['B'] = 1
    bio_dict['I'] = 2
    bio_dict['TAG'] = 3
    return bio_dict
def get_index_bio():
    bio_dict=get_bio_dict()
    index_bio={}
    for k in bio_dict:
        index_bio[bio_dict[k]]=k
    index_bio[3]='O'
    return index_bio
def get_entity(lable,index_bio):
    '''
    根据预测得到label标签解析出来实体，得到实体开始和结束位置
    :param lable:
    :param index_bio:
    :return:
    '''
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

def get_result(input_file,ner_pred_file,match_file,match_pred,out_file):
    '''
    根据NER模型以及Match模型的预测结果，进行融合，得到实体识别阶段的最终结果
    :param input_file: 原始文件
    :param ner_pred_file: NER预测文件
    :param match_file: Match输入文件,这个里面有匹配的实体，下面有对应的概率
    :param match_pred: Match预测文件
    :param out_file:
    :return:
    '''
    entity_id = pd.read_pickle('data/entity_id.pkl')
    link_prob = entity_link_prob()
    index_bio = get_index_bio()
    pred_all = np.load(ner_pred_file)
    pred_all = list(map(list, zip(*pred_all)))
    result_out = open(out_file,'w')
    match_pred=np.load(match_pred)
    print(match_pred.shape)
    match_entity = pd.read_pickle(match_file)['match_en']
    with open(input_file, 'r') as f:
        for line,pred,match_en,match_p in zip(f,pred_all,match_entity,match_pred):
            out_dict={}
            temDict = json.loads(line)
            text = temDict['text']
            text_id = temDict['text_id']
            entity_count = {}
            for p in pred:
                en_list = get_entity(np.argmax(p, axis=-1), index_bio)
                for en in en_list:
                    entity_count[en]=entity_count.get(en,0)+1

            entity_prob = {}
            for en,p in zip(match_en,match_p):
                if p>0.01:
                    entity_prob[en]=p[0]

            ner_set = set()
            match_set = get_match_entity(entity_prob,link_prob)

            for k in entity_count:
                if entity_count[k]>8 and text[k[0]-1:k[1]] in entity_id:
                    ner_set.add((text[k[0]-1:k[1]],str(k[0]-1),entity_count[k]))

            ner_list=[]
            match_list = []
            for en in ner_set:
                ner_list.append({'mention':en[0],'offset':en[1],'ner_num':en[2],'m_pred':1,})
            for en in match_set:
                match_list.append({'mention':en[0],'offset':en[1],'ner_num':18,'m_pred':en[2],})

            entity_list=filterEntity(ner_list,match_list)

            out_dict['text_id']=text_id
            out_dict['text'] = text
            out_dict['mention_data'] = entity_list
            result_out.write(json.dumps(out_dict, ensure_ascii=False))
            result_out.write('\n')

if __name__ == '__main__':
    # del_NIL()
    # get_result('data/dev.json','data/pred_dev_ner.npy','data/input_dev_match_bert.pkl','data/pred_dev_match.npy','result/dev_ner_result.json')

    # print(mention_evaluate('data/dev.json', 'result/dev_ner_result.json'))

    # get_result('original_data/develop.json', 'data/pred_test_ner.npy', 'data/input_test_match_bert.pkl',
    #            'data/pred_test_match.npy', 'result/test_ner_result.json')
    #
    # print(mention_evaluate('result/test_ner_result.json', 'result/test_ner_result.json'))

    get_result('original_data/eval722.json', 'data/pred_eval_ner.npy', 'data/input_eval_match_bert.pkl',
               'data/pred_eval_match.npy', 'result/eval_ner_result.json')
    print(mention_evaluate('result/eval_ner_result.json', 'result/eval_ner_result.json'))



    pass