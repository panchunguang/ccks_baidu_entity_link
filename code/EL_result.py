import json
import numpy as np
from evaluate import *
def link_result_mean(input_file_list,out_file):
    out_file = open(out_file, 'w')
    result_list=[]
    with open(input_file_list[0], 'r') as f:
        for line in f:
            temDict = json.loads(line)
            re_dict = {'text_id': temDict['text_id'], 'text': temDict['text']}
            re_dict['mention_data'] = []
            mention_data = temDict['mention_data']
            for men in mention_data:
                men['link_pred']=[0]*len(men['link_id'])
                re_dict['mention_data'].append(men)
            result_list.append(re_dict)

    for input_file in input_file_list:
        with open(input_file, 'r') as f:
            for i, line in enumerate(f):
                temDict = json.loads(line)
                mention_data = temDict['mention_data']
                mean_men_data = result_list[i]['mention_data']
                for men,mean_men in zip(mention_data,mean_men_data):
                    if len(men['link_id']) > 0:
                        men_pred = [p/len(input_file_list) for p in men['link_pred']]
                        mean_men['link_pred']=list(np.sum([mean_men['link_pred'],men_pred],axis=0))
                    else:
                        mean_men['link_pred']=[]

    for temDict in result_list:
        mention_data = temDict['mention_data']
        tem_men = []
        for men in mention_data:
            flag = True
            if len(men['link_id'])>0 and men['ner_num']>8 and men['m_pred']>0.45:
                link_pred = men['link_pred']
                arg_max = int(np.argmax(link_pred))
                if men['ner_num'] < 13 and link_pred[arg_max] <0.3:
                    flag = False
                if men['m_pred']<0.5 and link_pred[arg_max] <0.3:
                    flag = False
                if len(men['link_id'])==1:
                    if link_pred[arg_max] > 0.15 and flag==True:
                        men['kb_id'] = men['link_id'][arg_max]
                    else:
                        men['kb_id'] = 'NIL'
                else:
                    if link_pred[arg_max]>0.15 and flag==True:
                        men['kb_id'] = men['link_id'][arg_max]
                    else:
                        men['kb_id'] = 'NIL'
            else:
                men['kb_id']='NIL'
            men.pop('link_id')
            men.pop('link_pred')
            men.pop('ner_num')
            men.pop('m_pred')
            if not men['kb_id'] == 'NIL':
                tem_men.append(men)
        temDict['mention_data'] = tem_men
        out_file.write(json.dumps(temDict, ensure_ascii=False))
        out_file.write('\n')

if __name__ == '__main__':

    link_result_mean([

        'result/eval_bert_loss_final_0.json',
        'result/eval_bert_loss_final_1.json',
        'result/eval_bert_loss_final_2.json',
        'result/eval_bert_loss_final_3.json',
        'result/eval_bert_loss_final_4.json',
        'result/eval_bert_f1_final_0.json',
        'result/eval_bert_f1_final_1.json',
        'result/eval_bert_f1_final_2.json',
        'result/eval_bert_f1_final_3.json',
        'result/eval_bert_f1_final_4.json',
        'result/eval_bert_loss_384_final_0.json'
        'result/eval_bert_loss_384_final_1.json'
        'result/eval_bert_loss_384_final_6.json'
        'result/eval_bert_f1_384_final_0.json'
        'result/eval_bert_f1_384_final_1.json'
        'result/eval_bert_f1_384_final_6.json'
    ], 'result/result.json')
    link_eval('result/result.json', 'result/result.json')









