import json
import pandas as pd
def entity_clear(entity):
    '''
    将一些特殊字符替换
    :param entity: 一个实体名字
    :return: 替换后的实体
    '''
    pun = {'，': ',',
           '·': '•',
           '：': ':',
           '！': '!',
           }
    for p in pun:
        if p in entity:
            entity=entity.replace(p,pun[p])
    return entity
def new_alias():
    '''
    统计训练数据中不能链接到实体库的mentioin, 统计出现次数，将其添加到对应实体的别名中
    :return: 字典形式 key 为实体名字 value 为添加的新的别名字典
             如：'bilibili': {'b站', '哔哩哔哩', '哔哩哔哩弹幕视频网'}
    '''
    id_alias = {}
    entity_id={}
    id_entity={}
    with open('original_data/kb_data', 'r') as f:
        for line in f:
            temDict = json.loads(line)
            subject = temDict['subject']
            subject_id = temDict['subject_id']
            alias = set()
            for a in temDict['alias']:
                alias.add(a)
                alias.add(a.lower())
            alias.add(subject.lower())
            alias.add(entity_clear(subject))
            id_alias[subject_id] = alias
            subject_id = temDict['subject_id']
            entity_name = set(alias)
            entity_name.add(subject)
            entity_name.add(subject.lower())
            for a in alias:
                entity_name.add(a.lower())
            id_entity[subject_id] = subject
            for n in entity_name:
                if n in entity_id:
                    entity_id[n].add(subject_id)
                else:
                    entity_id[n] = set()
                    entity_id[n].add(subject_id)
    with open('original_data/train.json') as f:
        entity_alias_num={}
        for line in f:
            temDict = json.loads(line)
            mention_data=temDict['mention_data']
            for men in mention_data:
                mention=men['mention']
                kb_id = men['kb_id']
                if kb_id != 'NIL':
                    if id_entity[kb_id]!=mention:
                        if mention not in id_alias[kb_id]:
                            if id_entity[kb_id] in entity_alias_num:
                                entity_alias_num[id_entity[kb_id]]['count'] +=1
                                if mention in entity_alias_num[id_entity[kb_id]]:
                                    entity_alias_num[id_entity[kb_id]][mention] += 1
                                else:
                                    entity_alias_num[id_entity[kb_id]][mention] = 1
                            else:
                                entity_alias_num[id_entity[kb_id]]={}
                                entity_alias_num[id_entity[kb_id]]['count']=1
                                entity_alias_num[id_entity[kb_id]][mention]=1
    entity_alias={}
    for en in entity_alias_num:
        total_num=entity_alias_num[en]['count']
        if total_num>4:
            entity_alias[en]=set()
            for alias in entity_alias_num[en]:
                if alias=='count':
                    continue
                a_num=entity_alias_num[en][alias]
                if a_num>3:
                    entity_alias[en].add(alias)
            if len(entity_alias[en])==0:
                entity_alias.pop(en)
    return entity_alias


def get_len(text_lens, max_len=510, min_len=30):
    """
    戒断过长文本你的长度，小于30不在戒断，大于30按比例戒断
    :param text_lens: 列表形式 data 字段中每个 predicate+object 的长度
    :param max_len: 最长长度
    :param min_len: 最段长度
    :return: 列表形式 戒断后每个 predicate+object 保留的长度
            如 input：[638, 10, 46, 9, 16, 22, 10, 9, 63, 6, 9, 11, 34, 10, 8, 6, 6]
             output：[267, 10, 36, 9, 16, 22, 10, 9, 42, 6, 9, 11, 31, 10, 8, 6, 6]

    """
    new_len = [min_len]*len(text_lens)
    sum_len = sum(text_lens)
    del_len = sum_len - max_len
    del_index = []
    for i, l in enumerate(text_lens):
        if l > min_len:
            del_index.append(i)
        else:
            new_len[i]=l
    del_sum = sum([text_lens[i]-min_len for i in del_index])
    for i in del_index:
        new_len[i] = text_lens[i] - int(((text_lens[i]-min_len)/del_sum)*del_len) - 1
    return new_len

def get_text(en_data,max_len=510,min_len=30):
    '''
    根据data字段数据生成描述文本，将 predicate项与object项相连，在将过长的依据规则戒断
    :param en_data: kb里面的每个实体的data数据
    :param max_len: 每个 predicate+object 的最大长度
    :param min_len: 每个 predicate+object 的最小长度
    :return: 每个实体的描述文本
    '''
    texts = []
    text = ''
    for data in en_data:
        texts.append(data['predicate'] + ':'+ data['object'] + '，')
    text_lens=[]
    for t in texts:
        text_lens.append(len(t))
    if sum(text_lens)<max_len:
        for t in texts:
            text=text+t
    else:
        new_text_lens=get_len(text_lens,max_len=max_len,min_len=min_len)
        for t,l in zip(texts,new_text_lens):
            text=text+t[:l]
    return text[:max_len]

def del_bookname(entity_name):
    '''
    删除书名号
    :param entity_name: 实体名字
    :return: 删除后的实体名字
    '''
    if entity_name.startswith(u'《') and entity_name.endswith(u'》'):
        entity_name = entity_name[1:-1]
    return entity_name

def kb_processing():
    '''
    知识库处理
    :return: 得到后续要用的一些文件具体为：
            entity_id字典  key:entity name  value: kb_id list
                        '胜利': ['10001', '19044', '37234', '38870', '40008', '85426', '86532', '140750']
            id_entity字典 key:kb_id value:subject(实体名字)
                         10001 胜利
            id_text字典 key：kb_id value:实体描述文本
                       '10001': '摘要:英雄联盟胜利系列皮肤是拳头公司制作的具有纪念意义限定系列皮肤之一。'
            id_type字典  key：kb_id value: entity type
                        '10001': ['Thing']

            type_index字典 key：type name value：index
                        ‘NAN’: 0
                        'Thing' :1
    '''
    new_entity_alias=new_alias()
    id_text={}
    entity_id={}
    type_index={}
    type_index['NAN']=0
    type_i=1
    id_type={}
    id_entity={}

    with open('original_data/kb_data', 'r') as f:
        for line in f:
            temDict = json.loads(line)
            subject=temDict['subject']
            subject_id=temDict['subject_id']
            alias = set()
            for a in temDict['alias']:
                alias.add(a)
                alias.add(a.lower())
            alias.add(subject.lower())
            alias.add(entity_clear(subject))
            if subject in new_entity_alias:
                alias=alias|new_entity_alias[subject]
            en_data=temDict['data']
            en_type=temDict['type']
            entity_name=set(alias)
            entity_name.add(subject)
            for t in en_type:
                if not t in type_index:
                    type_index[t]=type_i
                    type_i+=1
            for n in entity_name:
                n=del_bookname(n)
                if n in entity_id:
                    entity_id[n].append(subject_id)
                else:
                    entity_id[n]=[]
                    entity_id[n].append(subject_id)
            id_type[subject_id]=en_type
            text=get_text(en_data)
            id_text[subject_id]=text
            id_entity[subject_id]=subject

    pd.to_pickle(entity_id,'data/entity_id.pkl')
    pd.to_pickle(id_entity, 'data/id_entity.pkl')
    pd.to_pickle(type_index, 'data/type_index.pkl')
    pd.to_pickle(id_type, 'data/id_type.pkl')
    pd.to_pickle(id_text, 'data/id_text.pkl')



if __name__ == '__main__':

    kb_processing()

    pass