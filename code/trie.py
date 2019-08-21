import pandas as pd
import json
class Trie:
    '''
    字典树
    '''
    def __init__(self):
        self.root = {}
        self.end = -1

    def insert(self, word):
        curNode = self.root
        for c in word:
            if not c in curNode:
                curNode[c] = {}
            curNode = curNode[c]
        curNode[self.end] = True

    def search(self, word):
        curNode = self.root
        for c in word:
            if not c in curNode:
                return False
            curNode = curNode[c]
        if not self.end in curNode:
            return False
        return True

    def startsWith(self, prefix):
        curNode = self.root
        for c in prefix:
            if not c in curNode:
                return False
            curNode = curNode[c]
        return True

    def search_entity(self,text):
        '''
        正向最大实体搜索
        :param text: 一段文本
        :return: 文本中实体列表
        '''
        text=text.lower()
        entitys=[]
        i = 0
        while i < len(text):
            e = i+1
            param = self.startsWith(text[i:e])
            if param:
                en = text[i:e]
                while e <= len(text):
                    p = self.startsWith(text[i:e])
                    if p:
                        en = text[i:e]
                        e += 1
                    else:
                        break
                if self.search(en):
                    entitys.append((en,i))
                    i = e - 1
                    #i+=1
                else:
                    i += 1
            else:
                i += 1
        return entitys

def get_Trie(min_len=2):
    '''
    构建实体字典树，长度小于2的不插入
    :param min_len: 实体长度
    :return:
    '''
    kb_en = pd.read_pickle('data/entity_id.pkl')
    trie_obj = Trie()
    for en in kb_en:
        if len(en)>=min_len:
            trie_obj.insert(en)
    return trie_obj

def get_split_entity():
    '''
    最大匹配时会出现一些实体重复，如 迅雷、下载 和 迅雷下载三个实体，还有 视频、报道 和 视频报道三个实体
    如果不处理最大匹配时将会漏掉 迅雷、下载 两个实体，仅仅会匹配 迅雷下载 这一个实体。为处理这种情况，统计他们
    出现的次数并根据出现次数来决定这类实体该怎么处理。处理分一下三种情况：
        1. 仅保留最大的实体，如迅雷下载
        2. 分开，保留小的实体具体保留那个看统计数据 如 迅雷
        3. 都保留 如 迅雷 下载 迅雷下载
    :return: 输入为字典形式，如：
            '迅雷下载': {'迅雷'},
            '高清视频': {'视频', '高清视频'}
    '''
    def entity_split():
        trie_obj = get_Trie()
        entity_count = {}
        with open('original_data/train.json') as f:
            for i, line in enumerate(f):
                temDict = json.loads(line)
                text = temDict['text']
                mention_data = temDict['mention_data']
                match_list = trie_obj.search_entity(text)
                for men in mention_data:
                    offset = int(men['offset'])
                    end = offset + len(men['mention'])
                    for en in match_list:
                        b= en[1]
                        e =en[1]+len(en[0])
                        if offset>=b and end <=e:
                            if (e-b) != (end-offset):
                                if len(men['mention'])>1:
                                    if en[0] in entity_count:
                                        entity_count[en[0]]['count']+=1
                                        if men['mention'] in entity_count[en[0]]['en']:
                                            entity_count[en[0]]['en'][men['mention']] += 1
                                        else:
                                            entity_count[en[0]]['en'][men['mention']] = 1
                                    else:
                                        entity_count[en[0]] = {}
                                        entity_count[en[0]]['count'] =1
                                        entity_count[en[0]]['en']={}
                                        entity_count[en[0]]['en'][men['mention']]=1
        return entity_count
    def get_sub_entity(count,en_dict):
        sub_entity=set()
        for en in en_dict:
            if en_dict[en]/count>0.42 and en_dict[en]>3:
                sub_entity.add(en)
        return sub_entity
    en_dict=entity_split()
    entity_full=dict()
    with open('original_data/train.json') as f:
        for i,line in enumerate(f):
            temDict = json.loads(line)
            mention_data = temDict['mention_data']
            for men in mention_data:

                if men['mention'] in en_dict:
                    if men['mention'] in entity_full:
                        entity_full[men['mention']]+=1
                    else:
                        entity_full[men['mention']] = 1
    split_dict=dict()
    del_en = {'经典名句','经典老歌','名段欣赏'}
    for en in en_dict:
        if en in del_en:
            continue
        if en_dict[en]['count']<10:
            continue
        if en in entity_full:
            if entity_full[en]/en_dict[en]['count']>1.5:
                continue
            if en_dict[en]['count']/entity_full[en] > 3:
                sub_entity = get_sub_entity(en_dict[en]['count'],en_dict[en]['en'])
                if len(sub_entity) > 0:
                    split_dict[en] = sub_entity
                continue
            sub_entity = get_sub_entity(en_dict[en]['count'], en_dict[en]['en'])
            sub_entity.add(en)
            if len(sub_entity)>0:
                split_dict[en]=sub_entity
        else:
            pass
            sub_entity = get_sub_entity(en_dict[en]['count'], en_dict[en]['en'])
            if len(sub_entity)>0:
                split_dict[en]=sub_entity

    pd.to_pickle(split_dict,'data/split_entity.pkl')
    print(len(split_dict))
    for en in split_dict:
        print(en,split_dict[en])
def fun_prob(x):
    return -0.066*x +1

def entity_link_prob():
    '''
    统计每个通过上述匹配方式，匹配到的mention出现的次数，以及被链接的次数，计算
    匹配链接比=匹配次数/链接次数 按照匹配链接比，计算排序前15000实体 并且 匹配次数>15
    计算实体链接概率得分
    :return:
    '''
    kb_en = pd.read_pickle('data/entity_id.pkl')
    entity_dict={}
    for en in kb_en:
        entity_dict[en]={'link_num':0.1,'match_num':0.1}
    trie_obj = Trie()
    for en in kb_en:
        if len(en)>1:
            trie_obj.insert(en)
    with open('original_data/train.json') as f:
        for i,line in enumerate(f):
            temDict = json.loads(line)
            text = temDict['text']
            mention_data = temDict['mention_data']
            match_list=trie_obj.search_entity(text)
            for men in mention_data:
                if men['mention'] in entity_dict:
                    entity_dict[men['mention']]['link_num']+=1
            for en in match_list:
                if en[0] in entity_dict:
                    entity_dict[en[0]]['match_num']+=1

    entity_num={}
    for en in entity_dict:
        entity_num[en]=entity_dict[en]['match_num']/entity_dict[en]['link_num']
    en_sorted = sorted(entity_num.items(), key=lambda item: item[1],reverse=True)
    entity_prob={}
    for i in en_sorted[:15000]:
        if entity_dict[i[0]]['match_num']>15:
            # print(i,entity_dict[i[0]]['match_num'],sigmoid(i[1]))
            entity_prob[i[0]] = fun_prob(i[1])
        # print(i)
    return entity_prob
if __name__ == '__main__':
    get_split_entity()
    # trie_obj = get_Trie()
    # text='《大话英雄·联盟》-原创-高清视频'
    # match_list = trie_obj.search_entity(text)
    # print(match_list)

    # pass
