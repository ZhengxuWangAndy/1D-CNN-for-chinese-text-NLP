import pandas as pd
import numpy as np
import time
import jieba
from gensim import models
from gensim.models import word2vec
from gensim import corpora
import matplotlib.pyplot as plt
import tensorflow as tf
import pkuseg
import re
import logging


contents, labels = [], []

# jieba = pkuseg.pkuseg(model_name = "medicine", user_dict = "分词字典.txt", postag = False)

testdata = pd.read_excel('/测试CASE(2)(1).xlsx')

testlabel = testdata['属性1']
# testcontent = testdata['客户项目名称']

labels = []

for i in range(len(testlabel)):
    if testlabel[i]:
        if testlabel[i] == '药品':
            labels.append(1)
        elif testlabel[i] == '项目':
            labels.append(2)
        elif testlabel[i] == '器械':
            labels.append(3)
        else:
            print(testlabel[i])


# with open('/Users/og_wang/Documents/亚信/CNN文本分类/模型/data/test.txt') as f:
#     for line in f:
#         try:
#             label, content = line.strip().split('\t')
#             if content:
#                 contents.append(content)
#                 if label == '药品':
#                     labels.append(1)
#                 elif label == '项目':
#                     labels.append(2)
#                 elif label == '器械':
#                     labels.append(3)
#                 else:
#                     print(label)
#         except:
#             pass
# print(len(contents))
print(len(labels))
print(labels[100])


from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
one_hot = encoder.fit_transform(labels)

# temp = np.array(labels) 



np.save("testlabel7000.npy",one_hot)



# vocab = []
# with open('vocab.txt','w+') as savefile:
#     for i in range(0,len(contents)):
#         result = re.sub("[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）]+", "",contents[i])
#         seg = jieba.cut(result)
#         vocab.append(seg)
#         savefile.write('\n'.join(seg))
#         savefile.write('\n')

# #####训练医疗项目模型并保存#####
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# sentences = word2vec.LineSentence('vocab.txt')

# careProgramModel = word2vec.Word2Vec(sentences, sg=0, hs=1, min_count=1, window=4, size=100)

# careProgramModel.save("vocab.model")
# #####################

