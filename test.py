# import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import time
import os
import pkuseg
from gensim.models import Word2Vec
import matplotlib.pyplot as plt

model = load_model('cnnmulti.model')


word2vec_model = Word2Vec.load('data/vocab.model')

testdata = pd.read_excel('测试CASE(2)(1).xlsx')

information = testdata['客户项目名称']

jieba = pkuseg.pkuseg(model_name = "medicine", user_dict = "分词字典.txt", postag = False)

data=[]
for i in range(0,len(information)):
    segs = information[i]
    cut = jieba.cut(segs)
    # 分词的输出结果为一个生成器
    # 把生成器转换为list
    cut_list = [i for i in cut]
#     list = ','.join(jieba.lcut(segs)).split(',')
    for i, word in enumerate(cut_list):
        try:
            # 将词转换为索引index
            cut_list[i] = word2vec_model.wv.vocab[word].index
        except:
            # 如果词不在字典中，则输出0
            cut_list[i] = 0
    data.append(np.array(cut_list))

# 获得所有data的长度
num_tokens = [ len(tokens) for tokens in data ]
# 将列表转换为 np 格式
num_tokens = np.array(num_tokens)
# 最长的评价tokens的长度
maxlength = np.max(num_tokens)

sentenceNum = len(data)

embedding_matrix = np.zeros((sentenceNum, maxlength), dtype='int32')
fileCounter = 0
for i in range(sentenceNum):
    indexCounter = 0
    for j in range(len(data[i])):
        if j < maxlength:
            try:
                embedding_matrix[fileCounter][indexCounter] = data[i][j]
            except:
                embedding_matrix[fileCounter][indexCounter] = 0
        else:
            continue
        indexCounter += 1
    fileCounter += 1

print(len(embedding_matrix))    #161836


x_test = np.array(embedding_matrix) 


# Evaluate the model on the test data using `evaluate`
# print("Evaluate on test data")
# results = model.evaluate(x_test, y_test, batch_size=32)
# print("test loss, test acc:", results)

# Generate predictions (probabilities -- the output of the last layer)
# on new data using `predict`
print("Generate predictions for 3 samples")


predict = model.predict(x_test)
predictions=np.argmax(predict,axis=1)

print("predictions shape:", predictions.shape)

predictions = predictions.tolist()


testResult = []
for i in range(len(predictions)):
    if predictions[i] == 0:
        testResult.append('药品')
    elif predictions[i] == 1:
        testResult.append('项目')
    elif predictions[i] == 2:
        testResult.append('器械')
    else:
        testResult.append('其他')

df = pd.DataFrame(testResult)
df.to_excel('模型预测结果.xlsx',sheet_name = 'Sheet2')





