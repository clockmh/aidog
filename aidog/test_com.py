import argparse
import requests
import json
#import tensorflow as tf
import numpy as np
import base64
from PIL import Image
import sys
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def cos_sim(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a
    :param vector_b: 向量 b
    :return: sim
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim


csv_data = pd.read_csv('dog.csv')
num = csv_data.shape[0]
max = -1
index_n = -1
for index in range(0, num):
    b = csv_data.iloc[index]
    #temp_n = cosine_similarity(0, b)
    temp_n = cos_sim(b,b)
    if (temp_n > max):
        max = temp_n
        index_n = index
    print(index)
print([index_n, max])