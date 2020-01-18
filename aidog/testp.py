import glob
import os
import argparse
import requests
import json
import tensorflow as tf
import numpy as np
import base64
from PIL import Image
import sys
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import csv

if __name__ == "__main__":
    WSI_MASK_PATH = './aidog/Images/*/'  # 存放图片的文件夹路径
    paths = glob.glob(os.path.join(WSI_MASK_PATH, '*.jpg'))
    for path in paths:
        file_name = path
        #file_name = "10.jpg"

        I = Image.open(file_name)
        I = I.convert("RGB")
        I_array = np.array(I)
        encoded_string = np.expand_dims(I_array, axis=0)

        endpoint = "http://10.250.210.8:8500"
        json_data = {"model_name": "default1",
                    "model_version": 3,
                    "data": {"inputs": encoded_string.tolist()}
                    }
        result = requests.post(endpoint, json=json_data)
        i = np.array(json.loads(result.text)["num_detections"][0])
        i = i.astype(int)

        for j in range(0,i):
            res = np.array(json.loads(result.text)["detection_boxes"][0][j])
            if(json.loads(result.text)["detection_classes"][0][j] == 18):
                w = I.size[0]
                h = I.size[1]
                a = res[1]*w
                b = h*res[0]
                c = res[3]*w
                d = h*res[2]
                cropped = I.crop((a, b, c, d))  # (left, upper, right, lower)
                cropped.save("temp.jpg")
                with open("temp.jpg", "rb") as image_file:
                    base_string = str(base64.urlsafe_b64encode(image_file.read()), "utf-8")
                endpoint = "http://10.249.81.246:8500"
                json_data = {"model_name": 'default',
                             "model_version": 1,
                             "data": {"image": base_string}
                             }
                result = requests.post(endpoint, json=json_data)
                res = np.array(json.loads(result.text)["prediction"][0])
                res = np.expand_dims(res, axis=0)
                res = pd.DataFrame(res)
                res.insert(0, 'name', file_name)
                path = 'dog.csv'
                with open(path , 'a+') as f:
                    csv_write = csv.writer(f)
                    csv_write.writerow(res.iloc[0])



