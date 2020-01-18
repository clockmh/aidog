import argparse

import requests

import json

import tensorflow as tf

import numpy as np

import base64

from PIL import Image




def load_labels(label_file):

  label = []

  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()

  for l in proto_as_ascii_lines:

    label.append(l.rstrip())

  return label





if __name__ == "__main__":

  file_name = "10.jpg"

  label_file = "dog_labels_inception_v3.txt"

  model_name = "default1"

  model_version = 3

  enable_ssl = False



  parser = argparse.ArgumentParser()

  parser.add_argument("--image", help="image to be processed")

  parser.add_argument("--labels", help="name of file containing labels")

  parser.add_argument("--model_name", help="name of predict model")

  parser.add_argument("--model_version", type=int, help="version of predict model")

  parser.add_argument("--enable_ssl", type=bool, help="if use https")

  args = parser.parse_args()



  if args.image:

    file_name = args.image

  if args.labels:

    label_file = args.labels

  if args.model_name:

    model_name = args.model_name

  if args.enable_ssl:

    enable_ssl = args.enable_ssl



  I = Image.open(file_name)
  I = I.convert("RGB")
  I_array = np.array(I)
  encoded_string = np.expand_dims(I_array, axis=0)



  if enable_ssl :

    endpoint = "https://127.0.0.1:8500"

  else:

    endpoint = "http://10.250.210.8:8500"



  json_data = {"model_name": model_name,

               "model_version": model_version,

               "data": {"inputs": encoded_string.tolist()}

              }

  result = requests.post(endpoint, json=json_data)

  print(result.text)

  i = np.array(json.loads(result.text)["num_detections"][0])

  i = i.astype(int)

  for j in range(0,i):

    res = np.array(json.loads(result.text)["detection_boxes"][0][j])

    w = I.size[0]

    h = I.size[1]

    a = res[1]*w

    b = h*res[0]

    c = res[3]*w

    d = h*res[2]

    cropped = I.crop((a, b, c, d))  # (left, upper, right, lower)

    cropped.save("pil_cut_thor%d.jpg" % j)

  '''res = np.array(json.loads(result.text)["prediction"][0])

  print(res)

  indexes = np.argsort(-res)

  labels = load_labels(label_file)

  top_k = 3

  for i in range(top_k):

    idx = indexes[i]

    print(labels[idx], res[idx])'''