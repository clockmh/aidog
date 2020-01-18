
import tensorflow as tf
import numpy as np
import base64
import sys

label_file = "output_labels.txt"
top_k = 3

def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label

with tf.Session(graph=tf.Graph()) as sess:
  sess.run(tf.global_variables_initializer())
  tf.saved_model.loader.load(sess, ["serve"], "./inception_v3/2")
  graph = tf.get_default_graph()
  with open("./timg[3].jpg", "rb") as image_file:
    encoded_string = str(base64.urlsafe_b64encode(image_file.read()), "utf-8")
  x = sess.graph.get_tensor_by_name('base64_string:0')
  y = sess.graph.get_tensor_by_name('myOutput:0')
  scores = sess.run(y,feed_dict={x: encoded_string})
  idx = np.argmax(scores, 1)
  print(scores)
  indexes = np.argsort(-scores)[0]
  labels = np.array(load_labels(label_file))
  top_k = 3
  for i in range(top_k):
    ide = indexes[i]
    print(labels[ide],',',scores[0][ide])


