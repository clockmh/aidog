
import tensorflow as tf
from PIL import Image
import numpy as np

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

I = Image.open('n02089973_1.jpg')
I_array = np.array(I)/255
#I_array = np.array(tf.expand_dims(I_array, 0))
I_array = np.expand_dims(I_array, axis=0)



with tf.Session(config=config,graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess, ['serve'] , "E:/model/faster_rcnn_inception_resnet_v2_atrous_oid_2018_01_28/saved_model/1")
    graph = tf.get_default_graph()
    x = sess.graph.get_tensor_by_name('image_tensor:0')
    y = sess.graph.get_tensor_by_name('detection_boxes:0')
    result = sess.run(y,
           feed_dict={x: I_array})
    print(result)
