import tensorflow as tf
import tensorflow_hub as hub

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('./3/_retrain_checkpoint.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./3'))
    graph = tf.get_default_graph()
    gd = tf.graph_util.convert_variables_to_constants(sess, tf.get_default_graph().as_graph_def(), ['final_result'])
    with tf.gfile.GFile('tmodel/model.pb', 'wb') as f:
        f.write(gd.SerializeToString())
