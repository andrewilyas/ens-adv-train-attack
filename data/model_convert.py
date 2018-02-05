import tensorflow as tf
import sys
sess = tf.Session()
saver = tf.train.import_meta_graph(sys.argv[1] + ".meta")
saver.restore(sess, sys.argv[1])
saver_v1 = tf.train.Saver(write_version=tf.train.SaverDef.V1)
saver_v1.save(sess, sys.argv[2], write_meta_graph=False, write_state=False)
