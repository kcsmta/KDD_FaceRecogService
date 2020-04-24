import tensorflow as tf
import os

class Facenet(object):
    """ Class for facenet calculate embedds
    """
    def __init__(self, path_to_model):
        self.graph = tf.Graph()
        with self.graph.as_default():
            # load_facenet(path_to_model)
            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True)))
            model_exp = os.path.expanduser(path_to_model)
            print('Model filename: %s' % model_exp)
            with tf.gfile.FastGFile(model_exp, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name='')
            self.inputs_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            
    def runEmbedd(self, input_images):
        # with tf.Session(graph=self.graph) as sess:
        feed_dict = {self.inputs_placeholder: input_images,self.phase_train_placeholder:False}
        emb_arrays = self.sess.run(self.embeddings, feed_dict=feed_dict)
        return emb_arrays