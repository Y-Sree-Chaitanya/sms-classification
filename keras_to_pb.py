

"""
The following code snippet will convert the keras model file,
which is saved using model.save('kerasmodel_weight_file'),
to the freezed .pb tensorflow weight file which holds both the
network architecture and its associated weights.
"""

input_fld = ''
weight_file = 'model_rsenet_aug_sqzn.txt'
num_output = 1
write_graph_def_ascii_flag = True
prefix_output_node_names_of_final_network = 'output_node'
output_graph_name = ''

from keras.models import load_model
import tensorflow as tf
import os
import os.path as osp
from keras import backend as K

output_fld = input_fld + 'tensorflow_model/'
if not os.path.isdir(output_fld):
    os.mkdir(output_fld)
weight_file_path = osp.join(input_fld, weight_file)

K.set_learning_phase(0)
net_model = load_model(weight_file_path)

pred = [None]*num_output
pred_node_names = [None]*num_output
for i in range(num_output):
    pred_node_names[i] = prefix_output_node_names_of_final_network+str(i)
    pred[i] = tf.identity(net_model.outputs[i], name=pred_node_names[i])
print('output nodes names are: ', pred_node_names)



sess = K.get_session()

if write_graph_def_ascii_flag:
    #f = 'only_the_graph_def.pb.ascii'
    f = ''
    tf.train.write_graph(sess.graph.as_graph_def(), output_fld, f, as_text=True)
    print('saved the graph definition in ascii format at: ', osp.join(output_fld, f))



from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), pred_node_names)
graph_io.write_graph(constant_graph, output_fld, output_graph_name, as_text=False)
print('saved the constant graph (ready for inference) at: ', osp.join(output_fld, output_graph_name))

