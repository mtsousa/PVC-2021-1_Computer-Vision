from tensorflow.python.saved_model import tag_constants
from tensorflow.python.tools import freeze_graph
from tensorflow.python import ops
from tensorflow.tools.graph_transforms import TransformGraph
import tensorflow as tf
import os
import argparse

def get_graph_def_from_file(graph_filepath):
    tf.compat.v1.reset_default_graph
    with ops.Graph().as_default():
        with tf.io.gfile.GFile(graph_filepath, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            return graph_def

def optimize_graph(in_graph, out_graph, input_names, output_names, transforms):
    graph_def = get_graph_def_from_file(in_graph)
    optimized_graph_def = TransformGraph(
        graph_def,
        input_names,  
        output_names,
        transforms)
    tf.io.write_graph(optimized_graph_def,
        logdir='./',
        as_text=False,
        name=out_graph)
    print('Graph optimized!')

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--in_graph', required=True)
parser.add_argument('--out_graph', required=True)
parser.add_argument('--inputs', required=True, nargs='+')
parser.add_argument('--outputs', required=True, nargs='+')
parser.add_argument('--transforms', required=True, nargs='+')
args = parser.parse_args()

optimize_graph(args.in_graph,
               args.out_graph,
               args.inputs,
               args.outputs,
               args.transforms)