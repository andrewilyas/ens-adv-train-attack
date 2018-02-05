import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np

def one_hot(index, total):
    arr = np.zeros((total))
    arr[index] = 1.0
    return arr

def optimistic_restore(session, save_file):
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
            if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            curr_var = tf.get_variable(saved_var_name)
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                restore_vars.append(curr_var)
    saver = tf.train.Saver(restore_vars)
    saver.restore(session, save_file)

def t_optimistic_restore(session, save_file):
    var_names = [var for var in tf.global_variables()
            if "InceptionV3" in var.name.split(':')[0]]
    saver = tf.train.Saver(var_names)
    saver.restore(session, save_file)
    
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def int_shape(tensor):
    return list(map(int, tensor.get_shape()))

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

def hms(seconds):
    seconds = int(seconds)
    hours = (seconds // (60 * 60))
    minutes = (seconds // 60) % 60
    seconds = seconds % 60
    if hours > 0:
        return '%d hrs %d min' % (hours, minutes)
    elif minutes > 0:
        return '%d min %d sec' % (minutes, seconds)
    else:
        return '%d sec' % seconds

_py_func_id = 0
def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    global _py_func_id

    rnd_name = 'PyFuncGrad' + '%08d' % _py_func_id
    _py_func_id += 1

    tf.RegisterGradient(rnd_name)(grad)
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name, "PyFuncStateless": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

def grad_clip_by_norm(x, clip_norm=1, name=None):
    if isinstance(clip_norm, int):
        clip_norm = float(clip_norm)
    with ops.name_scope(name, "grad_clip_by_norm", [x, clip_norm]) as name:
        identity, = py_func(
            lambda t,
            _: t,
            [x, clip_norm],
            [tf.float32],
            name=name,
            grad=_grad_clip_by_norm_grad,
            stateful=False
        )
        identity.set_shape(x.get_shape())
        return identity

# Actual gradient:
def _grad_clip_by_norm_grad(op, grad):
    _, norm = op.inputs
    return (tf.clip_by_norm(grad, norm), None)

def grad_clip_by_value(x, clip_magnitude=1, name=None):
    if isinstance(clip_magnitude, int):
        clip_magnitude = float(clip_magnitude)
    with ops.name_scope(name, "grad_clip_by_value", [x, clip_magnitude]) as name:
        identity, = py_func(
            lambda t,
            _: t,
            [x, clip_magnitude],
            [tf.float32],
            name=name,
            grad=_grad_clip_by_value_grad,
            stateful=False
        )
        identity.set_shape(x.get_shape())
        return identity

# Actual gradient:
def _grad_clip_by_value_grad(op, grad):
    _, mag = op.inputs
    return (tf.clip_by_value(grad, -mag, mag), None)
