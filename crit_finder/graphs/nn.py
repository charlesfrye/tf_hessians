"""make tensorflow graphs for evaluating, minimizing, and finding critical points on neural network loss surfaces.
"""
from collections import namedtuple
import tensorflow as tf
import numpy as np

from . import generic

NeuralNetwork = namedtuple("NeuralNetwork",
                           ["graph", "graph_dictionary", "hyperparameters"])

DEFAULTS = {"learning_rate":0.1, # courtesy of Moritz Hardt and Ilya Sutskever
            "newton_rate":1,
            "fudge_factor":0.0,
            "inverse_method":"fudged",
            "gradient_norm_min_rate":0.1}

def make(hyperparameters):
    """make a tensorflow graph to evaluate, minimize, and find critical points
    on the loss surface of the neural network described by hyperparameters,
    which also describes the hyperparameters of the minimizer and critical point finder.
    """
    graph = tf.Graph()

    with graph.as_default():

        input_size = hyperparameters["input_size"]
        output_size = hyperparameters["output_size"]

        num_parameters = calculate_num_parameters(hyperparameters)
        hyperparameters["num_parameters"] = num_parameters
        parameters_placeholder = tf.placeholder(tf.float32, shape=[num_parameters],
                                                name="initial_parameters")
        parameters_var = tf.Variable(initial_value=parameters_placeholder,
                                         name="parameters_variable",
                                    expected_shape=[num_parameters])

        weight_matrices, bias_vectors = make_weights_and_biases(parameters_var,
                                                                hyperparameters)

        input = tf.placeholder(tf.float32, shape=[None, input_size])

        network_output = build_by_layer(input, weight_matrices, bias_vectors,
                                      hyperparameters)

        labels = tf.placeholder(tf.float32, shape=[None, output_size])
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=network_output,
                                                                     labels=labels))

        network_predictions = tf.nn.softmax(network_output, name="network_predictions")
        prediction_correct = tf.equal(tf.argmax(network_predictions,1), tf.argmax(labels,1))
        accuracy = tf.reduce_mean(tf.cast(prediction_correct, tf.float32))

        graph_dictionary = {"parameters_placeholder": parameters_placeholder,
                            "parameters": parameters_var,
                            "input": input,
                            "weight_matrices": weight_matrices,
                            "bias_vectors": bias_vectors,
                            "labels": labels,
                            "cost": cost,
                            "accuracy": accuracy,
                           }

        generic.add_gradient_ops(cost, parameters_var, graph_dictionary)
        generic.add_hess_ops(cost, parameters_var, graph_dictionary)

        generic.add_optimizer(cost, parameters_var, hyperparameters, graph_dictionary)
        generic.add_crit_finder(cost, parameters_var, num_parameters, hyperparameters, graph_dictionary)

    return NeuralNetwork(graph, graph_dictionary, hyperparameters)

def calculate_num_parameters(hyperparameters):
    """calculate the number of parameters in a neural network
    from the information in hyperparameter dictionary
    """
    layer_sizes = hyperparameters["layer_sizes"][:]
    input_sizes = hyperparameters["input_size"]
    output_size = hyperparameters["output_size"]
    layer_sizes = [input_sizes] + layer_sizes + [output_size]

    num_weights = np.sum(np.multiply(layer_sizes[1:],layer_sizes[:-1]))
    num_biases = np.sum(layer_sizes[1:])

    return num_weights+num_biases

def make_weights_and_biases(parameters, hyperparameters):
    """return lists of the weight matrices and bias vectors for
    the network described by hyperparameter dictionary
    by slicing and reshaping the entries of parameters
    """
    layer_sizes = hyperparameters["layer_sizes"][:]
    input_sizes = hyperparameters["input_size"]
    output_size = hyperparameters["output_size"]
    layer_sizes = [input_sizes] + layer_sizes + [output_size]

    weight_matrices = make_weights(parameters, layer_sizes)
    bias_start_index = get_bias_start_index(layer_sizes, hyperparameters)
    bias_vectors = make_biases(parameters, layer_sizes, bias_start_index)

    return weight_matrices, bias_vectors

def make_weights(parameters, layer_sizes):
    """make list of weight matrices communicating between layers of sizes
    layer_sizes by slicing and reshaping the entries of parameters
    """
    weight_shapes = zip(layer_sizes[:-1], layer_sizes[1:])
    starting_index = 0
    weight_matrices = []

    with tf.variable_scope("weights"):

        for weight_shape in weight_shapes:
            num_weights = weight_shape[0]*weight_shape[1]

            weight_variables = tf.slice(parameters, [starting_index], [num_weights],
                                        name="sliced")
            weight_matrix = tf.reshape(weight_variables, weight_shape,
                                       name="reshaped")

            weight_matrices.append(weight_matrix)

            starting_index += num_weights

    return weight_matrices

def make_biases(parameters, layer_sizes, bias_start_index):
    """make list of bias vectors for layers of sizes layer_sizes
    by slicing the entries of parameters starting at bias_start_index
    """
    bias_shapes = layer_sizes[1:]
    slice_start_index = bias_start_index
    bias_vectors = []

    with tf.variable_scope("biases"):

        for bias_shape in bias_shapes:
            num_biases = bias_shape

            bias_vector = tf.slice(parameters, [slice_start_index], [num_biases],
                                     name="sliced")

            bias_vectors.append(bias_vector)

            slice_start_index += num_biases

    return bias_vectors

def get_bias_start_index(layer_sizes, hyperparameters):
    """determine and return the appropriate starting index for the biases
    inside the parameters vector, and update the hyperparameters
    with the total numbers of weights and biases
    """
    bias_shapes = layer_sizes[1:]
    total_biases = np.sum(bias_shapes)
    total_weights = hyperparameters["num_parameters"]-total_biases
    hyperparameters["total_weights"] = total_weights
    hyperparameters["total_biases"] = total_biases
    starting_index = total_weights-total_biases

    return starting_index

def build_by_layer(input, weight_matrices, bias_vectors, hyperparameters):
    """build the network described by hyperparameters layerwise
    from the layerwise lists of weight_matrices and bias_vectors,
    starting with the input and ending with an "output layer"
    that has no nonlinearity applied.
    """
    current_output = input

    for weight_matrix, bias_vector in zip(weight_matrices[:-1], bias_vectors[:-1]):
        current_output = build_layer(current_output, weight_matrix, bias_vector,
                             hyperparameters)

    final_output = build_output_layer(current_output, weight_matrices[-1], bias_vectors[-1],
                                      hyperparameters)

    return final_output

def build_layer(current_output, weight_matrix, bias_vector, hyperparameters):
    """build a layer that applies affine transformation parametrized by weight_matrix and bias_vector
    to current output, applies nonlinearity, and returns the output
    """
    with tf.variable_scope("internal_layers"):
        nonlinearity = hyperparameters["nonlinearity"]
        new_output = nonlinearity(tf.add(tf.matmul(current_output, weight_matrix), bias_vector))
    return new_output

def build_output_layer(current_output, weight_matrix, bias_vector, hyperparameters):
    """build a layer that applies the affine transformation parametrized by weight_matrix and bias_vector
    to current output and returns the output
    """
    with tf.variable_scope("output_layer"):
        final_output = tf.add(tf.matmul(current_output, weight_matrix), bias_vector)
    return final_output
