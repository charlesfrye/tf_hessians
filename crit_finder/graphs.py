"""make tensorflow graphs for evaluating, minimizing, and finding critical points on
quadratic forms and neural network loss surfaces.
"""
from collections import namedtuple
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

QuadraticForm = namedtuple("QuadraticForm", ["graph", "graph_dictionary"])

NeuralNetwork = namedtuple("NeuralNetwork",
                           ["graph", "graph_dictionary", "hyperparameter_dictionary"])

DEFAULTS = {"learning_rate":0.1, # courtesy of Moritz Hardt and Ilya Sutskever
            "newton_rate":1,
            "fudge_factor":0.0,
            "inverse_method":"fudged",
            "gradient_norm_min_rate":0.1}

def make_quadratic_form(matrix, initial_values, hyperparameters):
    """make a tensorflow graph to evaluate, minimize, and find critical points
    on the surface 0.5*x.transpose().dot(matrix).dot(x),
    the quadratic form on x defined by matrix.

    the input to the matrix is initialized to initial_values
    and the hyperparameters of the minimizer and critical point finder
    are determined by the hyperparameters argument.
    """

    assert matrix.shape[0] == matrix.shape[1], "only square matrices can be quadratic forms"
    assert matrix.shape[0] == len(initial_values), "initial_values and matrix must match shape"

    dimension = matrix.shape[0]

    graph = tf.Graph()

    with graph.as_default():
        quadratic_form = tf.constant(matrix, name='quadratic_form')

        inputs = tf.get_variable("inputs", shape=[dimension], dtype=tf.float32,
                                initializer = tf.constant_initializer(initial_values))

        input_vector = tf.reshape(inputs, (dimension, 1))

        output = 0.5*tf.squeeze(tf.matmul(input_vector,
                                  tf.matmul(quadratic_form, input_vector),
                                      transpose_a=True,
                                  name='output'),
                        name='squeezed_output')

        gradient_descent = tf.train.GradientDescentOptimizer(hyperparameters["learning_rate"])
        step_gradient_descent = gradient_descent.minimize(output)

        gradients = tf.gradients(output, inputs, name="gradients")
        gradient_norm = 0.5*tf.square(tf.norm(gradients, name="gradient_norm"))

        hessian_matrix = tf.hessians(output, inputs, name="hessian_matrix")[0]

        graph_dictionary = {"inputs": inputs,
                           "output": output,
                           "gradients": gradients,
                            "step_gradient_descent": step_gradient_descent,
                            "hessian": hessian_matrix,
                            "gradient_norm": gradient_norm
                           }

        if "inverse_method" in hyperparameters.keys():
            inverse_hessian = invert_hessian(hessian_matrix, len(initial_values), hyperparameters)
            graph_dictionary["inverse_hessian"] = inverse_hessian

            if "newton_rate" in hyperparameters.keys():
                newton_step_ct = tf.Variable(0, trainable=False)
                newton_base = tf.train.GradientDescentOptimizer(hyperparameters["newton_rate"])
                gd_grads_and_vars = newton_base.compute_gradients(output, inputs)
                step_newton = add_step_newton(newton_base, gd_grads_and_vars, inverse_hessian, newton_step_ct)
                graph_dictionary["step_newton"] = step_newton

        if "gradient_norm_min_rate" in hyperparameters.keys():
            gradmin_step_ct = tf.Variable(0, trainable=False)

            if "gradient_norm_decay_rate" in hyperparameters.keys():
                gradient_norm_min_rate =  tf.train.exponential_decay(hyperparameters["gradient_norm_min_rate"],
                                                         newton_step_ct,
                                                        decay_steps=100,
                                                        decay_rate=1.0)
            else:
                gradient_norm_min_rate = hyperparameters["gradient_norm_min_rate"]

            if "gradient_norm_momentum" in hyperparameters.keys():
                gradient_norm_optimizer = tf.train.MomentumOptimizer(gradient_norm_min_rate,
                                                                     hyperparameters["gradient_norm_momentum"])
            else:
                gradient_norm_optimizer = tf.train.GradientDescentOptimizer(gradient_norm_min_rate)

            step_gradmin = gradient_norm_optimizer.minimize(gradient_norm, global_step = gradmin_step_ct)

            graph_dictionary["step_gradient_norm_min"] = step_gradmin

    return QuadraticForm(graph, graph_dictionary)

# neural network construction functions

def make_neural_network(hyperparameters):
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

        with tf.variable_scope("grads_and_hess"):

            gradients = tf.gradients(cost, parameters_var, name="gradients")
            gradient_norm = tf.square(tf.norm(gradients, name="gradient_norm"))
            hessian_matrix = tf.hessians(cost, parameters_var, name="hessians_output")[0]
            eigenvalues, eigenvectors = tf.self_adjoint_eig(hessian_matrix)

            graph_dictionary.update({"gradients": gradients,
                                     "gradient_norm": gradient_norm,
                                     "hessian": hessian_matrix,
                                     "eigenvalues": eigenvalues,
                                     "eigenvectors": eigenvectors
                                    })

        with tf.variable_scope("optimizers"):

            with tf.variable_scope("gradient_descent"):

                gradient_descent_step_ct = tf.Variable(0, trainable=False)
                gradient_descent_rate = hyperparameters["learning_rate"]

                if "gradient_descent_decay_rate" in hyperparameters.keys():
                    assert gradient_descent_decay_every in hyperparameters.keys(), "missing decay_steps for gradient_descent"
                    gradient_descent_rate =  tf.train.exponential_decay(gradient_descent_rate,
                                                                 gradient_descent_step_ct,
                                                                decay_steps=hyperparameters["gradient_descent_decay_every"],
                                                                decay_rate=hyperparameters["gradient_descent_decay_rate"])

                if "gradient_descent_momentum" in hyperparameters.keys():
                    gradient_descent = tf.train.MomentumOptimizer(gradient_descent_rate,
                                                                  hyperparameters["gradient_descent_momentum"])
                else:
                    gradient_descent = tf.train.GradientDescentOptimizer(gradient_descent_rate)

                step_gradient_descent = gradient_descent.minimize(cost)

                graph_dictionary["step_gradient_descent"] = step_gradient_descent

            with tf.variable_scope("newton"):

                if "inverse_method" in hyperparameters.keys():
                    inverse_hessian = invert_hessian(hessian_matrix, num_parameters, hyperparameters)
                    graph_dictionary["inverse_hessian"] = inverse_hessian

                    if "newton_rate" in hyperparameters.keys():
                        newton_step_ct = tf.Variable(0, trainable=False)
                        newton_base = tf.train.GradientDescentOptimizer(hyperparameters["newton_rate"])
                        gd_grads_and_vars = newton_base.compute_gradients(cost, parameters_var)
                        step_newton = add_step_newton(newton_base, gd_grads_and_vars, inverse_hessian, newton_step_ct)
                        graph_dictionary["step_newton"] = step_newton

            with tf.variable_scope("gradient_norm_min"):

                if "gradient_norm_min_rate" in hyperparameters.keys():
                    gradmin_step_ct = tf.Variable(0, trainable=False)

                    if "gradient_norm_decay_rate" in hyperparameters.keys():
                        assert "gradient_norm_decay_every" in hyperparameters.keys(), "missing decay_steps for gradient_norm_min"
                        gradient_norm_min_rate =  tf.train.exponential_decay(hyperparameters["gradient_norm_min_rate"],
                                                                 gradmin_step_ct,
                                                                decay_steps=hyperparameters["gradient_norm_decay_every"],
                                                                decay_rate=hyperparameters["gradient_norm_decay_rate"])
                    else:
                        gradient_norm_min_rate = hyperparameters["gradient_norm_min_rate"]

                    if "gradient_norm_momentum" in hyperparameters.keys():
                        gradient_norm_optimizer = tf.train.MomentumOptimizer(gradient_norm_min_rate,
                                                                             hyperparameters["gradient_norm_momentum"])
                    else:
                        gradient_norm_optimizer = tf.train.GradientDescentOptimizer(gradient_norm_min_rate)

                    step_gradmin = gradient_norm_optimizer.minimize(gradient_norm, global_step = gradmin_step_ct)

                    graph_dictionary["step_gradient_norm_min"] = step_gradmin

    return NeuralNetwork(graph, graph_dictionary, hyperparameters)

# generic functions for adding second order calculations to a graph

def add_step_newton(gradient_descent, gd_grads_and_vars, inverse_hessian, newton_step_ct):
    """using gradient_descent and its grads_and_vars as a base,
    add a newton step that uses inverse_hessian as its (possibly approximate)
    inverted hessian matrix and tracks the number of steps with newton_step_ct
    (for use with decaying rates)
    """
    gd_gradients, gd_variables = gd_grads_and_vars[0]
    gd_gradient_vector = tf.expand_dims(gd_gradients, name="gradient_vector", axis=1)

    newton_gradient_vector = tf.matmul(inverse_hessian, gd_gradient_vector,
                                           name="newton_gradient_vector")
    newton_gradients = tf.squeeze(newton_gradient_vector)

    newton_grads_and_vars = [(newton_gradients, gd_variables)]

    step_newton = gradient_descent.apply_gradients(newton_grads_and_vars, global_step=newton_step_ct)

    return step_newton

def invert_hessian(hessian, num_parameters, hyperparameters):
    """invert a num_parameters by num_parameters hessian using the method described by hyperparameters.
    """
    method = hyperparameters["inverse_method"]

    if method == "fudged":
        assert "fudge_factor" in hyperparameters.keys(), \
            "fudged method requires fudge_factor"
        fudging_vector = tf.constant([hyperparameters["fudge_factor"]]*num_parameters,
                                         dtype=tf.float32, name="fudging_vector")

        fudged_hessian = tf.add(tf.diag(fudging_vector),
                                        hessian, name ="fudged_hessian")

        inverse_hessian = tf.matrix_inverse(fudged_hessian, name="inverse_hessian")

    elif method == "pseudo":
        assert "minimum_eigenvalue_magnitude" in hyperparameters.keys(), \
            "pseudo method requires minimum_eigenvalue_magnitude"
        eigenvalues, eigenvectors = tf.self_adjoint_eig(hessian)

        threshold = tf.constant(hyperparameters["minimum_eigenvalue_magnitude"], shape=[])
        keep = tf.greater(tf.abs(eigenvalues), threshold)

        truncated_eigenvalues = tf.boolean_mask(eigenvalues, keep, name="truncated_eigenvalues")
        # earlier versions of tf don't have axis kwarg, so we tranpose, mask, then transpose back
        truncated_eigenvectors = tf.transpose(tf.boolean_mask(tf.transpose(eigenvectors), keep),
                                             name="truncated_eigenvectors")

        inverted_eigenvalues = tf.divide(1.0, truncated_eigenvalues)

        rescaled_eigenvectors = tf.multiply(tf.expand_dims(inverted_eigenvalues, axis=0), truncated_eigenvectors,
                                            name="rescaled_eigenvectors")

        inverse_hessian = tf.matmul(truncated_eigenvectors, rescaled_eigenvectors,
                                    transpose_b=True, name="inverse_hessian")

    else:
        raise NotImplementedError("no inverse hessian method for {0}".format(method))

    return inverse_hessian

def calculate_num_parameters(hyperparameter_dictionary):
    """calculate the number of parameters in a neural network
    from the information in hyperparameter dictionary
    """
    layer_sizes = hyperparameter_dictionary["layer_sizes"][:]
    input_sizes = hyperparameter_dictionary["input_size"]
    output_size = hyperparameter_dictionary["output_size"]
    layer_sizes = [input_sizes] + layer_sizes + [output_size]

    num_weights = np.sum(np.multiply(layer_sizes[1:],layer_sizes[:-1]))
    num_biases = np.sum(layer_sizes[1:])

    return num_weights+num_biases

def make_weights_and_biases(parameters, hyperparameter_dictionary):
    """return lists of the weight matrices and bias vectors for
    the network described by hyperparameter dictionary
    by slicing and reshaping the entries of parameters
    """
    layer_sizes = hyperparameter_dictionary["layer_sizes"][:]
    input_sizes = hyperparameter_dictionary["input_size"]
    output_size = hyperparameter_dictionary["output_size"]
    layer_sizes = [input_sizes] + layer_sizes + [output_size]

    weight_matrices = make_weights(parameters, layer_sizes)
    bias_start_index = get_bias_start_index(hyperparameter_dictionary)
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

def get_bias_start_index(layer_sizes, hyperparameter_dictonary):
    """determine and return the appropriate starting index for the biases
    inside the parameters vector, and update the hyperparameter_dictionary
    with the total numbers of weights and biases
    """
    bias_shapes = layer_sizes[1:]
    total_biases = np.sum(bias_shapes)
    total_weights = hyperparameter_dictionary["num_parameters"]-total_biases
    hyperparameter_dictionary["total_weights"] = total_weights
    hyperparameter_dictionary["total_biases"] = total_biases
    starting_index = total_weights-total_biases

    return starting_index

def build_by_layer(input, weight_matrices, bias_vectors, hyperparameter_dictionary):
    """build the network described by hyperparameter_dictionary layerwise
    from the layerwise lists of weight_matrices and bias_vectors,
    starting with the input and ending with an "output layer"
    that has no nonlinearity applied.
    """
    current_output = input

    for weight_matrix, bias_vector in zip(weight_matrices[:-1], bias_vectors[:-1]):
        current_output = build_layer(current_output, weight_matrix, bias_vector,
                             hyperparameter_dictionary)

    final_output = build_output_layer(current_output, weight_matrices[-1], bias_vectors[-1],
                                      hyperparameter_dictionary)

    return final_output

def build_layer(current_output, weight_matrix, bias_vector, hyperparameter_dictionary):
    """build a layer that applies affine transformation parametrized by weight_matrix and bias_vector
    to current output, applies nonlinearity, and returns the output
    """
    with tf.variable_scope("internal_layers"):
        nonlinearity = hyperparameter_dictionary["nonlinearity"]
        new_output = nonlinearity(tf.add(tf.matmul(current_output, weight_matrix), bias_vector))
    return new_output

def build_output_layer(current_output, weight_matrix, bias_vector, hyperparameter_dictionary):
    """build a layer that applies the affine transformation parametrized by weight_matrix and bias_vector
    to current output and returns the output
    """
    with tf.variable_scope("output_layer"):
        final_output = tf.add(tf.matmul(current_output, weight_matrix), bias_vector)
    return final_output

# convenience functions for interacting with quadratic form graphs

def generate_initial_values(N, scaling_factor=None):
    """generate a vector of scaled random normal values.
    scaling_factor defaults to 1/sqrt(N).
    """
    if scaling_factor is None:
        scaling_factor = 1/np.sqrt(N)

    return scaling_factor*np.random.standard_normal(size=N).astype(np.float32)

def minimize(quadratic_form, algorithm, num_steps):
    """use algorithm to minimize (aka adjust the parameters of)
    quadratic form and return the resulting outputs and values
    """
    graph, graph_dictionary = quadratic_form

    with graph.as_default():
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            graph_dictionary["inputs"].initializer.run()
            for _ in range(num_steps):
                sess.run(graph_dictionary["step_"+algorithm])
            output = sess.run(graph_dictionary["output"])
            values = graph_dictionary["inputs"].eval()

    return output, values

def get_result(result_key, input_vector, quadratic_form_graph):
    """return the result of applying quadratic_form_graph.graph_dictionary["result_key"] on input_vector
    """

    graph, graph_dictionary = quadratic_form_graph
    result_op = graph_dictionary[result_key]
    input_placeholders = graph_dictionary["inputs"]

    input_feed_dict = make_feed_dict(input_placeholders, input_vector)

    result = run(graph, result_op, input_feed_dict)

    return result

def run(graph, op, input_feed_dict):
    """run a given op on a given graph with feed_dict input_feed_dict
    """

    with graph.as_default():
        with tf.Session() as sess:
            result = sess.run(op, feed_dict = input_feed_dict)

    return result

def make_feed_dict(input_placeholders, input_vector):
    """make a feed_dict for a quadratic_form_graph
    """

    feed_dict = {input_placeholders:input_vector}

    return feed_dict

# functions for generating various random matrices

def generate_gaussian(N):
    """generate an N by N gaussian random matrix with variance N
    """
    return 1/np.sqrt(N)*np.random.standard_normal(size=(N,N)).astype(np.float32)

def generate_symmetric(N):
    """generate an N by N symmetric gaussian random matrix with variance N
    """
    base_matrix = generate_gaussian(N)
    return (1/np.sqrt(2))*(base_matrix+base_matrix.T)

def generate_wishart(N, k=1):
    """generate an N by N wishart random matrix with rank min(N,k)
    """
    self_outer_product = lambda x: x.dot(x.T)
    wishart_random_matrix = 1/k*self_outer_product(np.random.standard_normal(size=(N,k))).astype(np.float32)

    return wishart_random_matrix

def generate_negative_wishart(N, k=1):
    """generate an N by N negative wishart random matric with rank min(N,k)
    """
    wishart_random_matrix = generate_wishart(N, k)
    negative_wishart_random_matrix = -1*wishart_random_matrix

    return negative_wishart_random_matrix
