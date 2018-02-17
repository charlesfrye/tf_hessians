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
                gradient_norm_optimizer = tf.train.MomentumOptimizer(gradient_norm_min_rate, gradient_norm_momentum)
            else:
                gradient_norm_optimizer = tf.train.GradientDescentOptimizer(gradient_norm_min_rate)
                
            step_gradmin = gradient_norm_optimizer.minimize(gradient_norm, global_step = gradmin_step_ct)
                
            graph_dictionary["step_gradient_norm_min"] = step_gradmin

    return QuadraticForm(graph, graph_dictionary)

# neural network construction functions

def make_neural_network(hyperparameter_dictionary):
    
    graph = tf.Graph()
    
    with graph.as_default():
        
        input_size = hyperparameter_dictionary["input_size"]
        output_size = hyperparameter_dictionary["output_size"]
        
        num_parameters = calculate_num_parameters(hyperparameter_dictionary)
        hyperparameter_dictionary["num_parameters"] = num_parameters
        parameters_placeholder = tf.placeholder(tf.float32, shape=[num_parameters],
                                                name="initial_parameters")
        parameters_var = tf.Variable(initial_value=parameters_placeholder,
                                         name="parameters_variable",
                                    expected_shape=[num_parameters])

        weight_matrices, bias_vectors = make_weights_and_biases(parameters_var,
                                                                hyperparameter_dictionary)
        
        input = tf.placeholder(tf.float32, shape=[None, input_size])
        
        network_output = build_by_layer(input, weight_matrices, bias_vectors,
                                      hyperparameter_dictionary)
        
        labels = tf.placeholder(tf.float32, shape=[None, output_size])
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=network_output,
                                                                     labels=labels))
        
        network_predictions = tf.nn.softmax(network_output, name="network_predictions")
        prediction_correct = tf.equal(tf.argmax(network_predictions,1), tf.argmax(labels,1))
        accuracy = tf.reduce_mean(tf.cast(prediction_correct, tf.float32))
        
        with tf.variable_scope("grads_and_hess"):

            gradients = tf.gradients(cost, parameters_var, name="gradients")

            hessian_matrix = tf.hessians(cost, parameters_var, name="hessians_output")[0]
            
            eigenvalues, eigenvectors = tf.self_adjoint_eig(hessian_matrix)
            
            inverse_hessian = invert_hessian(hessian_matrix, num_parameters,
                                            hyperparameter_dictionary)

            gradient_descent = tf.train.GradientDescentOptimizer(hyperparameter_dictionary["learning_rate"])
            step_gradient_descent = gradient_descent.minimize(cost)
            
            newton_step_ct = tf.Variable(0, trainable=False)
            newton_rate = tf.train.exponential_decay(hyperparameter_dictionary["newton_rate"],
                                                     newton_step_ct,
                                                    decay_steps=1,
                                                    decay_rate=0.1)
                
            newton_base = tf.train.GradientDescentOptimizer(newton_rate)
            gd_grads_and_vars = newton_base.compute_gradients(cost, parameters_var)
            step_newton = add_step_newton(newton_base, gd_grads_and_vars, inverse_hessian, newton_step_ct)
           
            gradmin_step_ct = tf.Variable(0, trainable=False)
            gradient_norm = tf.square(tf.norm(gradients, name="gradient_norm"))
            
            gradient_norm_descent_rate =  tf.train.exponential_decay(hyperparameter_dictionary["grad_norm_learning_rate"],
                                                     newton_step_ct,
                                                    decay_steps=100,
                                                    decay_rate=1.0)
            
            gradient_norm_optimizer = tf.train.GradientDescentOptimizer(gradient_norm_descent_rate)
#             gradient_norm_optimizer = tf.train.MomentumOptimizer(gradient_norm_descent_rate, 0.9)
            step_gradmin = gradient_norm_optimizer.minimize(gradient_norm, global_step = gradmin_step_ct)

        graph_dictionary = {"parameters_placeholder": parameters_placeholder,
                            "parameters": parameters_var,
                            "input": input,
                            "weight_matrices": weight_matrices,
                            "bias_vectors": bias_vectors,
                            "labels": labels,
                            "cost": cost,
                            "accuracy": accuracy,
                            "gradients": gradients,
                            "hessian": hessian_matrix,
                            "eigenvalues": eigenvalues,
                            "eigenvectors": eigenvectors,
                            "step_gradient_descent": step_gradient_descent,
                            "step_newton": step_newton,
                            "step_gradmin": step_gradmin,
                           }
    
    return NeuralNetwork(graph, graph_dictionary, hyperparameter_dictionary)

def calculate_num_parameters(hyperparameter_dictionary):
    layer_sizes = hyperparameter_dictionary["layer_sizes"][:]
    input_sizes = hyperparameter_dictionary["input_size"]
    output_size = hyperparameter_dictionary["output_size"]
    layer_sizes = [input_sizes] + layer_sizes + [output_size]
    
    num_weights = np.sum(np.multiply(layer_sizes[1:],layer_sizes[:-1]))
    num_biases = np.sum(layer_sizes[1:])
    
    return num_weights+num_biases

def make_weights_and_biases(parameters, hyperparameter_dictionary):
    layer_sizes = hyperparameter_dictionary["layer_sizes"][:]
    input_sizes = hyperparameter_dictionary["input_size"]
    output_size = hyperparameter_dictionary["output_size"]
    layer_sizes = [input_sizes] + layer_sizes + [output_size]
    
    weight_matrices = make_weights(parameters, layer_sizes, hyperparameter_dictionary)
    bias_vectors = make_biases(parameters, layer_sizes, hyperparameter_dictionary)
    
    return weight_matrices, bias_vectors

def make_weights(parameters, layer_sizes, hyperparameter_dictionary):
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

def make_biases(parameters, layer_sizes, hyperparameter_dictionary):
    bias_shapes = layer_sizes[1:]
    total_biases = np.sum(bias_shapes)
    total_weights = hyperparameter_dictionary["num_parameters"]-total_biases
    hyperparameter_dictionary["total_weights"] = total_weights
    hyperparameter_dictionary["total_biases"] = total_biases
    starting_index = total_weights-total_biases
    bias_vectors = []
    
    with tf.variable_scope("biases"):
        
        for bias_shape in bias_shapes:
            num_biases = bias_shape
            
            bias_vector = tf.slice(parameters, [starting_index], [num_biases],
                                     name="sliced")

            bias_vectors.append(bias_vector)
            
            starting_index += num_biases
            
    return bias_vectors

def build_by_layer(input, weight_matrices, bias_vectors, hyperparameter_dictionary):
    current_output = input
    
    for weight_matrix, bias_vector in zip(weight_matrices[:-1], bias_vectors[:-1]):
        current_output = build_layer(current_output, weight_matrix, bias_vector,
                             hyperparameter_dictionary)
        
    final_output = build_output_layer(current_output, weight_matrices[-1], bias_vectors[-1],
                                      hyperparameter_dictionary)
    
    return final_output

def build_layer(current_output, weight_matrix, bias_vector, hyperparameter_dictionary):
    with tf.variable_scope("internal_layers"):
        nonlinearity = hyperparameter_dictionary["nonlinearity"]
        new_output = nonlinearity(tf.add(tf.matmul(current_output, weight_matrix), bias_vector))
    return new_output

def build_output_layer(current_output, weight_matrix, bias_vector, hyperparameter_dictionary):
    with tf.variable_scope("output_layer"):
        final_output = tf.add(tf.matmul(current_output, weight_matrix), bias_vector)
    return final_output

# convenience functions for interacting with the graph

def minimize(quadratic_form, algorithm, num_steps):
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
    
    graph, graph_dictionary = quadratic_form_graph
    result_op = graph_dictionary[result_key]
    input_placeholders = graph_dictionary["inputs"]
    
    input_feed_dict = make_feed_dict(input_placeholders, input_vector)
    
    result = run(graph, result_op, input_feed_dict)
    
    return result

def run(graph, op, input_feed_dict):
    
    with graph.as_default():
        with tf.Session() as sess:
            result = sess.run(op, feed_dict = input_feed_dict)
            
    return result

def make_feed_dict(input_placeholders, input_vector):
    
    feed_dict = {input_placeholders:input_vector}
    
    return feed_dict

# generic functions for adding second order calculations to a graph

# def add_step_newton(gradient_descent, gd_grads_and_vars, inverse_hessian):
#     gd_gradients, gd_variables = gd_grads_and_vars[0]
#     gd_gradient_vector = tf.expand_dims(gd_gradients, name="gradient_vector", axis=1)

#     newton_gradient_vector = tf.matmul(inverse_hessian, gd_gradient_vector,
#                                            name="newton_gradient_vector")
#     newton_gradients = tf.squeeze(newton_gradient_vector)
      
#     newton_grads_and_vars = [(newton_gradients, gd_variables)]

#     step_newton = gradient_descent.apply_gradients(newton_grads_and_vars)
    
#     return step_newton

def add_step_newton(gradient_descent, gd_grads_and_vars, inverse_hessian, newton_step_ct):
    gd_gradients, gd_variables = gd_grads_and_vars[0]
    gd_gradient_vector = tf.expand_dims(gd_gradients, name="gradient_vector", axis=1)

    newton_gradient_vector = tf.matmul(inverse_hessian, gd_gradient_vector,
                                           name="newton_gradient_vector")
    newton_gradients = tf.squeeze(newton_gradient_vector)
      
    newton_grads_and_vars = [(newton_gradients, gd_variables)]

    step_newton = gradient_descent.apply_gradients(newton_grads_and_vars, global_step=newton_step_ct)
    
    return step_newton

def invert_hessian(hessian, num_parameters, hyperparameters):
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

# functions for testing the performance of descent methods on quadratic forms

def compare_descent_methods(matrix_generator, N=5, num_steps=5, num_matrices=10, num_runs=5,
                            methods=["gradient_descent","newton"],
                           hyperparameters={"learning_rate":0.1, "newton_rate":1,
                                            "fudge_factor":1e-6, "inverse_method":"fudged"}):
    
    num_methods = len(methods)
    matrices = [matrix_generator(N) for _ in range(num_matrices)]

    results = np.zeros((num_steps+1, num_matrices, num_runs, num_methods))

    for matrix_idx, matrix in enumerate(matrices):

        for run_idx in range(num_runs):

            initial_values = np.random.standard_normal(size=N).astype(np.float32)

            wishart_quadratic_form = make_quadratic_form(matrix, initial_values, hyperparameters)
            for step_idx in range(num_steps+1):
                for method_idx, method in enumerate(methods):
                    _, values =  minimize(wishart_quadratic_form, method, step_idx)        
                    results[step_idx, matrix_idx, run_idx, method_idx] = get_result("gradient_norm", 
                                                                           values, wishart_quadratic_form)

    return results

def gradient_test(N, matrix_generator, algorithm, num_steps, hyperparameters=DEFAULTS):

    random_matrix = matrix_generator(N)

    initial_values = np.random.standard_normal(size=N).astype(np.float32)

    quadratic_form = make_quadratic_form(random_matrix, initial_values, hyperparameters)

    initial_output = get_result("output", initial_values, quadratic_form)
    initial_gradients = get_result("gradients", initial_values, quadratic_form)[0]

    final_output, final_values = minimize(quadratic_form, algorithm, num_steps)
    final_gradients = get_result("gradients", final_values, quadratic_form)[0]

    print("output:\n" +
          "\tinitial: {0}".format(initial_output),
          "\tfinal: {0}".format(final_output))

    print("gradient norm:\n" +
          "\tinitial: {0}".format(np.linalg.norm(initial_gradients, 2)),
          "\tfinal: {0}".format(np.linalg.norm(final_gradients, 2)))

def normalize_runs(results):
    results = np.divide(results, results[0,:,:,:][None,:,:,:])
    return results

def plot_trajectories_comparison(results, methods=["gradient_descent","newton"],
                                 colors=["salmon","darkolivegreen","medium_blue"]):
    f = plt.figure(figsize=(16,4))
    ax = plt.subplot(111)
    for matrix_idx in range(results.shape[1]):
        for method_idx in range(results.shape[3]):
            color = colors[method_idx]
            method = methods[method_idx]
            plt.plot(results[:,matrix_idx,:,method_idx], color=color, linewidth=1, alpha=0.5, label=method);
        
    plt.legend()
    ax.set_ylabel(r"Normalized $\|\nabla f\|^2$",fontsize=16, fontweight="bold")
    ax.set_xlabel("Iteration Number",fontsize=16, fontweight="bold")
    ax.set_ylim([0,1.1])
    ax.set_yticks([0,1]);
    ax.set_xticks(range(0,1+int(max(ax.get_xlim()))));

def plot_benefit(results, methods=["gradient_descent","newton"]):
    f = plt.figure(figsize=(16,8))
    ax = plt.subplot(111)
    for matrix_idx in range(results.shape[1]):
        benefits = results[:,matrix_idx,:,1] - results[:,matrix_idx,:,0]
        plt.plot(benefits, linestyle="None",
                 marker='o', color='grey', markersize=24, alpha=0.05);
    xlim = ax.get_xlim()    
    plt.hlines(0,*xlim, linewidth=4)
    ax.set_xlim(xlim)
    
    ax.set_ylabel(r"$\leftarrow$"+ "{0} better".format(methods[1]) + 
                  "\t\t\t"+ "{0} better".format(methods[0]) + r"$\rightarrow$",
                  fontsize=16, fontweight="bold")
    ax.set_xlabel("Iteration Number",fontsize=16, fontweight="bold")
    ax.set_xticks(range(0,1+int(max(ax.get_xlim()))));