from collections import namedtuple
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

QuadraticForm = namedtuple("QuadraticForm", ["graph", "graph_dictionary"])

DEFAULTS = {"learning_rate":0.1, # courtesy of Moritz Hardt and Ilya Sutskever
            "newton_rate":1,
            "fudge_factor":0.0,
            "inverse_method":"fudged",}

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
    
        output = tf.squeeze(tf.matmul(input_vector,
                                  tf.matmul(quadratic_form, input_vector),
                                      transpose_a=True,
                                  name='output'),
                        name='squeezed_output')
        
        gradients = tf.gradients(output, inputs, name="gradients")
        
        hessian_matrix = tf.hessians(output, inputs, name="hessian_matrix")[0]
        
        inverse_hessian = invert_hessian(hessian_matrix, len(initial_values), hyperparameters)
        
        gradient_descent = tf.train.GradientDescentOptimizer(hyperparameters["learning_rate"])
        step_gradient_descent = gradient_descent.minimize(output)
        
        newton_base = tf.train.GradientDescentOptimizer(hyperparameters["newton_rate"])
        gd_grads_and_vars = newton_base.compute_gradients(output, inputs)
        step_newton = add_step_newton(newton_base, gd_grads_and_vars, inverse_hessian)
        
        graph_dictionary = {"inputs": inputs,
                           "output": output,
                           "gradients": gradients,
                           "hessian": hessian_matrix,
                            "step_gradient_descent": step_gradient_descent,
                            "step_newton": step_newton
                           }
    
    return QuadraticForm(graph, graph_dictionary)

# convenience functions for accessing QuadraticForm

def minimize(quadratic_form, algorithm, num_steps):
    graph, graph_dictionary = quadratic_form
    
    with graph.as_default():
        with tf.Session() as sess:
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

def add_step_newton(gradient_descent, gd_grads_and_vars, inverse_hessian):
    gd_gradients, gd_variables = gd_grads_and_vars[0]
    gd_gradient_vector = tf.expand_dims(gd_gradients, name="gradient_vector", axis=1)

    newton_gradient_vector = tf.matmul(inverse_hessian, gd_gradient_vector,
                                           name="newton_gradient_vector")
    newton_gradients = tf.squeeze(newton_gradient_vector)
      
    newton_grads_and_vars = [(newton_gradients, gd_variables)]

    step_newton = gradient_descent.apply_gradients(newton_grads_and_vars)
    
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
        eigenvalues, eigenvectors = tf.self_adjoint_eig(tf.expand_dims(hessian, axis=0))
        
        threshold = hyperparameters["minimum_eigenvalue_magnitude"]
        keep = tf.reduce_sum(tf.cast(tf.greater_equal(eigenvalues, threshold), tf.int32))
        
        truncated_eigenvalues = tf.squeeze(eigenvalues)[-keep:]
        truncated_eigenvectors = tf.squeeze(eigenvectors)[:, -keep:]

        inverse_hessian = tf.matmul(truncated_eigenvectors,
                                    (1. / tf.expand_dims(truncated_eigenvalues, axis=0)) * truncated_eigenvectors,
                                    transpose_b=True, name="inverse_hessian")
        
    else:
        raise NotImplementedError("no inverse hessian method for {0}".format(method))
        
    return inverse_hessian

def compare_descent_methods(matrix_generator, N=5, num_steps=5, num_matrices=10, num_runs=5,
                           hyperparameters={"learning_rate":0.1, "newton_rate":1,
                                            "fudge_factor":1e-6, "inverse_method":"fudged"}):
    
    matrices = [matrix_generator(N) for _ in range(num_matrices)]

    results = np.zeros((num_steps+1, num_matrices, num_runs, 2))

    for matrix_idx, matrix in enumerate(matrices):

        for run_idx in range(num_runs):

            initial_values = np.random.standard_normal(size=N).astype(np.float32)

            wishart_quadratic_form = make_quadratic_form(matrix, initial_values, hyperparameters)
            for step_idx in range(num_steps+1):
                results[step_idx, matrix_idx, run_idx, 0] = minimize(wishart_quadratic_form,
                                                                                  "newton", step_idx)[0]
                results[step_idx, matrix_idx, run_idx, 1] = minimize(wishart_quadratic_form,
                                                                       "gradient_descent", step_idx)[0]
                
    return results

def normalize_runs(results):
    results = np.divide(results, results[0,:,:,:][None,:,:,:])
    return results

def plot_trajectories_comparison(results):
    f = plt.figure(figsize=(16,4))
    ax = plt.subplot(111)
    for matrix_idx in range(results.shape[1]):
        plt.plot(results[:,matrix_idx,:,0], color='salmon', linewidth=1, alpha=0.5, label='newton');
        plt.plot(results[:,matrix_idx,:,1], color='darkolivegreen', linewidth=1, alpha=0.5, label='grad');
        
    plt.legend()
    ax.set_ylabel("Normalized Cost",fontsize=16, fontweight="bold")
    ax.set_xlabel("Iteration Number",fontsize=16, fontweight="bold")
    ax.set_ylim([0,1.1])
    ax.set_yticks([0,1]);
    ax.set_xticks(range(0,1+int(max(ax.get_xlim()))));

def plot_benefit(results):
    f = plt.figure(figsize=(16,8))
    ax = plt.subplot(111)
    for matrix_idx in range(results.shape[1]):
        benefits = results[:,matrix_idx,:,1] - results[:,matrix_idx,:,0]
        plt.plot(benefits, linestyle="None",
                 marker='o', color='grey', markersize=24, alpha=0.05);
    xlim = ax.get_xlim()    
    plt.hlines(0,*xlim, linewidth=4)
    ax.set_xlim(xlim)
    
    ax.set_ylabel(r"$\leftarrow$ GD better"+"\t\t\t\t"+r"NM Better $\rightarrow$",
                  fontsize=16, fontweight="bold")
    ax.set_xlabel("Iteration Number",fontsize=16, fontweight="bold")
    ax.set_xticks(range(0,1+int(max(ax.get_xlim()))));