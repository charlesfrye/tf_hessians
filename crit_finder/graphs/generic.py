"""functions for adding optimizers, crit_finders, and other ops to generic graphs
"""
import tensorflow as tf

def add_gradient_ops(function, inputs, graph_dictionary):
    """adds ops to calculate gradients and scaled squared gradient norm to graph and graph_dictionary
    to graph, calculates squared gradient norm, and adds these operations
    to the graph_dictionary
    """
    with tf.variable_scope("gradients"):

        gradients = tf.gradients(function, inputs, name="gradients")
        gradient_norm = tf.norm(gradients, name="gradient_norm")
        scaled_squared_gradient_norm = 0.5*tf.square(gradient_norm, name="scaled_squared_gradient_norm"))

    graph_dictionary.update({
                           "gradients": gradients,
                           "gradient_norm": gradient_norm
                           "scaled_squared_gradient_norm": scaled_squared_gradient_norm
                           })

def add_hess_ops(function, inputs, graph_dictionary):
    """adds ops to calculate and diagonalize the hessian to the graph and graph_dictionary
    """
    with tf.variable_scope("hessian"):
        hessian_matrix = tf.hessians(function, inputs, name="hessians_output")[0]
        eigenvalues, eigenvectors = tf.self_adjoint_eig(hessian_matrix)

    graph_dictionary.update({
                           "hessian_matrix": hessian_matrix,
                           "eigenvalues": eigenvalues,
                           "eigenvectors": eigenvectors
                           })

def add_optimizer(function, inputs, hyperparameters, graph_dictionary):
    """adds ops to optimize function according to hyperparameters to the graph and graph_dictionary
    """
    with tf.variable_scope("optimizer"):
        optimizer_step_ct = tf.Variable(0, trainable=False)
        optimizer_rate = hyperparameters["learning_rate"]

        if "optimizer_decay_rate" in hyperparameters.keys():
            assert "optimizer_decay_every" in hyperparameters.keys(), "missing decay_steps for gradient_descent"
            optimizer_rate =  tf.train.exponential_decay(optimizer_rate,
                                                        optimizer_step_ct,
                                                        decay_steps=hyperparameters["gradient_descent_decay_every"],
                                                        decay_rate=hyperparameters["gradient_descent_decay_rate"])

        if "momentum_rate" in hyperparameters.keys():
            optimizer = tf.train.MomentumOptimizer(optimizer_rate, hyperparameters["momentum_rate"])
            step_optimizer = optimizer.minimize(function)
            graph_dictionary["step_momentum"] = step_optimizer
        else:
            optimizer = tf.train.GradientDescentOptimizer(optimizer_rate)
            step_optimizer = optimizer.minimize(function)
            graph_dictionary["step_gradient_descent"] = step_optimizer

def add_crit_finder(function, inputs, input_size, hyperparameters, graph_dictionary):
    """adds ops to find critical points of function according to hyperparameters to the graph and graph_dictionary
    """

    hessian_matrix = graph_dictionary["hessian_matrix"]
    scaled_squared_gradient_norm = graph_dictionary["scaled_squared_gradient_norm"]

    with tf.variable_scope("crit_finder"):

        with tf.variable_scope("newton"):
            if "inverse_method" in hyperparameters.keys():
                inverse_hessian = invert_hessian(hessian_matrix, input_size, hyperparameters)
                graph_dictionary["inverse_hessian"] = inverse_hessian

                if "newton_rate" in hyperparameters.keys():
                    # add a newton step by processing the gradients of a GradientDescentOptimizer
                    newton_step_ct = tf.Variable(0, trainable=False)
                    newton_base = tf.train.GradientDescentOptimizer(hyperparameters["newton_rate"])
                    gd_grads_and_vars = newton_base.compute_gradients(function, inputs)

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

                    step_gradmin = gradient_norm_optimizer.minimize(scaled_squared_gradient_norm, global_step = gradmin_step_ct)

                    graph_dictionary["step_gradient_norm_min"] = step_gradmin

## generic functions for adding second order calculations to a graph

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
