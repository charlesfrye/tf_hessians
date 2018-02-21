"""train networks produced by crit_finder.graphs and track the Results in a NamedTuple."""
import tensorflow as tf
import numpy as np
from collections import namedtuple

Results = namedtuple("Results", ["cost", "gradient_norm", "scalar_index",
                                 "gradient", "parameters", "vector_index"])

TrainAndTrackParams = namedtuple("TrainAndTrackParams", ["num_steps", "batch_size",
                                                        "track_every", "tracking_batch_size",
                                                        "print_tracking_data", "track_string"])

def train_and_track(network, data, optimizer_str, crit_finder_str, optimizer_train_and_track_params, crit_finder_train_and_track_params):
    """train network on data using optimizer named by optimizer_str,
    then search for critical points with crit_finder named by crit_finder_str,
    using training and tracking parameters given by their respective train_and_track_params.

    note that, for both phases, gradient norm and cost are tracked on each track_every,
    while gradients and parameters are tracked only at the beginning and end.
    """

    graph = network.graph
    graph_dict = network.graph_dictionary
    hyperparameters = network.hyperparameters

    num_parameters = hyperparameters["num_parameters"]
    total_weights = hyperparameters["total_weights"]
    total_biases = hyperparameters["total_biases"]
    initialized_parameters = np.hstack([0.1*np.random.standard_normal(size=total_weights),
                                      [0.1]*total_biases]).astype(np.float32)

    with tf.Session(graph=graph) as sess:

        initial_parameters = graph_dict["parameters_placeholder"]

        step_optimizer = graph_dict[optimizer_str]
        step_crit_finder = graph_dict[crit_finder_str]

        initializer_feed_dict = {initial_parameters: initialized_parameters}
        tf.global_variables_initializer().run(initializer_feed_dict)

        optimizer_results = run_algorithm(sess, network, data, graph_dict, step_optimizer, *optimizer_train_and_track_params)

        crit_finder_results = run_algorithm(sess, network, data, step_crit_finder, *crit_finder_train_and_track_params)

    return gd_results, crit_finder_results

def run_algorithm(sess, data, graph_dict, step_algorithm, num_steps, batch_size, track_every,
                    tracking_batch_size, print_tracking_data, track_string):
    """ in the context of sess, call step_algorithm num_steps times, with each step
    drawing a batch of size batch_size from data.

    cost and gradient norm (scalar quantities) are stored after each track_every step,
    and before taking any steps,
    while the gradient and parameters (vector quantities) are stored only before taking any steps
    and on the last tracked step.
    """
    results = Results([],[],[],[],[],[])

    input = graph_dict["input"]
    labels = graph_dict["labels"]

    trained_parameters = graph_dict["parameters"]
    accuracy = graph_dict["accuracy"]
    cost = graph_dict["cost"]
    gradient_op = graph_dict["gradients"]

    for batch_idx in range(num_steps):

        if (batch_idx+1 == 1):

            batch_inputs, batch_labels = get_batch(data, tracking_batch_size)
            track_feed_dict = {input: batch_inputs,
                       labels: batch_labels}

            current_cost, gradients, parameters = sess.run([cost, gradient_op, trained_parameters],
                                                           feed_dict=track_feed_dict)
            gradient_norm = np.linalg.norm(gradients)
            gradient_max = np.max(np.abs(gradients))

            if print_tracking_data:
                print_tracking("init_values", current_cost, gradient_norm, gradient_max)

            add_to_results_scalars(results, current_cost, gradient_norm, batch_idx)
            add_to_results_vectors(results, gradients[0], parameters, batch_idx)

        batch_inputs, batch_labels = get_batch(data, batch_size)
        train_feed_dict = {input: batch_inputs,
                       labels: batch_labels}

        sess.run(step_optimizer, feed_dict=train_feed_dict)

        if (batch_idx+1 == 1) or ((batch_idx+1)%track_every == 0):

            batch_inputs, batch_labels = get_batch(data, tracking_batch_size)
            train_feed_dict = {input: batch_inputs,
                       labels: batch_labels}

            current_cost, gradients, parameters = sess.run([cost, gradient_op, trained_parameters],
                                                           feed_dict=train_feed_dict)
            gradient_norm = np.linalg.norm(gradients)
            gradient_max = np.max(np.abs(gradients))

            if print_tracking_data:
                print_tracking(track_string+" {0}:".format(batch_idx+1), current_cost, gradient_norm, gradient_max)

            add_to_results_scalars(results, current_cost, gradient_norm, batch_idx+1)

            if (batch_idx+1)+track_every > num_steps:
                # on last tracked iteration
                add_to_results_vectors(results, gradients[0], parameters, batch_idx+1)

    return results

def get_batch(data, batch_size):
    """draw, without replacement, a batch of size batch_size from data
    """

    if hasattr(data, "next_batch"):
        batch_inputs, batch_labels = data.next_batch(batch_size)
    else:
        num_elements = data["labels"].shape[0]
        indices = np.random.choice(num_elements, size=batch_size, replace=False)
        batch_inputs = data["images"][indices,:]
        batch_labels = data["labels"][indices]

    return batch_inputs, batch_labels

def add_to_results_scalars(results, cost, gradient_norm, index):
    """add scalar quantities (cost, gradient_norm)
    to results and add the index to scalar_index
    """
    results.cost.append(cost)
    results.gradient_norm.append(gradient_norm)
    results.scalar_index.append(index)

def add_to_results_vectors(results, gradient, parameters, index):
    """add vector quantities (gradient, parameters)
    to results and add the index to vector_index
    """
    results.parameters.append(parameters)
    results.gradient.append(gradient)
    results.vector_index.append(index)

def print_tracking(string, cost, gradient_norm, gradient_max):
    """print string, then tab and print cost, gradient_norm, and gradient_max
    """
    print(string)
    print("\tcost: {0:.2f}".format(cost))
    print("\tgrad_norm: {0:.10f}".format(gradient_norm))
    print("\tgrad_max: {0:.10f}".format(gradient_max))
