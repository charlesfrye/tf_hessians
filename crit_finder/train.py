import tensorflow as tf
import numpy as np
from collections import namedtuple

Results = namedtuple("Results", ["cost", "gradient_norm", "scalar_index",
                                 "gradients", "parameters", "vector_index"])

def train_and_track(network, data, crit_finder_str, num_steps_gd=5000, 
                   gradient_descent_batch_size=50, gradient_descent_track_every=50,
                    tracking_batch_size=50000, print_tracking_data=False,
                    num_steps_crit_finder=10, crit_finder_batch_size=10000, crit_finder_track_every=1):
    
    graph = network.graph
    graph_dict = network.graph_dictionary
    hyperparameter_dictionary = network.hyperparameter_dictionary
    num_parameters = hyperparameter_dictionary["num_parameters"]
    total_weights = hyperparameter_dictionary["total_weights"]
    total_biases = hyperparameter_dictionary["total_biases"]
    initialized_parameters = np.hstack([0.1*np.random.standard_normal(size=total_weights),
                                      [0.1]*total_biases]).astype(np.float32)

    with tf.Session(graph=graph) as sess:
        
        input = graph_dict["input"]
        labels = graph_dict["labels"]
        initial_parameters = graph_dict["parameters_placeholder"]
        trained_parameters = graph_dict["parameters"]

        step_gradient_descent = graph_dict["step_gradient_descent"]
        step_crit_finder = graph_dict[crit_finder_str]

        accuracy = graph_dict["accuracy"]
        cost = graph_dict["cost"]
        gradient_op = graph_dict["gradients"]

        initializer_feed_dict = {initial_parameters: initialized_parameters}
        tf.global_variables_initializer().run(initializer_feed_dict)    
        
        gd_track_string = "gd step"
        gd_results = run_optimizer(sess, network, data["train"], accuracy, input, labels, trained_parameters, cost,
                                   gradient_op, step_gradient_descent, num_steps_gd, gradient_descent_batch_size,
                                   gradient_descent_track_every, tracking_batch_size, print_tracking_data, gd_track_string)
            
        crit_finder_track_string = "crit_finder step"
        crit_finder_results = run_optimizer(sess, network, data["train"], accuracy, input, labels, trained_parameters,
                                            cost, gradient_op, step_crit_finder, num_steps_crit_finder,
                                            crit_finder_batch_size, crit_finder_track_every,
                                            tracking_batch_size, print_tracking_data, crit_finder_track_string)

    return gd_results, crit_finder_results

def run_optimizer(sess, network, data, accuracy, input, labels, trained_parameters, cost, gradient_op, step_optimizer, 
                  num_steps, batch_size, track_every, tracking_batch_size, print_tracking_data, track_string):
    
    results = Results([],[],[],[],[],[])
    
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
    
    if hasattr(data, "next_batch"):
        batch_inputs, batch_labels = data.next_batch(batch_size)
    else:
        num_elements = data["labels"].shape[0]
        indices = np.random.choice(num_elements, size=batch_size, replace=False)
        batch_inputs = data["images"][indices,:]
        batch_labels = data["labels"][indices]
    
    return batch_inputs, batch_labels

def add_to_results_scalars(results, cost, gradient_norm, index):
    results.cost.append(cost)
    results.gradient_norm.append(gradient_norm)
    results.scalar_index.append(index)
    
def add_to_results_vectors(results, gradients, parameters, index):
    results.parameters.append(parameters)
    results.gradients.append(gradients)
    results.vector_index.append(index)
    
def print_tracking(string, cost, gradient_norm, gradient_max):
    print(string)
    print("\tcost: {0:.2f}".format(cost))
    print("\tgrad_norm: {0:.10f}".format(gradient_norm))
    print("\tgrad_max: {0:.10f}".format(gradient_max))