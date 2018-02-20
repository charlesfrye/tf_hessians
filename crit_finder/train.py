import tensorflow as tf
import numpy as np
from collections import namedtuple

Results = namedtuple("Results", ["cost", "gradient_norm", "scalar_index",
                                 "gradients", "parameters", "vector_index"])

def train_and_track(network, mnist, crit_finder_str, num_steps_gd=5000, 
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
        
        # run_optimizer gd
        
        gd_results = Results([],[],[],[],[],[])
        
        for batch_idx in range(num_steps_gd):

            if (batch_idx+1 == 1):

                batch_inputs, batch_labels = mnist.train.next_batch(tracking_batch_size)
                train_feed_dict = {input: batch_inputs,
                           labels: batch_labels}

                current_cost = sess.run(cost, feed_dict=train_feed_dict)
                gradients = sess.run(gradient_op, feed_dict=train_feed_dict)
                gradient_norm = np.linalg.norm(gradients)
                gradient_max = np.max(np.abs(gradients))
                parameters = sess.run(trained_parameters)
                
                if print_tracking_data:
                    print_tracking("init_values", current_cost, gradient_norm, gradient_max)
                
                add_to_results_scalars(gd_results, current_cost, gradient_norm, batch_idx)
                add_to_results_vectors(gd_results, gradients[0], parameters, batch_idx)

            batch_inputs, batch_labels = mnist.train.next_batch(gradient_descent_batch_size)
            train_feed_dict = {input: batch_inputs,
                           labels: batch_labels}

            sess.run(step_gradient_descent, feed_dict=train_feed_dict)

            if (batch_idx+1 == 1) or ((batch_idx+1)%gradient_descent_track_every == 0):

                batch_inputs, batch_labels = mnist.train.next_batch(tracking_batch_size)
                train_feed_dict = {input: batch_inputs,
                           labels: batch_labels}

                current_cost = sess.run(cost, feed_dict=train_feed_dict)
                gradients = sess.run(gradient_op, feed_dict=train_feed_dict)
                gradient_norm = np.linalg.norm(gradients)
                gradient_max = np.max(np.abs(gradients))
                
                if print_tracking_data:
                    print_tracking("grad step {0}:".format(batch_idx+1), current_cost, gradient_norm, gradient_max)
                    
                add_to_results_scalars(gd_results, current_cost, gradient_norm, batch_idx+1)
                
                if (batch_idx+1)+gradient_descent_track_every > num_steps_gd:
                    # on last tracked iteration
                    parameters = sess.run(trained_parameters)
                    add_to_results_vectors(gd_results, gradients[0], parameters, batch_idx+1)
            
        # run optimizer crit_finder
        
        crit_finder_results = Results([],[],[],[],[],[])
        
        for batch_idx in range(num_steps_crit_finder):
            
            if (batch_idx+1 == 1):
                
                batch_inputs, batch_labels = mnist.train.next_batch(tracking_batch_size)
                train_feed_dict = {input: batch_inputs,
                           labels: batch_labels}
                
                current_cost = sess.run(cost, feed_dict=train_feed_dict)
                gradients = sess.run(gradient_op, feed_dict=train_feed_dict)
                gradient_norm = np.linalg.norm(gradients)
                gradient_max = np.max(np.abs(gradients))
                parameters = sess.run(trained_parameters)
                
                add_to_results_scalars(crit_finder_results, current_cost, gradient_norm, batch_idx)
                add_to_results_vectors(crit_finder_results, gradients[0], parameters, batch_idx)
                
            
            batch_inputs, batch_labels = mnist.train.next_batch(crit_finder_batch_size)
            train_feed_dict = {input: batch_inputs,
                           labels: batch_labels}

            sess.run(step_crit_finder, feed_dict=train_feed_dict)
            
            
            if (batch_idx+1 == 1) or ((batch_idx+1)%crit_finder_track_every == 0):

                batch_inputs, batch_labels = mnist.train.next_batch(tracking_batch_size)
                train_feed_dict = {input: batch_inputs,
                           labels: batch_labels}
                
                current_cost = sess.run(cost, feed_dict=train_feed_dict)
                gradients = sess.run(gradient_op, feed_dict=train_feed_dict)
                gradient_norm = np.linalg.norm(gradients)
                gradient_max = np.max(np.abs(gradients))

                if print_tracking_data:
                    print_tracking("crit_finder step: {0}".format(batch_idx+1),
                                   current_cost, gradient_norm, gradient_max)
                
                add_to_results_scalars(crit_finder_results, current_cost, gradient_norm, batch_idx+1)

                if (batch_idx+1)+crit_finder_track_every > num_steps_crit_finder:
                    # on last tracked iteration
                    parameters = sess.run(trained_parameters)
                    add_to_results_vectors(crit_finder_results, gradients[0], parameters, batch_idx+1)

    return gd_results, crit_finder_results


## REWRITE TO USE THIS FUNCTION INSTEAD!

def run_optimizer(network, mnist, accuracy, input, labels, trained_parameters, cost, gradient_op, step_optimizer, 
                  num_steps, batch_size, track_every, tracking_batch_size, print_tracking_data, track_string):
    
    results = Results([],[],[],[],[],[])
    
    for batch_idx in range(num_steps):

        if (batch_idx+1 == 1):

            batch_inputs, batch_labels = mnist.train.next_batch(tracking_batch_size)
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

        batch_inputs, batch_labels = mnist.train.next_batch(batch_size)
        train_feed_dict = {input: batch_inputs,
                       labels: batch_labels}

        sess.run(step_optimizer, feed_dict=train_feed_dict)

        if (batch_idx+1 == 1) or ((batch_idx+1)%track_every == 0):

            batch_inputs, batch_labels = mnist.train.next_batch(tracking_batch_size)
            train_feed_dict = {input: batch_inputs,
                       labels: batch_labels}

            current_cost, gradients, parameters = sess.run([cost, gradient_op, trained_parameters],
                                                           feed_dict=train_feed_dict)
            gradient_norm = np.linalg.norm(gradients)
            gradient_max = np.max(np.abs(gradients))

            if print_tracking_data:
                print_tracking(track_string+" {0}:".format(batch_idx+1), current_cost, gradient_norm, gradient_max)

            add_to_results_scalars(results, current_cost, gradient_norm, batch_idx+1)

            if (batch_idx+1)+gradient_descent_track_every > num_steps_gd:
                # on last tracked iteration
                add_to_results_vectors(gd_results, gradients[0], parameters, batch_idx+1)
                    
    return results

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