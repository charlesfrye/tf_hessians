"""Helper functions for running various tests and plotting their results."""
from . import graphs

from matplotlib import pyplot as plt
import matplotlib.ticker
import seaborn as sns

import numpy as np
import scipy.integrate

import inspect

## functions for examining the behavior of algorithms on quadratic forms

def compare_algorithms(matrix_generator, N=5, num_steps=5, num_matrices=10, num_runs=5,
                            algorithms=["gradient_descent","newton"],
                           hyperparameters={"learning_rate":0.1, "newton_rate":1,
                                            "fudge_factor":1e-6, "inverse_method":"fudged"}):
    """compare the gradient norms of results of running algorithms
    on num_matrices quadratic forms whose matrices are produced by matrix generator,
    which should be a function that takes a single argument, N, and returns a N by N matrix.

    The algorithms are both run from num_runs different initializations,
    and, for ease of implementation, num_steps *times*, with increasing numbers of steps.
    This is woefully inefficient, but works alright for num_steps less than 3.

    The results are returned as a 4-D array, results, with dimensions
    [num_steps+1, num_matrices, num_runs, len(algorithms)].
    """

    num_algorithms = len(algorithms)
    matrices = [matrix_generator(N) for _ in range(num_matrices)]

    results = np.zeros((num_steps+1, num_matrices, num_runs, num_algorithms))

    for matrix_idx, matrix in enumerate(matrices):

        for run_idx in range(num_runs):

            initial_values = np.random.standard_normal(size=N).astype(np.float32)

            quadratic_form = graphs.quadratics.make(matrix, initial_values, hyperparameters)
            for step_idx in range(num_steps+1):
                for algorithm_idx, algorithm in enumerate(algorithms):
                    _, values =  graphs.quadratics.run_algorithm(quadratic_form, algorithm, step_idx)
                    results[step_idx, matrix_idx, run_idx, algorithm_idx] = graphs.quadratics.get_result("gradient_norm",
                                                                           values, quadratic_form)

    return results

def normalize_runs(results):
    """normalize results from compare_algorthims so that each run
    starts with gradient_norm standardized to 1.
    """
    results = np.divide(results, results[0,:,:,:][None,:,:,:])
    return results

def gradient_test(N, matrix_generator, algorithm, num_steps, hyperparameters=graphs.quadratics.DEFAULTS):
    """output the initial and final outputs and gradient norms found by taking num_steps of algorithm
    with provided hyperparameters on a matrix returned by matrix_generator,
    a function that takes an argument N and returns an N by N matrix.
    """

    random_matrix = matrix_generator(N)

    initial_values = graphs.quadratics.generate_initial_values(N)

    quadratic_form = graphs.quadratics.make(random_matrix, initial_values, hyperparameters)

    initial_output = graphs.quadratics.get_result("output", initial_values, quadratic_form)
    initial_gradients = graphs.quadratics.get_result("gradients", initial_values, quadratic_form)[0]

    final_output, final_values = graphs.quadratics.run_algorithm(quadratic_form, algorithm, num_steps)
    final_gradients = graphs.quadratics.get_result("gradients", final_values, quadratic_form)[0]

    print("output:\n" +
          "\tinitial: {0}".format(initial_output),
          "\tfinal: {0}".format(final_output))

    print("gradient norm:\n" +
          "\tinitial: {0}".format(np.linalg.norm(initial_gradients, 2)),
          "\tfinal: {0}".format(np.linalg.norm(final_gradients, 2)))


# functions for plotting performance on quadratic forms

def plot_trajectories_comparison(results, algorithms=["gradient_descent","newton"],
                                 colors=["salmon","darkolivegreen","medium_blue"]):
    """plots the (normalized) gradient norms in results as a function of step count,
    for each algorithm in algorithms, with colors given by colors.
    """
    f = plt.figure(figsize=(16,4))
    ax = plt.subplot(111)
    for matrix_idx in range(results.shape[1]):
        for algorithm_idx in range(results.shape[3]):
            color = colors[algorithm_idx]
            algorithm = algorithms[algorithm_idx]
            plt.plot(results[:,matrix_idx,:,algorithm_idx], color=color, linewidth=1, alpha=0.5, label=algorithm);

    plt.legend()
    ax.set_ylabel(r"Normalized $\|\nabla f\|^2$",fontsize=16, fontweight="bold")
    ax.set_xlabel("Iteration Number",fontsize=16, fontweight="bold")
    ax.set_ylim([0,1.1])
    ax.set_yticks([0,1]);
    ax.set_xticks(range(0,1+int(max(ax.get_xlim()))));

def plot_benefit(results, algorithms=["gradient_descent","newton"]):
    """plots the difference in gradient norm between algorithms[1] and algorithms[0]
    on a per-run basis.
    """
    f = plt.figure(figsize=(16,8))
    ax = plt.subplot(111)
    for matrix_idx in range(results.shape[1]):
        benefits = results[:,matrix_idx,:,1] - results[:,matrix_idx,:,0]
        plt.plot(benefits, linestyle="None",
                 marker='o', color='grey', markersize=24, alpha=0.05);
    xlim = ax.get_xlim()
    plt.hlines(0,*xlim, linewidth=4)
    ax.set_xlim(xlim)

    ax.set_ylabel(r"$\leftarrow$"+ "{0} better".format(algorithms[1]) +
                  "\t\t\t"+ "{0} better".format(algorithms[0]) + r"$\rightarrow$",
                  fontsize=16, fontweight="bold")
    ax.set_xlabel("Iteration Number",fontsize=16, fontweight="bold")
    ax.set_xticks(range(0,1+int(max(ax.get_xlim()))));

## functions for plotting spectral distributions

def wigner_semicircle(lam):
    return 1/(2*np.pi)*np.sqrt(2**2-lam**2)

def plot_wigner_comparison(eigvals):
    plt.figure(figsize=(16,6))
    N = len(eigvals)
    sns.distplot(eigvals, kde=False, bins=max(N//20,10),
                 hist_kws={"normed":True, "histtype":"step", "linewidth":8, "alpha":0.8},
                label="empirical spectral density");
    
    lams = np.linspace(-2, 2, 100);
    plt.plot(lams, wigner_semicircle(lams),linewidth=8, label="expected spectral density");
    plt.ylabel(r"$\rho\left(\lambda\right)$", fontsize=24); plt.xlabel(r"$\lambda$", fontsize=24);
    plt.legend(fontsize=16, loc=8);

def marchenkopastur_density(x, N, k, sigma=1):
    """the density for the non-singular portion of the marchenko-pastur distribution,
    as given by https://en.wikipedia.org/wiki/Marchenko-Pastur_distribution.
    """
    lam = N/k
    scaling_factor = 1/(2*np.pi*sigma**2)

    lam_plus = sigma**2*(1+np.sqrt(lam))**2
    lam_minus = sigma**2*(1-np.sqrt(lam))**2

    if (x>lam_minus and x<lam_plus):
        return scaling_factor*(np.sqrt((lam_plus-x)*(x-lam_minus))/(lam*x))
    else:
        return 0

def marchenkopastur_cumulative_distribution(xs, N, k):
    """the cumulative distribution for the marchenko-pastur distribution,
    calculated by numerically integrating the density for the non-singular portion
    and then adding the singular part.
    valid for xs>=0."""
    return max((1-k/N), 0)+scipy.integrate.cumtrapz([marchenkopastur_density(x, N, k) for x in xs],xs)

def plot_marchenko_comparison(eigvals, N, k, eps=1e-6):
    """compare the histogram of the eigvals from a matrix
    to the marchenko-pastur distribution with parameters N, k, and sigma=1.
    the marchenko-pastur distribution is plotted from eps to max(eigvals) with precision 1e-5.
    """
    plt.figure(figsize=(16,6))

    sns.distplot(eigvals, kde=False, bins=max(len(eigvals)//10,10),
                     hist_kws={"normed":True, "histtype":"step", "linewidth":8, "alpha":0.8,
                              "cumulative":True},
                    label="empirical cumulative spectral distribution");

    eps = 1e-6
    xs = np.linspace(eps, max(eigvals),num=int(1e5))
    plt.plot(xs[:-1], marchenkopastur_cumulative_distribution(xs, N, k),linewidth=4,
            label="expected cumulative spectral distribution");
    plt.tick_params(labelsize=16)
    plt.ylabel(r"$P(\Lambda \leq \lambda)$", fontsize=24)
    plt.xlabel(r"$\lambda$", fontsize=24)
    plt.legend(fontsize=16,loc=8);

def display_function(function):
    lines = inspect.getsourcelines(function)
    print("".join(lines[0]))

## functions for testing when Newton's method fails to converge

def condition_test(kappas, hyperparameters, N=1000, k=100):
    """applies the Newton method with provided hyperparameters for one step
    to diagonal N by N matrices with condition numbers in kappas
    and returns the solution_norms.
    condition number is the ratio of largest to smallest eigenvalue.

    matrices have a block structure, with one block of size N-k
    equal to the identity matrix and another block of size k equal to
    1/kappa times the identity matrix:

    [[1,0, ... 0, 0],
    [0,1, ... 0, 0],
    [0,0, ... 1/kappa, 0],
    [0,0, ..., 0, 1/kappa]]
    """
    solution_norms = []

    initial_values = 1/np.sqrt(N)*np.random.standard_normal(size=N).astype(np.float32)

    for kappa in kappas:
        eigvals = [1]*(N-k)+k*[1/kappa]
        matrix = np.diag(eigvals).astype(np.float32)

        quadratic_form = graphs.quadratics.make(matrix, initial_values, hyperparameters)

        _, values = graphs.quadratics.run_algorithm(quadratic_form, "newton", 1)

        gradients = graphs.quadratics.get_result("gradients", values, quadratic_form)[0]

        if np.linalg.norm(gradients)>1e-4:
            print("convergence failed on kappa={0}".format(kappa))
            solution_norms.append(np.inf)
            continue

        solution_norms.append(np.linalg.norm(values))

    return solution_norms

def run_condition_tests(kappas, N, ks, hyperparameters):
    """ runs condition_test len(ks) times with kappas, hyperparameters, and N fixed
    and k given by ks.
    results are returned as a list of lists of solution norms.
    """
    results = []

    for k in ks:
        results.append(condition_test(kappas, hyperparameters, N, k,))

    return results

def plot_results_condition_tests(kappas, results, flat_fractions, minimum_eigenvalue_magnitude):
    """ plots the results of running run_condition_tests on kappas
    with flat_fraction equal to k/N for each result in results.

    for accurate results, should be used on outputs of run_condition_tests called with
    the "pseudo" inverse method with minimum_eigenvalue_magnitude as its threshold.
    """
    plt.figure(figsize=(16,6))

    for result, flat_fraction in zip(results, flat_fractions):
        plot_result_condition_test(kappas, result, flat_fraction)
    ylims = plt.ylim()
    plt.vlines(minimum_eigenvalue_magnitude**-1,*ylims,
               color='firebrick', linewidth=4, alpha=0.8,
               label="inverse eigenvalue threshold");
    plt.xticks(fontsize=16); plt.yticks(fontsize=16);
    plt.ylabel(r"$\|\|\ \theta\ \|\|$", fontsize=24)
    plt.xlabel(r"$\kappa$, condition number", fontsize=24)
    plt.ylim(ylims)
    plt.legend(fontsize=16);

def plot_result_condition_test(kappas, result, flat_fraction):
    """ plots the results of condition_test called on kappas with parameters flat_fraction = k/N.
    """
    plt.loglog(kappas, result, label="{0}% of directions are numerically flat".format(100*flat_fraction),
                 linestyle='None', marker='.', markersize=24, alpha=0.75);

def plot_condition_test_distance_scaling(distances, ks):
    """ plots the scaling of the distances of the solutions from the true critical point
    (which are equal to their norms, since the critical point for a quadratic form is at 0)
    as a function of k in ks, and compares to $\sqrt(k)$ scaling.
    """
    plt.figure(figsize=(12,6));
    plt.loglog(ks, distances, label="observed", linewidth=4);
    plt.loglog(ks, np.power(ks,0.5), label=r"$\sqrt{k}$ scaling, offset", linewidth=4);
    plt.legend(fontsize=16);
    plt.xticks(fontsize=16);
    plt.yticks(fontsize=16);
    plt.xlabel("$k$, number of flat directions", fontsize=24);
    plt.ylabel(r"$\|\|\ \theta\ \|\|$", fontsize=24);

## functions for plotting performance on neural networks

def plot_results(optimizer_results, crit_finder_results, titles):
    """ plots gradient norms as a function of step given by Results instances
    for optimizer and crit finder, and adds titles to plots
    """
    f, axs = plt.subplots(nrows=1, ncols=2, figsize=(20,6), sharey=True)

    axs[0].plot(optimizer_results.scalar_index, optimizer_results.gradient_norm, linewidth=4, color='C0')

    axs[0].set_ylabel(r"$\|\|\nabla f \|\|$",fontsize=24);
    axs[0].set_xlabel("batch index", fontsize=24);
    axs[0].tick_params(axis='both', which='major', labelsize=16)
    axs[0].set_title(titles[0], fontsize=28)

    axs[1].plot(np.asarray(crit_finder_results.scalar_index)+optimizer_results.scalar_index[-1],
             crit_finder_results.gradient_norm, linewidth=4, color='C1')

    axs[1].set_xlabel("batch index", fontsize=24);
    axs[1].tick_params(axis='both', which='major', labelsize=16)
    axs[1].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%d'))
    axs[1].set_title(titles[1], fontsize=28)

def compare_gradients_entrywise(optimizer_results, crit_finder_results, labels):
    """compares the sorted entries of the first entry in optimizer.gradient (i.e. at initialization),
    the first entry of crit_finder_results.gradient (i.e. after optimizer has been run)
    and the last entry of crit_finder_results (i.e. after the crit_finder has stopped running.
    legend entries are given by labels.
    the x axis is scaled logarithmically.
    """
    f, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,6),)

    gradients = [optimizer_results.gradient[0], crit_finder_results.gradient[0], crit_finder_results.gradient[-1]]

    for gradient, label in zip(gradients, labels):
        plot_gradient_entrywise(ax, gradient, label=label)

    plt.legend(fontsize=16)

def plot_gradient_entrywise(ax, gradient, label):
    """ plots sorted entries of a single gradient vector on ax
    with a log-scale for x and assigns legend entry label.
    """
    ax.semilogx(range(1,len(gradients)+1), sorted(gradients), linewidth=4, label=label)
    ax.tick_params(axis='both', which='major', labelsize=16)

def compare_gradient_histograms(optimizer_results, crit_finder_results, titles, colors=['C0','C1','C2']):
    """compares the log-scaled histogram of gradient entries for
    the first entry in optimizer_results.gradient (i.e. at initialization),
    the first entry of crit_finder_results.gradient (i.e. after optimizer has been run)
    and the last entry of crit_finder_results (i.e. after the crit_finder has stopped running.


    plots are titled with titles and histograms are colored with colors.
    """
    f, axs = plt.subplots(nrows=3, ncols=1, figsize=(20,18), sharex=True, sharey=True)
    axs[0].set_yscale('log')

    gradients = [optimizer_results.0], crit_finder_results.gradient[0], crit_finder_results.gradient[-1]]

    for gradient, ax, title, color in zip(gradients, axs, titles, colors):

        plot_gradient_histogram(ax, gradient, title, color)

def plot_gradient_histogram(ax, gradient, title, color):
    """plots a histogram with hue given by color
    of entries in gradient vector on ax and gives it title
    """
    ax.hist(gradient, normed=True, bins=100, histtype='step', color=color, linewidth=4);
    ax.set_title(title, fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=16)
