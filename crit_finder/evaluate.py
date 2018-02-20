from . import graphs

from matplotlib import pyplot as plt
import matplotlib.ticker
import seaborn as sns

import numpy as np
import scipy.integrate


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

            quadratic_form = graphs.make_quadratic_form(matrix, initial_values, hyperparameters)
            for step_idx in range(num_steps+1):
                for method_idx, method in enumerate(methods):
                    _, values =  graphs.minimize(quadratic_form, method, step_idx)        
                    results[step_idx, matrix_idx, run_idx, method_idx] = graphs.get_result("gradient_norm", 
                                                                           values, quadratic_form)

    return results

def gradient_test(N, matrix_generator, algorithm, num_steps, hyperparameters=graphs.DEFAULTS):

    random_matrix = matrix_generator(N)

    initial_values = graphs.generate_initial_values(N)

    quadratic_form = graphs.make_quadratic_form(random_matrix, initial_values, hyperparameters)

    initial_output = graphs.get_result("output", initial_values, quadratic_form)
    initial_gradients = graphs.get_result("gradients", initial_values, quadratic_form)[0]

    final_output, final_values = graphs.minimize(quadratic_form, algorithm, num_steps)
    final_gradients = graphs.get_result("gradients", final_values, quadratic_form)[0]

    print("output:\n" +
          "\tinitial: {0}".format(initial_output),
          "\tfinal: {0}".format(final_output))

    print("gradient norm:\n" +
          "\tinitial: {0}".format(np.linalg.norm(initial_gradients, 2)),
          "\tfinal: {0}".format(np.linalg.norm(final_gradients, 2)))

def normalize_runs(results):
    results = np.divide(results, results[0,:,:,:][None,:,:,:])
    return results

# functions for plotting performance on quadratic forms

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

# functions for plotting spectral distributions

def marchenkopastur_density(x, N, k, sigma=1):
    lam = N/k
    scaling_factor = 1/(2*np.pi*sigma**2)
    
    lam_plus = sigma**2*(1+np.sqrt(lam))**2
    lam_minus = sigma**2*(1-np.sqrt(lam))**2
    
    if (x>lam_minus and x<lam_plus):
        return scaling_factor*(np.sqrt((lam_plus-x)*(x-lam_minus))/(lam*x))
    else:
        return 0
    
def marchenkopastur_cumulative_distribution(xs, N, k):
    return (1-k/N)+scipy.integrate.cumtrapz([marchenkopastur_density(x, N, k) for x in xs],xs)
                                            
def plot_marchenko_comparison(eigvals, N, k, eps=1e-6):
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
    
# functions for testing when Newton's method fails to converge

def condition_test(kappas, hyperparameters, N=1000, k=100):

    results = []

    initial_values = 1/np.sqrt(N)*np.random.standard_normal(size=N).astype(np.float32)

    for kappa in kappas:
        eigvals = [1]*(N-k)+k*[1/kappa]
        matrix = np.diag(eigvals).astype(np.float32)

        quadratic_form = graphs.make_quadratic_form(matrix, initial_values, hyperparameters)
        
        _, values = graphs.minimize(quadratic_form, "newton", 1)

        gradients = graphs.get_result("gradients", values, quadratic_form)[0]

        if np.linalg.norm(gradients)>1e-4:
            print("convergence failed on kappa={0}".format(kappa))
            results.append(np.inf)
            continue

        results.append(np.linalg.norm(values))
    
    return results
    
def run_condition_tests(kappas, N, ks, hyperparameters):
    results = []
    
    for k in ks:
        results.append(condition_test(kappas, hyperparameters, N, k,))
    
    return results

def plot_results_condition_tests(kappas, results, flat_fractions, minimum_eigenvalue_magnitude):
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
    plt.loglog(kappas, result, label="{0}% of directions are numerically flat".format(100*flat_fraction),
                 linestyle='None', marker='.', markersize=24, alpha=0.75);
    
def plot_condition_test_distance_scaling(distances, ks):
    plt.figure(figsize=(12,6));
    plt.loglog(ks, distances, label="observed", linewidth=4);
    plt.loglog(ks, np.power(ks,0.5), label=r"$\sqrt{k}$ scaling, offset", linewidth=4);
    plt.legend(fontsize=16);
    plt.xticks(fontsize=16);
    plt.yticks(fontsize=16);
    plt.xlabel("$k$, number of flat directions", fontsize=24);
    plt.ylabel(r"$\|\|\ \theta\ \|\|$", fontsize=24);
    
# functions for plotting performance on neural networks

def plot_results(gd_results, crit_finder_results, crit_finder_name):
    f, axs = plt.subplots(nrows=1, ncols=2, figsize=(20,6), sharey=True)

    axs[0].plot(gd_results.scalar_index, gd_results.gradient_norm, linewidth=4, color='C0')
    
    axs[0].set_ylabel(r"$\|\|\nabla f \|\|^2$",fontsize=24);
    axs[0].set_xlabel("batch index", fontsize=24);
    axs[0].tick_params(axis='both', which='major', labelsize=16)
    axs[0].set_title("gradient descent", fontsize=28)

    axs[1].plot(np.asarray(crit_finder_results.scalar_index)+gd_results.scalar_index[-1],
             crit_finder_results.gradient_norm, linewidth=4, color='C1')

    axs[1].set_xlabel("batch index", fontsize=24);
    axs[1].tick_params(axis='both', which='major', labelsize=16)
    axs[1].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%d'))
    axs[1].set_title(crit_finder_name, fontsize=28)
    
def compare_gradients_entrywise(gd_results, crit_finder_results, labels):
    f, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,6),)

    plot_gradients_entrywise(ax, gd_results.gradients[0], label=labels[0])

    plot_gradients_entrywise(ax, crit_finder_results.gradients[0], label=labels[1])

    plot_gradients_entrywise(ax, crit_finder_results.gradients[-1], label=labels[2])

    plt.legend(fontsize=16)
    
def plot_gradients_entrywise(ax, gradients, label):
    ax.semilogx(range(1,len(gradients)+1), sorted(gradients), linewidth=4, label=label)
    ax.tick_params(axis='both', which='major', labelsize=16)
    
def compare_gradients_histogram(gd_results, crit_finder_results, titles, colors=['C0','C1','C2']):
    f, axs = plt.subplots(nrows=3, ncols=1, figsize=(20,18), sharex=True, sharey=True)
    axs[0].set_yscale('log')
    
    plot_gradients_histogram(axs[0], gd_results.gradients[0], title=titles[0], color = colors[0])
    
    plot_gradients_histogram(axs[1], crit_finder_results.gradients[0], title=titles[1], color = colors[1])
    
    plot_gradients_histogram(axs[2], crit_finder_results.gradients[-1], title=titles[2], color = colors[2])
    
def plot_gradients_histogram(ax, gradients, title, color):
    ax.hist(gradients, normed=True, bins=100, histtype='step', color=color, linewidth=4);
    ax.set_title(title, fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=16)