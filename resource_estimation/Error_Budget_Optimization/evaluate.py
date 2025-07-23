import numpy as np
from qsharp.estimator import EstimatorParams, ErrorBudgetPartition, LogicalCounts
from qsharp.interop.qiskit import estimate
import joblib
import matplotlib.pyplot as plt

def evaluate(X, Y, total_budget):
    qubits_diffs = []
    runtime_diffs = []
    product_diffs = []
    qubits_list = []
    runtime_list = []
    default_qubits_list = []
    default_runtime_list = []
    no_changes = 0
    for i, params in enumerate(Y):
        c = {}
        c['numQubits'] = int(X[i,0])
        c['tCount'] = int(X[i,1])
        c['rotationCount'] = int(X[i,2])
        c['rotationDepth'] = int(X[i,3])
        c['cczCount'] = int(X[i,4])
        c['ccixCount'] = int(X[i,5])
        c['measurementCount'] = int(X[i,6])
        logical_counts = LogicalCounts(c)
        params_sum = params[0] + params[1] + params[2]
        params = [params[0]/params_sum * total_budget, params[1]/params_sum * total_budget, params[2]/params_sum * total_budget]
        
        parameters = EstimatorParams()
        parameters.error_budget = ErrorBudgetPartition()
        parameters.error_budget.logical = params[0]
        parameters.error_budget.t_states = params[1]
        parameters.error_budget.rotations = params[2]

        default_parameters = EstimatorParams()
        default_parameters.error_budget = total_budget

        result = logical_counts.estimate(parameters)
        default_result = logical_counts.estimate(default_parameters)
        qubits = result["physicalCounts"]["physicalQubits"]
        runtime = result["physicalCounts"]["runtime"]
        default_qubits = default_result["physicalCounts"]["physicalQubits"]
        default_runtime = default_result["physicalCounts"]["runtime"]

        qubits_diff = (qubits - default_qubits)/default_qubits
        runtime_diff = (runtime - default_runtime)/default_runtime
        product_diff = ((qubits * runtime) - (default_qubits * default_runtime))/(default_qubits * default_runtime)
        if product_diff > 0:
            product_diff = 0
        
        if product_diff == 0:
            no_changes += 1

        qubits_diffs.append(qubits_diff)
        runtime_diffs.append(runtime_diff)
        product_diffs.append(product_diff)
        qubits_list.append(qubits)
        runtime_list.append(runtime)
        default_qubits_list.append(default_qubits)
        default_runtime_list.append(default_runtime)

    return qubits_diffs, runtime_diffs, product_diffs, qubits_list, runtime_list, default_qubits_list, default_runtime_list

    

def plot_results(product_diffs, product_diffs_optimal, name, legend=False, bin_width=4):
    product_diffs = [100 * i for i in product_diffs]
    product_diffs_optimal = [100 * i for i in product_diffs_optimal]

    all_data = product_diffs + product_diffs_optimal
    data_min = min(all_data)
    data_max = max(all_data)

    data_min = min(0, data_min)
    data_max += bin_width

    bin_edges = np.arange(data_min, data_max + bin_width, bin_width)

    x_ticks = np.arange(-100, 1, 20)

    fig, ax = plt.subplots(figsize=(5,2.5))

    ax.hist(product_diffs_optimal, bins=bin_edges, color='steelblue', edgecolor='black', alpha=0.5, label='Best Distributions Determined')
    ax.hist(product_diffs, bins=bin_edges, color='orange', edgecolor='black', alpha=0.5, label='Predicted Distributions')

    ax.set_xlim(data_min, data_max)

    ax.set_xticks([-100, -80, -60, -40, -20, 0])
    ax.set_yticks([0, 40, 80, 120])

    ax.set_xlabel('Space-Time Difference [%]', fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=15)

    if legend:
        ax.legend(loc='upper left', fontsize=12)

    plt.tight_layout()
    plt.show()