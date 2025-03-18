from __future__ import annotations

from math import ceil

from qsharp.estimator import EstimatorParams, EstimatorResult, LogicalCounts, QECScheme, QubitParams

# For all experiments, we are using the logical resource counts as a starting
# point.  These have been computed using the qsharp Python package (version
# 1.8.0) for the https://aka.ms/fcidump/XVIII-cas4-fb-64e-56o Hamiltonian on
# the sample
# https://github.com/microsoft/qsharp/tree/main/samples/estimation/df-chemistry:
#
# ```
# $ python chemistry.py -f https://aka.ms/fcidump/XVIII-cas4-fb-64e-56o
# $ jq '.logicalCounts' < resource_estimate.json
# ```


logical_counts = LogicalCounts(
    {
        "numQubits": 1318,
        "tCount": 96,
        "rotationCount": 11987084,
        "rotationDepth": 11986482,
        "cczCount": 67474931068,
        "measurementCount": 63472407520,
    }
)

# --- Default qubit models ---

params = EstimatorParams(6)
params.error_budget = 0.01
params.items[0].qubit_params.name = QubitParams.GATE_US_E3
params.items[1].qubit_params.name = QubitParams.GATE_US_E4
params.items[2].qubit_params.name = QubitParams.GATE_NS_E3
params.items[3].qubit_params.name = QubitParams.GATE_NS_E4
params.items[4].qubit_params.name = QubitParams.MAJ_NS_E4
params.items[4].qec_scheme.name = QECScheme.FLOQUET_CODE
params.items[5].qubit_params.name = QubitParams.MAJ_NS_E6
params.items[5].qec_scheme.name = QECScheme.FLOQUET_CODE

results = logical_counts.estimate(params=params)

print()
print("Default qubit models")
print(results.summary_data_frame())
print()

# --- Evaluating different number of T factories ---

params = EstimatorParams(num_items=14)
params.qubit_params.name = QubitParams.MAJ_NS_E6
params.qec_scheme.name = QECScheme.FLOQUET_CODE

params.error_budget = 0.01
for i in range(14):
    params.items[i].constraints.max_t_factories = 14 - i

results = logical_counts.estimate(params=params)

print()
print("Different number of T factories")
print(results.summary_data_frame())
print()

# --- Modifying error rates and operating times ---

base_time = 50  # ns
base_error = 1e-3

error_growth = 1e-1
time_growth = 0.9

params = EstimatorParams(num_items=5)
params.error_budget = 0.01
for t in range(5):
    params.items[t].qubit_params.instruction_set = "gateBased"
    params.items[t].qubit_params.name = f"t{t}"
    params.items[t].qubit_params.one_qubit_measurement_time = f"{(2 * base_time) * time_growth**t} ns"
    params.items[t].qubit_params.one_qubit_gate_time = f"{base_time * time_growth**t} ns"
    params.items[t].qubit_params.two_qubit_gate_time = f"{base_time * time_growth**t} ns"
    params.items[t].qubit_params.t_gate_time = f"{base_time * time_growth**t} ns"
    params.items[t].qubit_params.one_qubit_measurement_error_rate = base_error * error_growth**t
    params.items[t].qubit_params.one_qubit_gate_error_rate = base_error * error_growth**t
    params.items[t].qubit_params.two_qubit_gate_error_rate = base_error * error_growth**t
    params.items[t].qubit_params.t_gate_error_rate = base_error * error_growth**t
    params.items[t].qubit_params.idle_error_rate = base_error * error_growth**t

results = logical_counts.estimate(params=params)

print()
print("Modifying error rates and operating times")
print(results.summary_data_frame())
print()

# --- Modifying logical counts ---


def modified_logical_counts(space_factor: float, time_factor: float):
    return LogicalCounts(
        {
            "numQubits": ceil(logical_counts["numQubits"] * space_factor),
            "tCount": ceil(logical_counts["tCount"] * time_factor),
            "rotationCount": ceil(logical_counts["rotationCount"] * time_factor),
            "rotationDepth": ceil(logical_counts["rotationDepth"] * time_factor),
            "cczCount": ceil(logical_counts["cczCount"] * time_factor),
            "measurementCount": ceil(logical_counts["measurementCount"] * time_factor),
        }
    )


params = EstimatorParams()
params.error_budget = 0.01
params.qubit_params.name = QubitParams.MAJ_NS_E6
params.qec_scheme.name = QECScheme.FLOQUET_CODE
estimates = []
for space_factor, time_factor in [(1.0, 1.0), (0.5, 2.0), (2.0, 0.5), (0.75, 0.75)]:
    counts = modified_logical_counts(space_factor, time_factor)
    estimates.append(counts.estimate(params=params))

print()
print("Modifying logical counts")
print(EstimatorResult(estimates).summary_data_frame())
print()
