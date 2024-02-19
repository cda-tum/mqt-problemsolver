import os

from azure.quantum import Workspace
from azure.quantum.target.microsoft import MicrosoftEstimator, QubitParams, QECScheme
from azure.quantum.chemistry import df_chemistry

import qsharp

resource_id = os.environ.get("AZURE_QUANTUM_RESOURCE_ID")
location = os.environ.get("AZURE_QUANTUM_LOCATION")

workspace = Workspace(resource_id=resource_id, location=location)
estimator = MicrosoftEstimator(workspace)

### Default qubit models

params = estimator.make_params(num_items=6)

# select Hamiltonian
params.file_uris["fcidumpUri"] = "https://aka.ms/fcidump/XVIII-cas4-fb-64e-56o"

params.error_budget = 0.01
params.items[0].qubit_params.name = QubitParams.GATE_US_E3
params.items[1].qubit_params.name = QubitParams.GATE_US_E4
params.items[2].qubit_params.name = QubitParams.GATE_NS_E3
params.items[3].qubit_params.name = QubitParams.GATE_NS_E4
params.items[4].qubit_params.name = QubitParams.MAJ_NS_E4
params.items[4].qec_scheme.name = QECScheme.FLOQUET_CODE
params.items[5].qubit_params.name = QubitParams.MAJ_NS_E6
params.items[5].qec_scheme.name = QECScheme.FLOQUET_CODE

job = estimator.submit(df_chemistry(), input_params=params)
results = job.get_results()

print()
print("Default qubit models")
print(results.summary_data_frame())
print()


### Evaluating different number of T factories

# For the next experiment, estimate from logical counts to save time
lcounts = results[0]["logicalCounts"]
program = f"""
open Microsoft.Quantum.ResourceEstimation;

operation Main() : Unit {{
    use qubits = Qubit[1369];

    AccountForEstimates(
        [
            CczCount({lcounts["cczCount"] + lcounts["ccixCount"]}),
            TCount({lcounts["tCount"]}),
            RotationCount({lcounts["rotationCount"]}),
            RotationDepth({lcounts["rotationDepth"]}),
            MeasurementCount({lcounts["measurementCount"]})
        ],
        PSSPCLayout(),
        qubits
    );
}}
"""

Main = qsharp.compile(program)

params = estimator.make_params(num_items=14)
params.qubit_params.name = QubitParams.MAJ_NS_E6
params.qec_scheme.name = QECScheme.FLOQUET_CODE

params.error_budget = 0.01
for i in range(14):
    params.items[i].constraints.max_t_factories = 14 - i

job = estimator.submit(Main, input_params=params)
results = job.get_results()

print()
print("Different number of T factories")
print(results.summary_data_frame())
print()

### Modifying error rates and operating times

base_time = 50 # ns
base_error = 1e-3

error_growth = 1e-1
time_growth = .9

params = estimator.make_params(num_items=5)
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

job = estimator.submit(Main, input_params=params)
results = job.get_results()

print()
print("Modifying error rates and operating times")
print(results.summary_data_frame())
print()

### Modifying logical counts

program_with_args = f"""
open Microsoft.Quantum.Math;
open Microsoft.Quantum.ResourceEstimation;

operation Main(spaceFactor : Double, timeFactor : Double) : Unit {{
    use qubits = Qubit[Ceiling(1369.0 * spaceFactor)];

    AccountForEstimates(
        [
            CczCount(Ceiling({lcounts["cczCount"] + lcounts["ccixCount"]}.0 * timeFactor)),
            TCount(Ceiling({lcounts["tCount"]}.0 * timeFactor)),
            RotationCount(Ceiling({lcounts["rotationCount"]}.0 * timeFactor)),
            RotationDepth(Ceiling({lcounts["rotationDepth"]}.0 * timeFactor)),
            MeasurementCount(Ceiling({lcounts["measurementCount"]}.0 * timeFactor))
        ],
        PSSPCLayout(),
        qubits
    );
}}
"""

MainWithArgs = qsharp.compile(program_with_args)

params = estimator.make_params(num_items=4)
params.error_budget = 0.01
params.qubit_params.name = QubitParams.MAJ_NS_E6
params.qec_scheme.name = QECScheme.FLOQUET_CODE
params.items[0].arguments["spaceFactor"] = 1.0
params.items[0].arguments["timeFactor"] = 1.0
params.items[1].arguments["spaceFactor"] = 0.5
params.items[1].arguments["timeFactor"] = 2.0
params.items[2].arguments["spaceFactor"] = 2.0
params.items[2].arguments["timeFactor"] = 0.5
params.items[3].arguments["spaceFactor"] = 0.75
params.items[3].arguments["timeFactor"] = 0.75

job = estimator.submit(MainWithArgs, input_params=params)
results = job.get_results()

print()
print("Modifying logical counts")
print(results.summary_data_frame())
print()
