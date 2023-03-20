from mqt.problemsolver.partialcompiler.evaluator import evaluate_QAOA


for i in range(45, 130, 10):
    print("Qubits:", i)
    #print(evaluate_QAOA(i, 5, sample_probability=0.5, optimize_swaps=False, opt_level_baseline=2))
    for j in [0.3,0.5,0.7]  :
       print(evaluate_QAOA(i,5, sample_probability=j, optimize_swaps=False, opt_level_baseline=2))