// Benchmark was created by MQT Bench on 2024-03-17
// For more information about MQT Bench, please visit https://www.cda.cit.tum.de/mqtbench/
// MQT Bench version: 1.1.0
// Qiskit version: 1.0.2

OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg meas[2];
h q[1];
cx q[1],q[0];
barrier q[0],q[1];
measure q[0] -> meas[0];
measure q[1] -> meas[1];