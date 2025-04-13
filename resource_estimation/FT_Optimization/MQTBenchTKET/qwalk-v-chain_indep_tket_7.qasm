// Benchmark was created by MQT Bench on 2024-03-19
// For more information about MQT Bench, please visit https://www.cda.cit.tum.de/mqtbench/
// MQT Bench version: 1.1.0
// TKET version: 1.25.0

OPENQASM 2.0;
include "qelib1.inc";

qreg anc[2];
qreg coin[1];
qreg node[4];
creg meas[7];
h anc[0];
h anc[1];
h coin[0];
t anc[0];
t anc[1];
cx node[1],anc[0];
tdg anc[0];
cx coin[0],anc[0];
t anc[0];
cx node[1],anc[0];
tdg anc[0];
h anc[0];
cx anc[0],anc[1];
tdg anc[1];
cx node[2],anc[1];
t anc[1];
cx anc[0],anc[1];
tdg anc[1];
h anc[1];
ccx node[3],anc[1],node[0];
h anc[1];
t anc[1];
cx anc[0],anc[1];
tdg anc[1];
cx node[2],anc[1];
t anc[1];
cx anc[0],anc[1];
h anc[0];
tdg anc[1];
t anc[0];
h anc[1];
cx node[1],anc[0];
h anc[1];
tdg anc[0];
t anc[1];
cx coin[0],anc[0];
t anc[0];
cx node[1],anc[0];
tdg anc[0];
h anc[0];
h anc[0];
t anc[0];
cx node[2],anc[0];
tdg anc[0];
cx coin[0],anc[0];
t anc[0];
cx node[2],anc[0];
tdg anc[0];
h anc[0];
ccx node[3],anc[0],node[1];
h anc[0];
x node[1];
t anc[0];
cx node[2],anc[0];
tdg anc[0];
cx coin[0],anc[0];
t anc[0];
cx node[2],anc[0];
tdg anc[0];
ccx coin[0],node[3],node[2];
h anc[0];
cx coin[0],node[3];
x node[2];
h anc[0];
x coin[0];
x node[3];
t anc[0];
cx node[1],anc[0];
tdg anc[0];
cx coin[0],anc[0];
t anc[0];
cx node[1],anc[0];
tdg anc[0];
h anc[0];
cx anc[0],anc[1];
tdg anc[1];
cx node[2],anc[1];
t anc[1];
cx anc[0],anc[1];
tdg anc[1];
h anc[1];
ccx node[3],anc[1],node[0];
h anc[1];
t anc[1];
cx anc[0],anc[1];
tdg anc[1];
cx node[2],anc[1];
t anc[1];
cx anc[0],anc[1];
h anc[0];
tdg anc[1];
t anc[0];
h anc[1];
cx node[1],anc[0];
h anc[1];
tdg anc[0];
t anc[1];
cx coin[0],anc[0];
t anc[0];
cx node[1],anc[0];
tdg anc[0];
h anc[0];
h anc[0];
t anc[0];
cx node[2],anc[0];
tdg anc[0];
cx coin[0],anc[0];
t anc[0];
cx node[2],anc[0];
tdg anc[0];
h anc[0];
ccx node[3],anc[0],node[1];
h anc[0];
x node[1];
t anc[0];
cx node[2],anc[0];
tdg anc[0];
cx coin[0],anc[0];
t anc[0];
cx node[2],anc[0];
tdg anc[0];
ccx coin[0],node[3],node[2];
h anc[0];
cx coin[0],node[3];
x node[2];
h anc[0];
x coin[0];
x node[3];
t anc[0];
h coin[0];
cx node[1],anc[0];
tdg anc[0];
cx coin[0],anc[0];
t anc[0];
cx node[1],anc[0];
tdg anc[0];
h anc[0];
cx anc[0],anc[1];
tdg anc[1];
cx node[2],anc[1];
t anc[1];
cx anc[0],anc[1];
tdg anc[1];
h anc[1];
ccx node[3],anc[1],node[0];
h anc[1];
t anc[1];
cx anc[0],anc[1];
tdg anc[1];
cx node[2],anc[1];
t anc[1];
cx anc[0],anc[1];
h anc[0];
tdg anc[1];
t anc[0];
h anc[1];
cx node[1],anc[0];
h anc[1];
tdg anc[0];
t anc[1];
cx coin[0],anc[0];
t anc[0];
cx node[1],anc[0];
tdg anc[0];
h anc[0];
h anc[0];
t anc[0];
cx node[2],anc[0];
tdg anc[0];
cx coin[0],anc[0];
t anc[0];
cx node[2],anc[0];
tdg anc[0];
h anc[0];
ccx node[3],anc[0],node[1];
h anc[0];
x node[1];
t anc[0];
cx node[2],anc[0];
tdg anc[0];
cx coin[0],anc[0];
t anc[0];
cx node[2],anc[0];
tdg anc[0];
ccx coin[0],node[3],node[2];
h anc[0];
cx coin[0],node[3];
x node[2];
h anc[0];
x coin[0];
x node[3];
t anc[0];
cx node[1],anc[0];
tdg anc[0];
cx coin[0],anc[0];
t anc[0];
cx node[1],anc[0];
tdg anc[0];
h anc[0];
cx anc[0],anc[1];
tdg anc[1];
cx node[2],anc[1];
t anc[1];
cx anc[0],anc[1];
tdg anc[1];
h anc[1];
ccx node[3],anc[1],node[0];
h anc[1];
t anc[1];
cx anc[0],anc[1];
tdg anc[1];
cx node[2],anc[1];
t anc[1];
cx anc[0],anc[1];
h anc[0];
tdg anc[1];
t anc[0];
h anc[1];
cx node[1],anc[0];
h anc[1];
tdg anc[0];
t anc[1];
cx coin[0],anc[0];
t anc[0];
cx node[1],anc[0];
tdg anc[0];
h anc[0];
h anc[0];
t anc[0];
cx node[2],anc[0];
tdg anc[0];
cx coin[0],anc[0];
t anc[0];
cx node[2],anc[0];
tdg anc[0];
h anc[0];
ccx node[3],anc[0],node[1];
h anc[0];
x node[1];
t anc[0];
cx node[2],anc[0];
tdg anc[0];
cx coin[0],anc[0];
t anc[0];
cx node[2],anc[0];
tdg anc[0];
ccx coin[0],node[3],node[2];
h anc[0];
cx coin[0],node[3];
x node[2];
h anc[0];
x coin[0];
x node[3];
t anc[0];
h coin[0];
cx node[1],anc[0];
tdg anc[0];
cx coin[0],anc[0];
t anc[0];
cx node[1],anc[0];
tdg anc[0];
h anc[0];
cx anc[0],anc[1];
tdg anc[1];
cx node[2],anc[1];
t anc[1];
cx anc[0],anc[1];
tdg anc[1];
h anc[1];
ccx node[3],anc[1],node[0];
h anc[1];
t anc[1];
cx anc[0],anc[1];
tdg anc[1];
cx node[2],anc[1];
t anc[1];
cx anc[0],anc[1];
h anc[0];
tdg anc[1];
t anc[0];
h anc[1];
cx node[1],anc[0];
h anc[1];
tdg anc[0];
t anc[1];
cx coin[0],anc[0];
t anc[0];
cx node[1],anc[0];
tdg anc[0];
h anc[0];
h anc[0];
t anc[0];
cx node[2],anc[0];
tdg anc[0];
cx coin[0],anc[0];
t anc[0];
cx node[2],anc[0];
tdg anc[0];
h anc[0];
ccx node[3],anc[0],node[1];
h anc[0];
x node[1];
t anc[0];
cx node[2],anc[0];
tdg anc[0];
cx coin[0],anc[0];
t anc[0];
cx node[2],anc[0];
tdg anc[0];
ccx coin[0],node[3],node[2];
h anc[0];
cx coin[0],node[3];
x node[2];
h anc[0];
x coin[0];
x node[3];
t anc[0];
cx node[1],anc[0];
tdg anc[0];
cx coin[0],anc[0];
t anc[0];
cx node[1],anc[0];
tdg anc[0];
h anc[0];
cx anc[0],anc[1];
tdg anc[1];
cx node[2],anc[1];
t anc[1];
cx anc[0],anc[1];
tdg anc[1];
h anc[1];
ccx node[3],anc[1],node[0];
h anc[1];
t anc[1];
cx anc[0],anc[1];
tdg anc[1];
cx node[2],anc[1];
t anc[1];
cx anc[0],anc[1];
h anc[0];
tdg anc[1];
t anc[0];
h anc[1];
cx node[1],anc[0];
tdg anc[0];
cx coin[0],anc[0];
t anc[0];
cx node[1],anc[0];
tdg anc[0];
h anc[0];
h anc[0];
t anc[0];
cx node[2],anc[0];
tdg anc[0];
cx coin[0],anc[0];
t anc[0];
cx node[2],anc[0];
tdg anc[0];
h anc[0];
ccx node[3],anc[0],node[1];
h anc[0];
x node[1];
t anc[0];
cx node[2],anc[0];
tdg anc[0];
cx coin[0],anc[0];
t anc[0];
cx node[2],anc[0];
tdg anc[0];
ccx coin[0],node[3],node[2];
h anc[0];
cx coin[0],node[3];
x node[2];
x coin[0];
x node[3];
barrier node[0],node[1],node[2],node[3],coin[0],anc[0],anc[1];
measure anc[0] -> meas[5];
measure anc[1] -> meas[6];
measure coin[0] -> meas[4];
measure node[0] -> meas[0];
measure node[1] -> meas[1];
measure node[2] -> meas[2];
measure node[3] -> meas[3];
