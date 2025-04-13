// Benchmark was created by MQT Bench on 2024-03-19
// For more information about MQT Bench, please visit https://www.cda.cit.tum.de/mqtbench/
// MQT Bench version: 1.1.0
// TKET version: 1.25.0

OPENQASM 2.0;
include "qelib1.inc";

qreg flag[1];
qreg q[6];
creg meas[7];
h flag[0];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
x q[5];
t flag[0];
cu1(0.06249999999999997*pi) q[4],q[5];
cx q[4],q[3];
cu1(1.9375*pi) q[3],q[5];
cx q[4],q[3];
cu1(0.06249999999999997*pi) q[3],q[5];
cx q[3],q[2];
cu1(1.9375*pi) q[2],q[5];
cx q[4],q[2];
cu1(0.06249999999999997*pi) q[2],q[5];
cx q[3],q[2];
cu1(1.9375*pi) q[2],q[5];
cx q[4],q[2];
cu1(0.06249999999999997*pi) q[2],q[5];
cx q[2],q[1];
cu1(1.9375*pi) q[1],q[5];
cx q[4],q[1];
cu1(0.06249999999999997*pi) q[1],q[5];
cx q[3],q[1];
cu1(1.9375*pi) q[1],q[5];
cx q[4],q[1];
cu1(0.06249999999999997*pi) q[1],q[5];
cx q[2],q[1];
cu1(1.9375*pi) q[1],q[5];
cx q[4],q[1];
cu1(0.06249999999999997*pi) q[1],q[5];
cx q[3],q[1];
cu1(1.9375*pi) q[1],q[5];
cx q[4],q[1];
cu1(0.06249999999999997*pi) q[1],q[5];
cx q[1],q[0];
cu1(1.9375*pi) q[0],q[5];
cx q[4],q[0];
cu1(0.06249999999999997*pi) q[0],q[5];
cx q[3],q[0];
cu1(1.9375*pi) q[0],q[5];
cx q[4],q[0];
cu1(0.06249999999999997*pi) q[0],q[5];
cx q[2],q[0];
cu1(1.9375*pi) q[0],q[5];
cx q[4],q[0];
cu1(0.06249999999999997*pi) q[0],q[5];
cx q[3],q[0];
cu1(1.9375*pi) q[0],q[5];
cx q[4],q[0];
cu1(0.06249999999999997*pi) q[0],q[5];
cx q[1],q[0];
cu1(1.9375*pi) q[0],q[5];
h q[1];
cx q[4],q[0];
x q[1];
cu1(0.06249999999999997*pi) q[0],q[5];
cx q[3],q[0];
cu1(1.9375*pi) q[0],q[5];
cx q[4],q[0];
cu1(0.06249999999999997*pi) q[0],q[5];
cx q[2],q[0];
cu1(1.9375*pi) q[0],q[5];
h q[2];
cx q[4],q[0];
x q[2];
cu1(0.06249999999999997*pi) q[0],q[5];
cx q[3],q[0];
cu1(1.9375*pi) q[0],q[5];
h q[3];
cx q[4],q[0];
x q[3];
cu1(0.06249999999999997*pi) q[0],q[5];
h q[4];
h q[0];
x q[4];
h q[5];
x q[0];
h q[4];
t q[5];
cx q[1],q[5];
tdg q[5];
cx q[0],q[5];
t q[5];
cx q[1],q[5];
tdg q[5];
h q[5];
cx q[5],flag[0];
tdg flag[0];
cx q[2],flag[0];
t flag[0];
cx q[5],flag[0];
tdg flag[0];
h flag[0];
ccx q[3],flag[0],q[4];
h flag[0];
x q[3];
h q[4];
t flag[0];
h q[3];
x q[4];
cx q[5],flag[0];
h q[4];
tdg flag[0];
cx q[2],flag[0];
t flag[0];
x q[2];
cx q[5],flag[0];
h q[2];
tdg flag[0];
h q[5];
h flag[0];
t q[5];
h flag[0];
cx q[1],q[5];
t flag[0];
tdg q[5];
cx q[0],q[5];
x q[0];
t q[5];
h q[0];
cx q[1],q[5];
x q[1];
tdg q[5];
h q[1];
h q[5];
cu1(0.06249999999999997*pi) q[4],q[5];
cx q[4],q[3];
cu1(1.9375*pi) q[3],q[5];
cx q[4],q[3];
cu1(0.06249999999999997*pi) q[3],q[5];
cx q[3],q[2];
cu1(1.9375*pi) q[2],q[5];
cx q[4],q[2];
cu1(0.06249999999999997*pi) q[2],q[5];
cx q[3],q[2];
cu1(1.9375*pi) q[2],q[5];
cx q[4],q[2];
cu1(0.06249999999999997*pi) q[2],q[5];
cx q[2],q[1];
cu1(1.9375*pi) q[1],q[5];
cx q[4],q[1];
cu1(0.06249999999999997*pi) q[1],q[5];
cx q[3],q[1];
cu1(1.9375*pi) q[1],q[5];
cx q[4],q[1];
cu1(0.06249999999999997*pi) q[1],q[5];
cx q[2],q[1];
cu1(1.9375*pi) q[1],q[5];
cx q[4],q[1];
cu1(0.06249999999999997*pi) q[1],q[5];
cx q[3],q[1];
cu1(1.9375*pi) q[1],q[5];
cx q[4],q[1];
cu1(0.06249999999999997*pi) q[1],q[5];
cx q[1],q[0];
cu1(1.9375*pi) q[0],q[5];
cx q[4],q[0];
cu1(0.06249999999999997*pi) q[0],q[5];
cx q[3],q[0];
cu1(1.9375*pi) q[0],q[5];
cx q[4],q[0];
cu1(0.06249999999999997*pi) q[0],q[5];
cx q[2],q[0];
cu1(1.9375*pi) q[0],q[5];
cx q[4],q[0];
cu1(0.06249999999999997*pi) q[0],q[5];
cx q[3],q[0];
cu1(1.9375*pi) q[0],q[5];
cx q[4],q[0];
cu1(0.06249999999999997*pi) q[0],q[5];
cx q[1],q[0];
cu1(1.9375*pi) q[0],q[5];
h q[1];
cx q[4],q[0];
x q[1];
cu1(0.06249999999999997*pi) q[0],q[5];
cx q[3],q[0];
cu1(1.9375*pi) q[0],q[5];
cx q[4],q[0];
cu1(0.06249999999999997*pi) q[0],q[5];
cx q[2],q[0];
cu1(1.9375*pi) q[0],q[5];
h q[2];
cx q[4],q[0];
x q[2];
cu1(0.06249999999999997*pi) q[0],q[5];
cx q[3],q[0];
cu1(1.9375*pi) q[0],q[5];
h q[3];
cx q[4],q[0];
x q[3];
cu1(0.06249999999999997*pi) q[0],q[5];
h q[4];
h q[0];
x q[4];
h q[5];
x q[0];
h q[4];
t q[5];
cx q[1],q[5];
tdg q[5];
cx q[0],q[5];
t q[5];
cx q[1],q[5];
tdg q[5];
h q[5];
cx q[5],flag[0];
tdg flag[0];
cx q[2],flag[0];
t flag[0];
cx q[5],flag[0];
tdg flag[0];
h flag[0];
ccx q[3],flag[0],q[4];
h flag[0];
x q[3];
h q[4];
t flag[0];
h q[3];
x q[4];
cx q[5],flag[0];
h q[4];
tdg flag[0];
cx q[2],flag[0];
t flag[0];
x q[2];
cx q[5],flag[0];
h q[2];
tdg flag[0];
h q[5];
h flag[0];
t q[5];
h flag[0];
cx q[1],q[5];
t flag[0];
tdg q[5];
cx q[0],q[5];
x q[0];
t q[5];
h q[0];
cx q[1],q[5];
x q[1];
tdg q[5];
h q[1];
h q[5];
cu1(0.06249999999999997*pi) q[4],q[5];
cx q[4],q[3];
cu1(1.9375*pi) q[3],q[5];
cx q[4],q[3];
cu1(0.06249999999999997*pi) q[3],q[5];
cx q[3],q[2];
cu1(1.9375*pi) q[2],q[5];
cx q[4],q[2];
cu1(0.06249999999999997*pi) q[2],q[5];
cx q[3],q[2];
cu1(1.9375*pi) q[2],q[5];
cx q[4],q[2];
cu1(0.06249999999999997*pi) q[2],q[5];
cx q[2],q[1];
cu1(1.9375*pi) q[1],q[5];
cx q[4],q[1];
cu1(0.06249999999999997*pi) q[1],q[5];
cx q[3],q[1];
cu1(1.9375*pi) q[1],q[5];
cx q[4],q[1];
cu1(0.06249999999999997*pi) q[1],q[5];
cx q[2],q[1];
cu1(1.9375*pi) q[1],q[5];
cx q[4],q[1];
cu1(0.06249999999999997*pi) q[1],q[5];
cx q[3],q[1];
cu1(1.9375*pi) q[1],q[5];
cx q[4],q[1];
cu1(0.06249999999999997*pi) q[1],q[5];
cx q[1],q[0];
cu1(1.9375*pi) q[0],q[5];
cx q[4],q[0];
cu1(0.06249999999999997*pi) q[0],q[5];
cx q[3],q[0];
cu1(1.9375*pi) q[0],q[5];
cx q[4],q[0];
cu1(0.06249999999999997*pi) q[0],q[5];
cx q[2],q[0];
cu1(1.9375*pi) q[0],q[5];
cx q[4],q[0];
cu1(0.06249999999999997*pi) q[0],q[5];
cx q[3],q[0];
cu1(1.9375*pi) q[0],q[5];
cx q[4],q[0];
cu1(0.06249999999999997*pi) q[0],q[5];
cx q[1],q[0];
cu1(1.9375*pi) q[0],q[5];
h q[1];
cx q[4],q[0];
x q[1];
cu1(0.06249999999999997*pi) q[0],q[5];
cx q[3],q[0];
cu1(1.9375*pi) q[0],q[5];
cx q[4],q[0];
cu1(0.06249999999999997*pi) q[0],q[5];
cx q[2],q[0];
cu1(1.9375*pi) q[0],q[5];
h q[2];
cx q[4],q[0];
x q[2];
cu1(0.06249999999999997*pi) q[0],q[5];
cx q[3],q[0];
cu1(1.9375*pi) q[0],q[5];
h q[3];
cx q[4],q[0];
x q[3];
cu1(0.06249999999999997*pi) q[0],q[5];
h q[4];
h q[0];
x q[4];
h q[5];
x q[0];
h q[4];
t q[5];
cx q[1],q[5];
tdg q[5];
cx q[0],q[5];
t q[5];
cx q[1],q[5];
tdg q[5];
h q[5];
cx q[5],flag[0];
tdg flag[0];
cx q[2],flag[0];
t flag[0];
cx q[5],flag[0];
tdg flag[0];
h flag[0];
ccx q[3],flag[0],q[4];
h flag[0];
x q[3];
h q[4];
t flag[0];
h q[3];
x q[4];
cx q[5],flag[0];
h q[4];
tdg flag[0];
cx q[2],flag[0];
t flag[0];
x q[2];
cx q[5],flag[0];
h q[2];
tdg flag[0];
h q[5];
h flag[0];
t q[5];
h flag[0];
cx q[1],q[5];
t flag[0];
tdg q[5];
cx q[0],q[5];
x q[0];
t q[5];
h q[0];
cx q[1],q[5];
x q[1];
tdg q[5];
h q[1];
h q[5];
cu1(0.06249999999999997*pi) q[4],q[5];
cx q[4],q[3];
cu1(1.9375*pi) q[3],q[5];
cx q[4],q[3];
cu1(0.06249999999999997*pi) q[3],q[5];
cx q[3],q[2];
cu1(1.9375*pi) q[2],q[5];
cx q[4],q[2];
cu1(0.06249999999999997*pi) q[2],q[5];
cx q[3],q[2];
cu1(1.9375*pi) q[2],q[5];
cx q[4],q[2];
cu1(0.06249999999999997*pi) q[2],q[5];
cx q[2],q[1];
cu1(1.9375*pi) q[1],q[5];
cx q[4],q[1];
cu1(0.06249999999999997*pi) q[1],q[5];
cx q[3],q[1];
cu1(1.9375*pi) q[1],q[5];
cx q[4],q[1];
cu1(0.06249999999999997*pi) q[1],q[5];
cx q[2],q[1];
cu1(1.9375*pi) q[1],q[5];
cx q[4],q[1];
cu1(0.06249999999999997*pi) q[1],q[5];
cx q[3],q[1];
cu1(1.9375*pi) q[1],q[5];
cx q[4],q[1];
cu1(0.06249999999999997*pi) q[1],q[5];
cx q[1],q[0];
cu1(1.9375*pi) q[0],q[5];
cx q[4],q[0];
cu1(0.06249999999999997*pi) q[0],q[5];
cx q[3],q[0];
cu1(1.9375*pi) q[0],q[5];
cx q[4],q[0];
cu1(0.06249999999999997*pi) q[0],q[5];
cx q[2],q[0];
cu1(1.9375*pi) q[0],q[5];
cx q[4],q[0];
cu1(0.06249999999999997*pi) q[0],q[5];
cx q[3],q[0];
cu1(1.9375*pi) q[0],q[5];
cx q[4],q[0];
cu1(0.06249999999999997*pi) q[0],q[5];
cx q[1],q[0];
cu1(1.9375*pi) q[0],q[5];
h q[1];
cx q[4],q[0];
x q[1];
cu1(0.06249999999999997*pi) q[0],q[5];
cx q[3],q[0];
cu1(1.9375*pi) q[0],q[5];
cx q[4],q[0];
cu1(0.06249999999999997*pi) q[0],q[5];
cx q[2],q[0];
cu1(1.9375*pi) q[0],q[5];
h q[2];
cx q[4],q[0];
x q[2];
cu1(0.06249999999999997*pi) q[0],q[5];
cx q[3],q[0];
cu1(1.9375*pi) q[0],q[5];
h q[3];
cx q[4],q[0];
x q[3];
cu1(0.06249999999999997*pi) q[0],q[5];
h q[4];
h q[0];
x q[4];
h q[5];
x q[0];
h q[4];
t q[5];
cx q[1],q[5];
tdg q[5];
cx q[0],q[5];
t q[5];
cx q[1],q[5];
tdg q[5];
h q[5];
cx q[5],flag[0];
tdg flag[0];
cx q[2],flag[0];
t flag[0];
cx q[5],flag[0];
tdg flag[0];
h flag[0];
ccx q[3],flag[0],q[4];
h flag[0];
x q[3];
h q[4];
t flag[0];
h q[3];
x q[4];
cx q[5],flag[0];
h q[4];
tdg flag[0];
cx q[2],flag[0];
t flag[0];
x q[2];
cx q[5],flag[0];
h q[2];
tdg flag[0];
h q[5];
h flag[0];
t q[5];
cx q[1],q[5];
tdg q[5];
cx q[0],q[5];
x q[0];
t q[5];
h q[0];
cx q[1],q[5];
x q[1];
tdg q[5];
h q[1];
h q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5],flag[0];
measure flag[0] -> meas[6];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
measure q[5] -> meas[5];
