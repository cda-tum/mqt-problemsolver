// Benchmark was created by MQT Bench on 2024-03-19
// For more information about MQT Bench, please visit https://www.cda.cit.tum.de/mqtbench/
// MQT Bench version: 1.1.0
// Qiskit version: 1.0.2

OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
creg meas[6];
ry(-1.6091521556564607) q[0];
ry(-1.568191406613579) q[1];
ry(-1.469942259085186) q[2];
ry(1.6246858840637424) q[3];
ry(1.687856548265116) q[4];
ry(-1.5644608391463715) q[5];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
cx q[0],q[1];
ry(-2.5384568520297575) q[0];
ry(-1.5762010081061046) q[1];
ry(-1.7151601595934756) q[2];
ry(1.2955637548649466) q[3];
ry(2.0692609481189055) q[4];
ry(-0.0030759346664153073) q[5];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
cx q[0],q[1];
ry(-0.4892811952799393) q[0];
ry(-0.7537309450701157) q[1];
ry(1.5153866838329422) q[2];
ry(-1.7229336573342435) q[3];
ry(-1.0725300977013172) q[4];
ry(0.017670641518661795) q[5];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
cx q[0],q[1];
ry(-2.049018501087617) q[0];
ry(0.41910807393610094) q[1];
ry(1.559432626994918) q[2];
ry(1.4194797279644502) q[3];
ry(1.3300726524968276) q[4];
ry(1.492714219983881) q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
measure q[5] -> meas[5];