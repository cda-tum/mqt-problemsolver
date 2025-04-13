// Benchmark was created by MQT Bench on 2024-03-18
// For more information about MQT Bench, please visit https://www.cda.cit.tum.de/mqtbench/
// MQT Bench version: 1.1.0
// Qiskit version: 1.0.2

OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
creg meas[8];
ry(4.846350532897925) q[0];
ry(0.13038834331032634) q[1];
cx q[0],q[1];
ry(3.9813292796090525) q[2];
cx q[0],q[2];
cx q[1],q[2];
ry(4.704873552725635) q[3];
cx q[0],q[3];
cx q[1],q[3];
cx q[2],q[3];
ry(3.1322119352256292) q[4];
cx q[0],q[4];
cx q[1],q[4];
cx q[2],q[4];
cx q[3],q[4];
ry(1.4124389803026796) q[5];
cx q[0],q[5];
cx q[1],q[5];
cx q[2],q[5];
cx q[3],q[5];
cx q[4],q[5];
ry(1.2444656817555668) q[6];
cx q[0],q[6];
cx q[1],q[6];
cx q[2],q[6];
cx q[3],q[6];
cx q[4],q[6];
cx q[5],q[6];
ry(4.778555396547324) q[7];
cx q[0],q[7];
ry(1.0625547235745711) q[0];
cx q[1],q[7];
ry(0.5550554224571163) q[1];
cx q[0],q[1];
cx q[2],q[7];
ry(4.306242740899814) q[2];
cx q[0],q[2];
cx q[1],q[2];
cx q[3],q[7];
ry(5.990347064774806) q[3];
cx q[0],q[3];
cx q[1],q[3];
cx q[2],q[3];
cx q[4],q[7];
ry(0.02480768898038398) q[4];
cx q[0],q[4];
cx q[1],q[4];
cx q[2],q[4];
cx q[3],q[4];
cx q[5],q[7];
ry(3.2181989037565684) q[5];
cx q[0],q[5];
cx q[1],q[5];
cx q[2],q[5];
cx q[3],q[5];
cx q[4],q[5];
cx q[6],q[7];
ry(5.105848086558706) q[6];
cx q[0],q[6];
cx q[1],q[6];
cx q[2],q[6];
cx q[3],q[6];
cx q[4],q[6];
cx q[5],q[6];
ry(3.848614783366913) q[7];
cx q[0],q[7];
ry(4.534922405866221) q[0];
cx q[1],q[7];
ry(1.8339114230470697) q[1];
cx q[0],q[1];
cx q[2],q[7];
ry(5.766544881882964) q[2];
cx q[0],q[2];
cx q[1],q[2];
cx q[3],q[7];
ry(4.489812063110712) q[3];
cx q[0],q[3];
cx q[1],q[3];
cx q[2],q[3];
cx q[4],q[7];
ry(3.408906801581391) q[4];
cx q[0],q[4];
cx q[1],q[4];
cx q[2],q[4];
cx q[3],q[4];
cx q[5],q[7];
ry(0.8932807542109366) q[5];
cx q[0],q[5];
cx q[1],q[5];
cx q[2],q[5];
cx q[3],q[5];
cx q[4],q[5];
cx q[6],q[7];
ry(2.345769178126651) q[6];
cx q[0],q[6];
cx q[1],q[6];
cx q[2],q[6];
cx q[3],q[6];
cx q[4],q[6];
cx q[5],q[6];
ry(4.2357064252607195) q[7];
cx q[0],q[7];
ry(2.7761197097590844) q[0];
cx q[1],q[7];
ry(2.72699034602209) q[1];
cx q[2],q[7];
ry(3.8815444023791414) q[2];
cx q[3],q[7];
ry(3.2241426661697035) q[3];
cx q[4],q[7];
ry(4.086566017342803) q[4];
cx q[5],q[7];
ry(3.7764391210740293) q[5];
cx q[6],q[7];
ry(5.059366559339688) q[6];
ry(3.2776057234517526) q[7];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
measure q[5] -> meas[5];
measure q[6] -> meas[6];
measure q[7] -> meas[7];