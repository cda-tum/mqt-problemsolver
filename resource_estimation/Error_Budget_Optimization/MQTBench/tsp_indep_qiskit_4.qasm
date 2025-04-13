// Benchmark was created by MQT Bench on 2024-03-19
// For more information about MQT Bench, please visit https://www.cda.cit.tum.de/mqtbench/
// MQT Bench version: 1.1.0
// Qiskit version: 1.0.2

OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg meas[4];
ry(0.3639515108210906) q[0];
ry(-3.635990261432437) q[1];
cz q[0],q[1];
ry(-1.002813228816168) q[0];
ry(3.362268080070584) q[2];
cz q[1],q[2];
ry(6.243016014063322) q[1];
cz q[0],q[1];
ry(0.8811055252112134) q[0];
ry(2.9808572300959044) q[3];
cz q[2],q[3];
ry(3.076180161567417) q[2];
cz q[1],q[2];
ry(3.5997755951020247) q[1];
cz q[0],q[1];
ry(-3.8705959422613034) q[0];
ry(-1.9599706708457578) q[3];
cz q[2],q[3];
ry(4.560604436086811) q[2];
cz q[1],q[2];
ry(3.836947786309806) q[1];
cz q[0],q[1];
ry(-3.6364557194733527) q[0];
ry(5.8061284709341265) q[3];
cz q[2],q[3];
ry(-3.8237047898941157) q[2];
cz q[1],q[2];
ry(5.382258257999492) q[1];
cz q[0],q[1];
ry(0.4608030009462126) q[0];
ry(5.991413266982324) q[3];
cz q[2],q[3];
ry(-3.1604705609650257) q[2];
cz q[1],q[2];
ry(-3.8085796950542656) q[1];
ry(6.4608361773846) q[3];
cz q[2],q[3];
ry(-2.709332149091332) q[2];
ry(1.6710555259169388) q[3];
barrier q[0],q[1],q[2],q[3];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];