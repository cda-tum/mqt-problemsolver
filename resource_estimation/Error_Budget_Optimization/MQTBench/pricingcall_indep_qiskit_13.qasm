// Benchmark was created by MQT Bench on 2024-03-19
// For more information about MQT Bench, please visit https://www.cda.cit.tum.de/mqtbench/
// MQT Bench version: 1.1.0
// Qiskit version: 1.0.2

OPENQASM 2.0;
include "qelib1.inc";
qreg q[13];
creg meas[13];
ry(1.5947651642342846) q[0];
ry(1.6163407297663968) q[1];
ry(1.6471480310867437) q[2];
ry(1.6641834847953176) q[3];
ry(1.647965453628736) q[4];
ry(1.5135407831050693) q[5];
cx q[5],q[4];
ry(0.8675203465795795) q[4];
cx q[5],q[4];
cx q[4],q[3];
ry(0.2651463631592207) q[3];
cx q[5],q[3];
ry(0.11948144070221331) q[3];
cx q[4],q[3];
ry(0.5408089121933445) q[3];
cx q[5],q[3];
cx q[3],q[2];
ry(0.08302290186802641) q[2];
cx q[4],q[2];
ry(0.02050393170389886) q[2];
cx q[3],q[2];
ry(0.16558102406055075) q[2];
cx q[5],q[2];
ry(0.08590759816983659) q[2];
cx q[3],q[2];
ry(0.008924784773706368) q[2];
cx q[4],q[2];
ry(0.04225215312000222) q[2];
cx q[3],q[2];
ry(0.3106106052351637) q[2];
cx q[5],q[2];
cx q[2],q[1];
ry(0.023352367884611147) q[1];
cx q[3],q[1];
ry(0.0036541454868429524) q[1];
cx q[2],q[1];
ry(0.04642649036298441) q[1];
cx q[4],q[1];
ry(0.014284617666040361) q[1];
cx q[2],q[1];
ry(0.0011717076460700249) q[1];
cx q[3],q[1];
ry(0.00717266122529478) q[1];
cx q[2],q[1];
ry(0.0904870543367701) q[1];
cx q[5],q[1];
ry(0.04968469189281871) q[1];
cx q[2],q[1];
ry(0.004297558515080208) q[1];
cx q[3],q[1];
ry(0.0007118603607449148) q[1];
cx q[2],q[1];
ry(0.008577993286537133) q[1];
cx q[4],q[1];
ry(0.02557000187782313) q[1];
cx q[2],q[1];
ry(0.002173500732966016) q[1];
cx q[3],q[1];
ry(0.012863799817036892) q[1];
cx q[2],q[1];
ry(0.16419764672772927) q[1];
cx q[5],q[1];
cx q[1],q[0];
ry(0.00607810382084828) q[0];
cx q[2],q[0];
ry(0.0005290492905115074) q[0];
cx q[1],q[0];
ry(0.012128273605703843) q[0];
cx q[3],q[0];
ry(0.0020831693400870377) q[0];
cx q[1],q[0];
ry(0.00010980751798170263) q[0];
cx q[2],q[0];
ry(0.001044891396071418) q[0];
cx q[1],q[0];
ry(0.024037705084336253) q[0];
cx q[4],q[0];
ry(0.00784601091180187) q[0];
cx q[1],q[0];
ry(0.0004078366023760882) q[0];
cx q[2],q[0];
ry(4.939240389767452e-05) q[0];
cx q[1],q[0];
ry(0.0008125410440799619) q[0];
cx q[3],q[0];
ry(0.003972240253410031) q[0];
cx q[1],q[0];
ry(0.00020705673196974583) q[0];
cx q[2],q[0];
ry(0.001992392092304285) q[0];
cx q[1],q[0];
ry(0.04647545488859184) q[0];
cx q[5],q[0];
ry(0.02594931292485137) q[0];
cx q[1],q[0];
ry(0.0012748476163882208) q[0];
cx q[2],q[0];
ry(0.0001490364404026709) q[0];
cx q[1],q[0];
ry(0.0025400076978466454) q[0];
cx q[3],q[0];
ry(0.000583690630582355) q[0];
cx q[1],q[0];
ry(3.8180208535819327e-05) q[0];
cx q[2],q[0];
ry(0.0002930870747028039) q[0];
cx q[1],q[0];
ry(0.0050038849326091775) q[0];
cx q[4],q[0];
ry(0.013548305692801361) q[0];
cx q[1],q[0];
ry(0.0006750727887607846) q[0];
cx q[2],q[0];
ry(7.953032958903788e-05) q[0];
cx q[1],q[0];
ry(0.0013450437080120767) q[0];
cx q[3],q[0];
ry(0.006853385389968679) q[0];
cx q[1],q[0];
ry(0.00034263435541070225) q[0];
cx q[2],q[0];
ry(0.0034368195918104485) q[0];
cx q[1],q[0];
ry(0.08349336314455161) q[0];
cx q[5],q[0];
ry(3*pi/8) q[6];
cry(0) q[0],q[6];
cry(0) q[1],q[6];
cry(0) q[2],q[6];
x q[2];
cry(0) q[3],q[6];
cry(0) q[4],q[6];
cry(0) q[5],q[6];
x q[5];
x q[7];
cx q[0],q[8];
ccx q[1],q[8],q[9];
x q[9];
x q[10];
ccx q[2],q[9],q[10];
ccx q[3],q[10],q[11];
ccx q[4],q[11],q[12];
x q[12];
ccx q[5],q[12],q[7];
x q[5];
cx q[7],q[6];
u(0.29425236476954253,0,0) q[6];
cx q[7],q[6];
u3(0.29425236476954253,-pi,-pi) q[6];
cx q[7],q[6];
u(-0.005451995606891006,0,0) q[6];
cx q[7],q[6];
u(0.005451995606891006,0,0) q[6];
x q[12];
ccx q[4],q[11],q[12];
ccx q[3],q[10],q[11];
x q[10];
ccx q[2],q[9],q[10];
x q[2];
x q[9];
ccx q[1],q[8],q[9];
cx q[0],q[8];
ccx q[7],q[0],q[6];
cx q[7],q[6];
u(0.005451995606891006,0,0) q[6];
cx q[7],q[6];
u(-0.005451995606891006,0,0) q[6];
ccx q[7],q[0],q[6];
cx q[0],q[8];
cx q[7],q[6];
u(-0.010903991213782011,0,0) q[6];
cx q[7],q[6];
u(0.010903991213782011,0,0) q[6];
ccx q[7],q[1],q[6];
cx q[7],q[6];
u(0.010903991213782011,0,0) q[6];
cx q[7],q[6];
u(-0.010903991213782011,0,0) q[6];
ccx q[7],q[1],q[6];
ccx q[1],q[8],q[9];
cx q[7],q[6];
u(-0.021807982427564022,0,0) q[6];
cx q[7],q[6];
u(0.021807982427564022,0,0) q[6];
ccx q[7],q[2],q[6];
cx q[7],q[6];
u(0.021807982427564022,0,0) q[6];
cx q[7],q[6];
u(-0.021807982427564022,0,0) q[6];
ccx q[7],q[2],q[6];
x q[2];
cx q[7],q[6];
u(-0.043615964855128045,0,0) q[6];
cx q[7],q[6];
u(0.043615964855128045,0,0) q[6];
ccx q[7],q[3],q[6];
cx q[7],q[6];
u(0.043615964855128045,0,0) q[6];
cx q[7],q[6];
u(-0.043615964855128045,0,0) q[6];
ccx q[7],q[3],q[6];
cx q[7],q[6];
u(-0.08723192971025609,0,0) q[6];
cx q[7],q[6];
u(0.08723192971025609,0,0) q[6];
ccx q[7],q[4],q[6];
cx q[7],q[6];
u(0.08723192971025609,0,0) q[6];
cx q[7],q[6];
u(-0.08723192971025609,0,0) q[6];
ccx q[7],q[4],q[6];
cx q[7],q[6];
u(-0.17446385942051218,0,0) q[6];
cx q[7],q[6];
u(0.17446385942051218,0,0) q[6];
ccx q[7],q[5],q[6];
cx q[7],q[6];
u(0.17446385942051218,0,0) q[6];
cx q[7],q[6];
u(-0.17446385942051218,0,0) q[6];
ccx q[7],q[5],q[6];
x q[5];
x q[9];
ccx q[2],q[9],q[10];
x q[10];
ccx q[3],q[10],q[11];
ccx q[4],q[11],q[12];
x q[12];
ccx q[5],q[12],q[7];
x q[5];
x q[7];
x q[12];
ccx q[4],q[11],q[12];
ccx q[3],q[10],q[11];
ccx q[2],q[9],q[10];
x q[2];
x q[9];
ccx q[1],q[8],q[9];
cx q[0],q[8];
x q[10];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
measure q[5] -> meas[5];
measure q[6] -> meas[6];
measure q[7] -> meas[7];
measure q[8] -> meas[8];
measure q[9] -> meas[9];
measure q[10] -> meas[10];
measure q[11] -> meas[11];
measure q[12] -> meas[12];