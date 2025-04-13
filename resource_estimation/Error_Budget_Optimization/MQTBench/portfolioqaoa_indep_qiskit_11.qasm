// Benchmark was created by MQT Bench on 2024-03-17
// For more information about MQT Bench, please visit https://www.cda.cit.tum.de/mqtbench/
// MQT Bench version: 1.1.0
// Qiskit version: 1.0.2

OPENQASM 2.0;
include "qelib1.inc";
qreg q[11];
creg meas[11];
creg meas8[11];
u2(-1.571050683239874,-pi) q[0];
u2(-1.499912735386558,-pi) q[1];
rzz(-4.72944170605565) q[0],q[1];
u2(-1.5655285820497635,-pi) q[2];
rzz(-4.730799456015475) q[0],q[2];
rzz(-4.730788016156666) q[1],q[2];
u2(-1.4446886421974077,-pi) q[3];
rzz(-4.7309250482264416) q[0],q[3];
rzz(-4.727040830875577) q[1],q[3];
rzz(-4.730610852210385) q[2],q[3];
u2(-1.5514260025951065,-pi) q[4];
rzz(-4.730506919898661) q[0],q[4];
rzz(-4.731119671633744) q[1],q[4];
rzz(-4.730776390858986) q[2],q[4];
rzz(-4.731091240559228) q[3],q[4];
u2(-1.448366877147258,-pi) q[5];
rzz(-4.731583357162163) q[0],q[5];
rzz(-4.729860019809269) q[1],q[5];
rzz(-4.731802266542036) q[2],q[5];
rzz(-4.732745777086017) q[3],q[5];
rzz(-4.730015446318286) q[4],q[5];
u2(-1.54233432352048,-pi) q[6];
rzz(-4.730743083178209) q[0],q[6];
rzz(-4.730240063803366) q[1],q[6];
rzz(-4.730835563010314) q[2],q[6];
rzz(-4.73086195380837) q[3],q[6];
rzz(-4.730706635039922) q[4],q[6];
rzz(-4.731340133837832) q[5],q[6];
u2(-1.571999627167994,-pi) q[7];
rzz(-4.731030489455035) q[0],q[7];
rzz(-4.730413033800714) q[1],q[7];
rzz(-4.73067780407356) q[2],q[7];
rzz(-4.731542974824649) q[3],q[7];
rzz(-4.730753615759774) q[4],q[7];
rzz(-4.729927412736728) q[5],q[7];
rzz(-4.7307831465639625) q[6],q[7];
u2(-1.529482331187558,-pi) q[8];
rzz(-4.729081121328583) q[0],q[8];
rzz(-4.729482145708308) q[1],q[8];
rzz(-4.730983469546935) q[2],q[8];
rzz(-4.73677483744752) q[3],q[8];
rzz(-4.73076017049151) q[4],q[8];
rzz(-4.724958375478925) q[5],q[8];
rzz(-4.73105890320335) q[6],q[8];
rzz(-4.732164678209279) q[7],q[8];
u2(-1.5567153306406842,-pi) q[9];
rzz(-4.730261728715132) q[0],q[9];
rzz(-4.730800801780218) q[1],q[9];
rzz(-4.730789826929389) q[2],q[9];
rzz(-4.730506152339172) q[3],q[9];
rzz(-4.730760689203675) q[4],q[9];
rzz(-4.731326605906959) q[5],q[9];
rzz(-4.730850706474157) q[6],q[9];
rzz(-4.730798171719479) q[7],q[9];
rzz(-4.730608581512352) q[8],q[9];
u2(-1.5151236613219157,-pi) q[10];
rzz(-4.731058470498585) q[0],q[10];
u3(0.41689189469197396,-1.4207031071880452,pi/2) q[0];
rzz(-4.730800191483786) q[1],q[10];
u3(0.41689189469197396,-1.3235814693555166,pi/2) q[1];
rzz(-6.456907155555261) q[0],q[1];
rzz(-4.731030977512123) q[2],q[10];
u3(0.41689189469197396,-1.413164015493043,pi/2) q[2];
rzz(-6.458760834271678) q[0],q[2];
rzz(-6.458745215915264) q[1],q[2];
rzz(-4.7313012304870545) q[3],q[10];
u3(0.416891894691974,-1.248186345652964,pi/2) q[3];
rzz(-6.458932300016701) q[0],q[3];
rzz(-6.453629342000659) q[1],q[3];
rzz(-6.458503341456598) q[2],q[3];
rzz(-4.730811760080542) q[4],q[10];
u3(0.41689189469197396,-1.3939103589618784,pi/2) q[4];
rzz(-6.458361447057857) q[0],q[4];
rzz(-6.459198011140625) q[1],q[4];
rzz(-6.458729344386987) q[2],q[4];
rzz(-6.45915919539452) q[3],q[4];
rzz(-4.732364227810258) q[5],q[10];
u3(0.4168918946919739,-1.2532080845903724,pi/2) q[5];
rzz(-6.459831061422763) q[0],q[5];
rzz(-6.457478261668178) q[1],q[5];
rzz(-6.46012992916005) q[2],q[5];
rzz(-6.461418064284959) q[3],q[5];
rzz(-6.457690458921175) q[4],q[5];
rzz(-4.730838809127155) q[6],q[10];
u3(0.41689189469197396,-1.3814978732349048,pi/2) q[6];
rzz(-6.45868387081617) q[0],q[6];
rzz(-6.457997119693592) q[1],q[6];
rzz(-6.458810129627839) q[2],q[6];
rzz(-6.458846159870603) q[3],q[6];
rzz(-6.458634109711258) q[4],q[6];
rzz(-6.459498998883276) q[5],q[6];
rzz(-4.73109618341829) q[7],q[10];
u3(0.41689189469197396,-1.4219986602830024,pi/2) q[7];
rzz(-6.459076254475961) q[0],q[7];
rzz(-6.458233268330785) q[1],q[7];
rzz(-6.458594748009692) q[2],q[7];
rzz(-6.459775929121689) q[3],q[7];
rzz(-6.458698250505254) q[4],q[7];
rzz(-6.457570270387724) q[5],q[7];
rzz(-6.4587385676658675) q[6],q[7];
rzz(-4.73195120294687) q[8],q[10];
u3(0.4168918946919739,-1.3639515905201116,pi/2) q[8];
rzz(-6.456414864445847) q[0],q[8];
rzz(-6.456962366106282) q[1],q[8];
rzz(-6.459012060179903) q[2],q[8];
rzz(-6.466918770350336) q[3],q[8];
rzz(-6.458707199403851) q[4],q[8];
rzz(-6.450786253537481) q[5],q[8];
rzz(-6.4591150465673355) q[6],q[8];
rzz(-6.460624714513781) q[7],q[8];
rzz(-4.730639745149279) q[9],q[10];
u3(0.4168918946919739,-1.401131655306822,pi/2) q[9];
rzz(-6.458026697883259) q[0],q[9];
rzz(-6.4587626715874835) q[1],q[9];
rzz(-6.45874768808684) q[2],q[9];
rzz(-6.458360399141281) q[3],q[9];
rzz(-6.458707907579678) q[4],q[9];
rzz(-6.459480529770202) q[5],q[9];
rzz(-6.458830804359895) q[6],q[9];
rzz(-6.4587590808765585) q[7],q[9];
rzz(-6.458500241368409) q[8],q[9];
u3(0.41689189469197396,-1.3443483045591047,pi/2) q[10];
rzz(-6.459114455813782) q[0],q[10];
u3(3.028679074999626,-2.8639651946234244,pi/2) q[0];
rzz(-6.458761838375523) q[1],q[10];
u3(3.028679074999626,-2.8834878498162,pi/2) q[1];
rzz(1.2979185156148147) q[0],q[1];
rzz(-6.459076920799705) q[2],q[10];
u3(3.028679074999626,-2.865480645669277,pi/2) q[2];
rzz(1.298291128054498) q[0],q[2];
rzz(1.2982879885709209) q[1],q[2];
rzz(-6.459445885780372) q[3],q[10];
u3(3.028679074999626,-2.898643205884224,pi/2) q[3];
rzz(1.298325594798389) q[0],q[3];
rzz(1.2972596343887481) q[1],q[3];
rzz(1.2982393688013016) q[2],q[3];
rzz(-6.4587776324924215) q[4],q[10];
u3(3.028679074999626,-2.8693508697104555,pi/2) q[4];
rzz(1.2982108462652093) q[0],q[4];
rzz(1.2983790060337121) q[1],q[4];
rzz(1.2982847981966557) q[2],q[4];
rzz(1.2983712035867583) q[3],q[4];
rzz(-6.460897151162023) q[5],q[10];
u3(3.028679074999626,-2.8976337740013705,pi/2) q[5];
rzz(1.2985062570011967) q[0],q[5];
rzz(1.2980333150351064) q[1],q[5];
rzz(1.2985663331275215) q[2],q[5];
rzz(1.2988252642827876) q[3],q[5];
rzz(1.2980759693178758) q[4],q[5];
rzz(-6.45881456141416) q[6],q[10];
u3(3.028679074999626,-2.8718459334711266,pi/2) q[6];
rzz(1.2982756574442624) q[0],q[6];
rzz(1.29813761194103) q[1],q[6];
rzz(1.2983010370332013) q[2],q[6];
rzz(1.2983082795593952) q[3],q[6];
rzz(1.2982656548442766) q[4],q[6];
rzz(1.2984395083074398) q[5],q[6];
rzz(-6.459165943671385) q[7],q[10];
u3(3.028679074999626,-2.863704772364067,pi/2) q[7];
rzz(1.2983545314322258) q[0],q[7];
rzz(1.2981850807494342) q[1],q[7];
rzz(1.298257742653496) q[2],q[7];
rzz(1.2984951747240865) q[3],q[7];
rzz(1.2982785479403252) q[4],q[7];
rzz(1.2980518099302352) q[5],q[7];
rzz(1.298286652190239) q[6],q[7];
rzz(-6.460333265747722) q[8],q[10];
u3(3.028679074999626,-2.8753729541742423,pi/2) q[8];
rzz(1.2978195589888966) q[0],q[8];
rzz(1.297929613621136) q[1],q[8];
rzz(1.2983416275815984) q[2],q[8];
rzz(1.2999309745058416) q[3],q[8];
rzz(1.2982803467800645) q[4],q[8];
rzz(1.2966881383041688) q[5],q[8];
rzz(1.2983623291242852) q[6],q[8];
rzz(1.2986657911275192) q[7],q[8];
rzz(-6.458542787766645) q[9],q[10];
u3(3.028679074999626,-2.8678992994761243,pi/2) q[9];
rzz(1.2981435575244344) q[0],q[9];
rzz(1.2982914973777893) q[1],q[9];
rzz(1.2982884855081052) q[2],q[9];
rzz(1.298210635620965) q[3],q[9];
rzz(1.2982804891321993) q[4],q[9];
rzz(1.2984357957863755) q[5],q[9];
rzz(1.298305192911069) q[6],q[9];
rzz(1.2982907756003064) q[7],q[9];
rzz(1.2982387456450808) q[8],q[9];
u3(3.028679074999626,-2.8793134580722723,pi/2) q[10];
rzz(1.2983622103754853) q[0],q[10];
rx(2.8797398619265353) q[0];
rzz(1.2982913298918388) q[1],q[10];
rx(2.8797398619265353) q[1];
rzz(1.2983546653715723) q[2],q[10];
rx(2.8797398619265353) q[2];
rzz(1.298428831914235) q[3],q[10];
rx(2.8797398619265353) q[3];
rzz(1.2982945047054557) q[4],q[10];
rx(2.8797398619265353) q[4];
rzz(1.2987205542767433) q[5],q[10];
rx(2.8797398619265353) q[5];
rzz(1.2983019278772827) q[6],q[10];
rx(2.8797398619265353) q[6];
rzz(1.2983725600742035) q[7],q[10];
rx(2.8797398619265353) q[7];
rzz(1.2986072063065348) q[8],q[10];
rx(2.8797398619265353) q[8];
rzz(1.2982472979994377) q[9],q[10];
rx(2.8797398619265353) q[9];
rx(2.8797398619265353) q[10];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10];
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
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10];
measure q[0] -> meas8[0];
measure q[1] -> meas8[1];
measure q[2] -> meas8[2];
measure q[3] -> meas8[3];
measure q[4] -> meas8[4];
measure q[5] -> meas8[5];
measure q[6] -> meas8[6];
measure q[7] -> meas8[7];
measure q[8] -> meas8[8];
measure q[9] -> meas8[9];
measure q[10] -> meas8[10];