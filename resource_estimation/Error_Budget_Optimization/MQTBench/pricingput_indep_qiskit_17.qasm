// Benchmark was created by MQT Bench on 2024-03-19
// For more information about MQT Bench, please visit https://www.cda.cit.tum.de/mqtbench/
// MQT Bench version: 1.1.0
// Qiskit version: 1.0.2

OPENQASM 2.0;
include "qelib1.inc";
qreg q[17];
creg meas[17];
ry(1.5765713111425983) q[0];
ry(1.582309018857911) q[1];
ry(1.5935292771638934) q[2];
ry(1.6140899847616945) q[3];
ry(1.6438454055481635) q[4];
ry(1.6611919395054304) q[5];
ry(1.6456790807969381) q[6];
ry(1.5132577997678178) q[7];
cx q[7],q[6];
ry(0.8541072460194206) q[6];
cx q[7],q[6];
cx q[6],q[5];
ry(0.2608149580033374) q[5];
cx q[7],q[5];
ry(0.11717965633363409) q[5];
cx q[6],q[5];
ry(0.5301791166053287) q[5];
cx q[7],q[5];
cx q[5],q[4];
ry(0.08104360060191351) q[4];
cx q[6],q[4];
ry(0.020012001043265504) q[4];
cx q[5],q[4];
ry(0.16147933412681503) q[4];
cx q[7],q[4];
ry(0.0831343389171055) q[4];
cx q[5],q[4];
ry(0.008771669551962208) q[4];
cx q[6],q[4];
ry(0.04097601255509495) q[4];
cx q[5],q[4];
ry(0.3029932533202989) q[4];
cx q[7],q[4];
cx q[4],q[3];
ry(0.022654896250005274) q[3];
cx q[5],q[3];
ry(0.0035065493768620115) q[3];
cx q[4],q[3];
ry(0.045042141657073986) q[3];
cx q[6],q[3];
ry(0.013696914413324296) q[3];
cx q[4],q[3];
ry(0.0011182692042165099) q[3];
cx q[5],q[3];
ry(0.0068789420142285895) q[3];
cx q[4],q[3];
ry(0.08782952500298738) q[3];
cx q[7],q[3];
ry(0.04773871371000188) q[3];
cx q[4],q[3];
ry(0.004088011156408972) q[3];
cx q[5],q[3];
ry(0.0006771520901946437) q[3];
cx q[4],q[3];
ry(0.008156349278967384) q[3];
cx q[6],q[3];
ry(0.02455986917443201) q[3];
cx q[4],q[3];
ry(0.002070238206548891) q[3];
cx q[5],q[3];
ry(0.012355959017209234) q[3];
cx q[4],q[3];
ry(0.15971324147727728) q[3];
cx q[7],q[3];
cx q[3],q[2];
ry(0.0058812799616279055) q[2];
cx q[4],q[2];
ry(0.0005033013085247173) q[2];
cx q[3],q[2];
ry(0.011736443963990428) q[2];
cx q[5],q[2];
ry(0.0019827921022166906) q[2];
cx q[3],q[2];
ry(0.00010269935370271766) q[2];
cx q[4],q[2];
ry(0.000994441347655639) q[2];
cx q[3],q[2];
ry(0.023268196170339206) q[2];
cx q[6],q[2];
ry(0.007482407830416843) q[2];
cx q[3],q[2];
ry(0.00038226346340952677) q[2];
cx q[4],q[2];
ry(4.553299390319676e-05) q[2];
cx q[3],q[2];
ry(0.00076167707869082) q[2];
cx q[5],q[2];
ry(0.0037865929359791606) q[2];
cx q[3],q[2];
ry(0.0001939861331175241) q[2];
cx q[4],q[2];
ry(0.0018990775964480566) q[2];
cx q[3],q[2];
ry(0.04503586725250658) q[2];
cx q[7],q[2];
ry(0.024871747647917324) q[2];
cx q[3],q[2];
ry(0.0012025349853346731) q[2];
cx q[4],q[2];
ry(0.00013833285958950864) q[2];
cx q[3],q[2];
ry(0.002396213285530864) q[2];
cx q[5],q[2];
ry(0.000542086004984059) q[2];
cx q[3],q[2];
ry(3.490754302187121e-05) q[2];
cx q[4],q[2];
ry(0.00027216459264121223) q[2];
cx q[3],q[2];
ry(0.004722820783543698) q[2];
cx q[6],q[2];
ry(0.012968905961824616) q[2];
cx q[3],q[2];
ry(0.0006356976881615947) q[2];
cx q[4],q[2];
ry(7.368688837620235e-05) q[2];
cx q[3],q[2];
ry(0.0012667345393641075) q[2];
cx q[5],q[2];
ry(0.006557869129176518) q[2];
cx q[3],q[2];
ry(0.0003225051948650068) q[2];
cx q[4],q[2];
ry(0.0032883142271762714) q[2];
cx q[3],q[2];
ry(0.08113056048600503) q[2];
cx q[7],q[2];
cx q[2],q[1];
ry(0.0014856810784273922) q[1];
cx q[3],q[1];
ry(6.555538256913795e-05) q[1];
cx q[2],q[1];
ry(0.0029695171860681935) q[1];
cx q[4],q[1];
ry(0.00026093054583226083) q[1];
cx q[2],q[1];
ry(7.325488031839633e-06) q[1];
cx q[3],q[1];
ry(0.00013059457851427592) q[1];
cx q[2],q[1];
ry(0.005924405143682365) q[1];
cx q[5],q[1];
ry(0.0010240159327806753) q[1];
cx q[2],q[1];
ry(2.8480147776066556e-05) q[1];
cx q[3],q[1];
ry(1.9869471996336596e-06) q[1];
cx q[2],q[1];
ry(5.6877280861206136e-05) q[1];
cx q[4],q[1];
ry(0.000513990981408613) q[1];
cx q[2],q[1];
ry(1.4322888106960141e-05) q[1];
cx q[3],q[1];
ry(0.0002572463179696949) q[1];
cx q[2],q[1];
ry(0.011735710313869496) q[1];
cx q[6],q[1];
ry(0.0038283366824546985) q[1];
cx q[2],q[1];
ry(0.0001031895250753409) q[1];
cx q[3],q[1];
ry(6.9612453846359945e-06) q[1];
cx q[2],q[1];
ry(0.0002060971473033732) q[1];
cx q[4],q[1];
ry(2.7587148058054456e-05) q[1];
cx q[2],q[1];
ry(1.114486990728969e-06) q[1];
cx q[3],q[1];
ry(1.3819412787434426e-05) q[1];
cx q[2],q[1];
ry(0.0004099705676689838) q[1];
cx q[5],q[1];
ry(0.00194155207761703) q[1];
cx q[2],q[1];
ry(5.2698893132287206e-05) q[1];
cx q[3],q[1];
ry(3.582603319195188e-06) q[1];
cx q[2],q[1];
ry(0.00010525164289971328) q[1];
cx q[4],q[1];
ry(0.0009743518384821639) q[1];
cx q[2],q[1];
ry(2.6495243993639173e-05) q[1];
cx q[3],q[1];
ry(0.0004876279991697166) q[1];
cx q[2],q[1];
ry(0.02266832888749081) q[1];
cx q[7],q[1];
ry(0.01257130028214936) q[1];
cx q[2],q[1];
ry(0.0003135656583474339) q[1];
cx q[3],q[1];
ry(1.95181273455608e-05) q[1];
cx q[2],q[1];
ry(0.0006263986132509433) q[1];
cx q[4],q[1];
ry(7.74433938269467e-05) q[1];
cx q[2],q[1];
ry(2.9002187562077175e-06) q[1];
cx q[3],q[1];
ry(3.878473604909671e-05) q[1];
cx q[2],q[1];
ry(0.0012470090203194825) q[1];
cx q[5],q[1];
ry(0.0003002909957535091) q[1];
cx q[2],q[1];
ry(1.1142751813783547e-05) q[1];
cx q[3],q[1];
ry(9.582838538246818e-07) q[1];
cx q[2],q[1];
ry(2.2239151318305395e-05) q[1];
cx q[4],q[1];
ry(0.00015110136508988264) q[1];
cx q[2],q[1];
ry(5.617598716312067e-06) q[1];
cx q[3],q[1];
ry(7.567197922149493e-05) q[1];
cx q[2],q[1];
ry(0.002449904387736075) q[1];
cx q[6],q[1];
ry(0.006578726657583266) q[1];
cx q[2],q[1];
ry(0.00016757938526271166) q[1];
cx q[3],q[1];
ry(1.068127480455261e-05) q[1];
cx q[2],q[1];
ry(0.0003347480333972183) q[1];
cx q[4],q[1];
ry(4.2364905481255544e-05) q[1];
cx q[2],q[1];
ry(1.6251699034713907e-06) q[1];
cx q[3],q[1];
ry(2.1218546250020842e-05) q[1];
cx q[2],q[1];
ry(0.0006662528346959504) q[1];
cx q[5],q[1];
ry(0.0033314438914753206) q[1];
cx q[2],q[1];
ry(8.540083480301774e-05) q[1];
cx q[3],q[1];
ry(5.483175785761518e-06) q[1];
cx q[2],q[1];
ry(0.00017058926842026395) q[1];
cx q[4],q[1];
ry(0.0016711956386644064) q[1];
cx q[2],q[1];
ry(4.291234975718761e-05) q[1];
cx q[3],q[1];
ry(0.0008362891461063393) q[1];
cx q[2],q[1];
ry(0.04073519783710192) q[1];
cx q[7],q[1];
cx q[1],q[0];
ry(0.00037241299307844145) q[0];
cx q[2],q[0];
ry(8.283709164946806e-06) q[0];
cx q[1],q[0];
ry(0.0007447069517185767) q[0];
cx q[3],q[0];
ry(3.30909809936053e-05) q[0];
cx q[1],q[0];
ry(4.7513678383404034e-07) q[0];
cx q[2],q[0];
ry(1.654987824148413e-05) q[0];
cx q[1],q[0];
ry(0.0014884640301892904) q[0];
cx q[4],q[0];
ry(0.00013167257674659427) q[0];
cx q[1],q[0];
ry(1.8848549069583331e-06) q[0];
cx q[2],q[0];
ry(6.931266521847945e-08) q[0];
cx q[1],q[0];
ry(3.7681350297247285e-06) q[0];
cx q[3],q[0];
ry(6.590555845839563e-05) q[0];
cx q[1],q[0];
ry(9.440009052508191e-07) q[0];
cx q[2],q[0];
ry(3.296146993087001e-05) q[0];
cx q[1],q[0];
ry(0.0029694043250572677) q[0];
cx q[5],q[0];
ry(0.0005162265996585397) q[0];
cx q[1],q[0];
ry(7.306594838760172e-06) q[0];
cx q[2],q[0];
ry(2.6494281374356277e-07) q[0];
cx q[1],q[0];
ry(1.4607266697497234e-05) q[0];
cx q[3],q[0];
ry(1.0566397838224217e-06) q[0];
cx q[1],q[0];
ry(2.359580671129624e-08) q[0];
cx q[2],q[0];
ry(5.286333052412218e-07) q[0];
cx q[1],q[0];
ry(2.9167380216715832e-05) q[0];
cx q[4],q[0];
ry(0.00025916744815572546) q[0];
cx q[1],q[0];
ry(3.676816859879417e-06) q[0];
cx q[2],q[0];
ry(1.3371845591914067e-07) q[0];
cx q[1],q[0];
ry(7.350633996011657e-06) q[0];
cx q[3],q[0];
ry(0.0001297173625280229) q[0];
cx q[1],q[0];
ry(1.8414056848631288e-06) q[0];
cx q[2],q[0];
ry(6.487544608541063e-05) q[0];
cx q[1],q[0];
ry(0.0058808439723883055) q[0];
cx q[6],q[0];
ry(0.0019253364994135137) q[0];
cx q[1],q[0];
ry(2.629682738255687e-05) q[0];
cx q[2],q[0];
ry(9.141306227360146e-07) q[0];
cx q[1],q[0];
ry(5.257414512304173e-05) q[0];
cx q[3],q[0];
ry(3.6466885777013125e-06) q[0];
cx q[1],q[0];
ry(7.775090237263638e-08) q[0];
cx q[2],q[0];
ry(1.8243284409420746e-06) q[0];
cx q[1],q[0];
ry(0.00010499290347988285) q[0];
cx q[4],q[0];
ry(1.443299405004192e-05) q[0];
cx q[1],q[0];
ry(3.065315490914655e-07) q[0];
cx q[2],q[0];
ry(1.5437643902799225e-08) q[0];
cx q[1],q[0];
ry(6.126136461274967e-07) q[0];
cx q[3],q[0];
ry(7.23191990098862e-06) q[0];
cx q[1],q[0];
ry(1.5371468780819852e-07) q[0];
cx q[2],q[0];
ry(3.6178988885825214e-06) q[0];
cx q[1],q[0];
ry(0.000208764149626426) q[0];
cx q[5],q[0];
ry(0.0009769805966448782) q[0];
cx q[1],q[0];
ry(1.345145317415905e-05) q[0];
cx q[2],q[0];
ry(4.722748540957056e-07) q[0];
cx q[1],q[0];
ry(2.689271034997734e-05) q[0];
cx q[3],q[0];
ry(1.8838956190557476e-06) q[0];
cx q[1],q[0];
ry(4.062974508570183e-08) q[0];
cx q[2],q[0];
ry(9.424686025667761e-07) q[0];
cx q[1],q[0];
ry(5.3704222894131876e-05) q[0];
cx q[4],q[0];
ry(0.0004903700522214081) q[0];
cx q[1],q[0];
ry(6.766233723410481e-06) q[0];
cx q[2],q[0];
ry(2.3821033862461705e-07) q[0];
cx q[1],q[0];
ry(1.3527308067261962e-05) q[0];
cx q[3],q[0];
ry(0.0002454231036835598) q[0];
cx q[1],q[0];
ry(3.388272286113797e-06) q[0];
cx q[2],q[0];
ry(0.00012274141112959477) q[0];
cx q[1],q[0];
ry(0.011353318651995573) q[0];
cx q[7],q[0];
ry(0.006302929143255991) q[0];
cx q[1],q[0];
ry(7.923038682090301e-05) q[0];
cx q[2],q[0];
ry(2.515453777955992e-06) q[0];
cx q[1],q[0];
ry(0.00015841180979550042) q[0];
cx q[3],q[0];
ry(1.0039216665003528e-05) q[0];
cx q[1],q[0];
ry(1.9524683192613523e-07) q[0];
cx q[2],q[0];
ry(5.021869670497961e-06) q[0];
cx q[1],q[0];
ry(0.0003164333692978142) q[0];
cx q[4],q[0];
ry(3.980235198218322e-05) q[0];
cx q[1],q[0];
ry(7.714914165273673e-07) q[0];
cx q[2],q[0];
ry(3.55729208225547e-08) q[0];
cx q[1],q[0];
ry(1.5420290971120826e-06) q[0];
cx q[3],q[0];
ry(1.9936719825417934e-05) q[0];
cx q[1],q[0];
ry(3.86698450425757e-07) q[0];
cx q[2],q[0];
ry(9.972824711683079e-06) q[0];
cx q[1],q[0];
ry(0.0006297902629391044) q[0];
cx q[5],q[0];
ry(0.0001539387003823018) q[0];
cx q[1],q[0];
ry(2.9474412454547766e-06) q[0];
cx q[2],q[0];
ry(1.3401904054057e-07) q[0];
cx q[1],q[0];
ry(5.8913441292624685e-06) q[0];
cx q[3],q[0];
ry(5.339652221717645e-07) q[0];
cx q[1],q[0];
ry(1.4082592427258622e-08) q[0];
cx q[2],q[0];
ry(2.671939146412644e-07) q[0];
cx q[1],q[0];
ry(1.175455140702171e-05) q[0];
cx q[4],q[0];
ry(7.750163745491416e-05) q[0];
cx q[1],q[0];
ry(1.4877469644274804e-06) q[0];
cx q[2],q[0];
ry(6.78495383054506e-08) q[0];
cx q[1],q[0];
ry(2.973696580822402e-06) q[0];
cx q[3],q[0];
ry(3.881861413956682e-05) q[0];
cx q[1],q[0];
ry(7.456690044021358e-07) q[0];
cx q[2],q[0];
ry(1.9417822099484713e-05) q[0];
cx q[1],q[0];
ry(0.0012362912729959093) q[0];
cx q[6],q[0];
ry(0.0033014326275069474) q[0];
cx q[1],q[0];
ry(4.245986665986877e-05) q[0];
cx q[2],q[0];
ry(1.385707865916505e-06) q[0];
cx q[1],q[0];
ry(8.489191154090713e-05) q[0];
cx q[3],q[0];
ry(5.52956377021345e-06) q[0];
cx q[1],q[0];
ry(1.1091758926433382e-07) q[0];
cx q[2],q[0];
ry(2.766109591717178e-06) q[0];
cx q[1],q[0];
ry(0.00016956213568060895) q[0];
cx q[4],q[0];
ry(2.191039502506993e-05) q[0];
cx q[1],q[0];
ry(4.3791368401803477e-07) q[0];
cx q[2],q[0];
ry(2.086208699715597e-08) q[0];
cx q[1],q[0];
ry(8.752490220798742e-07) q[0];
cx q[3],q[0];
ry(1.0976041372472503e-05) q[0];
cx q[1],q[0];
ry(2.1953454549562823e-07) q[0];
cx q[2],q[0];
ry(5.490639842150105e-06) q[0];
cx q[1],q[0];
ry(0.00033737836948429835) q[0];
cx q[5],q[0];
ry(0.0016724633419180745) q[0];
cx q[1],q[0];
ry(2.16633439330937e-05) q[0];
cx q[2],q[0];
ry(7.134338908861698e-07) q[0];
cx q[1],q[0];
ry(4.3312208515227865e-05) q[0];
cx q[3],q[0];
ry(2.8467470646522525e-06) q[0];
cx q[1],q[0];
ry(5.7720203441510853e-08) q[0];
cx q[2],q[0];
ry(1.4240728891719984e-06) q[0];
cx q[1],q[0];
ry(8.650905548571695e-05) q[0];
cx q[4],q[0];
ry(0.0008390728533808154) q[0];
cx q[1],q[0];
ry(1.0889235346702233e-05) q[0];
cx q[2],q[0];
ry(3.5950178394784926e-07) q[0];
cx q[1],q[0];
ry(2.1771152516116346e-05) q[0];
cx q[3],q[0];
ry(0.0004198957504853249) q[0];
cx q[1],q[0];
ry(5.451930803259189e-06) q[0];
cx q[2],q[0];
ry(0.00020999292422927036) q[0];
cx q[1],q[0];
ry(0.02038920687440985) q[0];
cx q[7],q[0];
ry(5*pi/8) q[8];
cry(-0.005387725081176802) q[0],q[8];
cry(-0.010775450162353603) q[1],q[8];
x q[1];
cry(-0.021550900324707207) q[2],q[8];
x q[2];
cry(-0.043101800649414414) q[3],q[8];
x q[3];
cry(-0.08620360129882883) q[4],q[8];
cry(-0.17240720259765765) q[5],q[8];
x q[5];
cry(-0.3448144051953153) q[6],q[8];
x q[6];
cry(-0.6896288103906306) q[7],q[8];
x q[10];
x q[11];
ccx q[1],q[10],q[11];
x q[11];
x q[12];
ccx q[2],q[11],q[12];
x q[12];
x q[13];
ccx q[3],q[12],q[13];
ccx q[4],q[13],q[14];
x q[14];
x q[15];
ccx q[5],q[14],q[15];
x q[15];
x q[16];
ccx q[6],q[15],q[16];
ccx q[7],q[16],q[9];
cx q[9],q[8];
u(pi/8,0,0) q[8];
cx q[9],q[8];
u3(pi/8,-pi,-pi) q[8];
cx q[9],q[8];
u(-0.0013469312702942004,0,0) q[8];
cx q[9],q[8];
u(0.0013469312702942004,0,0) q[8];
ccx q[9],q[0],q[8];
cx q[9],q[8];
u(0.0013469312702942004,0,0) q[8];
cx q[9],q[8];
u(-0.0013469312702942004,0,0) q[8];
ccx q[9],q[0],q[8];
cx q[9],q[8];
u(-0.002693862540588401,0,0) q[8];
cx q[9],q[8];
u(0.002693862540588401,0,0) q[8];
x q[16];
ccx q[6],q[15],q[16];
ccx q[5],q[14],q[15];
x q[5];
x q[6];
x q[14];
ccx q[4],q[13],q[14];
x q[13];
ccx q[3],q[12],q[13];
ccx q[2],q[11],q[12];
ccx q[1],q[10],q[11];
x q[1];
x q[2];
x q[3];
ccx q[9],q[1],q[8];
cx q[9],q[8];
u(0.002693862540588401,0,0) q[8];
cx q[9],q[8];
u(-0.002693862540588401,0,0) q[8];
ccx q[9],q[1],q[8];
x q[1];
ccx q[1],q[10],q[11];
cx q[9],q[8];
u(-0.005387725081176802,0,0) q[8];
cx q[9],q[8];
u(0.005387725081176802,0,0) q[8];
ccx q[9],q[2],q[8];
cx q[9],q[8];
u(0.005387725081176802,0,0) q[8];
cx q[9],q[8];
u(-0.005387725081176802,0,0) q[8];
ccx q[9],q[2],q[8];
x q[2];
ccx q[2],q[11],q[12];
cx q[9],q[8];
u(-0.010775450162353603,0,0) q[8];
cx q[9],q[8];
u(0.010775450162353603,0,0) q[8];
ccx q[9],q[3],q[8];
cx q[9],q[8];
u(0.010775450162353603,0,0) q[8];
cx q[9],q[8];
u(-0.010775450162353603,0,0) q[8];
ccx q[9],q[3],q[8];
x q[3];
ccx q[3],q[12],q[13];
cx q[9],q[8];
u(-0.021550900324707207,0,0) q[8];
cx q[9],q[8];
u(0.021550900324707207,0,0) q[8];
ccx q[9],q[4],q[8];
cx q[9],q[8];
u(0.021550900324707207,0,0) q[8];
cx q[9],q[8];
u(-0.021550900324707207,0,0) q[8];
ccx q[9],q[4],q[8];
cx q[9],q[8];
u(-0.043101800649414414,0,0) q[8];
cx q[9],q[8];
u(0.043101800649414414,0,0) q[8];
ccx q[9],q[5],q[8];
cx q[9],q[8];
u(0.043101800649414414,0,0) q[8];
cx q[9],q[8];
u(-0.043101800649414414,0,0) q[8];
ccx q[9],q[5],q[8];
x q[5];
cx q[9],q[8];
u(-0.08620360129882883,0,0) q[8];
cx q[9],q[8];
u(0.08620360129882883,0,0) q[8];
ccx q[9],q[6],q[8];
cx q[9],q[8];
u(0.08620360129882883,0,0) q[8];
cx q[9],q[8];
u(-0.08620360129882883,0,0) q[8];
ccx q[9],q[6],q[8];
x q[6];
cx q[9],q[8];
u(-0.17240720259765765,0,0) q[8];
cx q[9],q[8];
u(0.17240720259765765,0,0) q[8];
ccx q[9],q[7],q[8];
cx q[9],q[8];
u(0.17240720259765765,0,0) q[8];
cx q[9],q[8];
u(-0.17240720259765765,0,0) q[8];
ccx q[9],q[7],q[8];
x q[13];
ccx q[4],q[13],q[14];
x q[14];
ccx q[5],q[14],q[15];
ccx q[6],q[15],q[16];
x q[16];
ccx q[7],q[16],q[9];
ccx q[6],q[15],q[16];
x q[6];
x q[15];
ccx q[5],q[14],q[15];
x q[5];
x q[14];
ccx q[4],q[13],q[14];
ccx q[3],q[12],q[13];
x q[3];
x q[12];
ccx q[2],q[11],q[12];
x q[2];
x q[11];
ccx q[1],q[10],q[11];
x q[1];
x q[10];
x q[11];
x q[12];
x q[13];
x q[15];
x q[16];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16];
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
measure q[13] -> meas[13];
measure q[14] -> meas[14];
measure q[15] -> meas[15];
measure q[16] -> meas[16];