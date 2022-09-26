import matplotlib.pyplot as plt


load_model_time_including_gpu = [1.08128, 1.0492, 1.07145, 1.08119, 1.07726, 1.08424, 1.08515, 1.0744, 1.11702, 1.04176, 1.04966, 1.07441, 1.08739, 1.07814, 1.02391, 1.05715, 1.08487, 1.10191, 1.05018, 1.08671, 1.11531, 1.07106, 1.05263, 1.05777, 1.08985, 1.07186, 1.06457, 1.0122, 1.05255, 1.02078, 1.12653, 1.10948, 1.01771, 1.07434, 1.08179, 1.07222, 1.10458, 1.09366, 1.08837, 1.00326, 1.07682, 1.11145, 1.0678, 1.03039]
prediction_time_including_numpy_preprocess = [2.207, 0.00213556, 0.00146617, 0.00128793, 0.00130109, 0.00163412, 0.00127072, 0.00216934, 0.00267619, 0.00146945, 0.00132069, 0.00137391, 0.00157586, 0.0016277, 0.00214253, 0.00159231, 0.00159375, 0.00165529, 0.00154832, 0.00159753, 0.00157687, 0.0022343, 0.00286231, 0.00188309, 0.00163541, 0.00142207, 0.00140154, 0.00159819, 0.001512, 0.00160142, 0.00164461, 0.00203436, 0.00140466, 0.00154682, 0.00145621, 0.00146256, 0.00149032, 0.0015745, 0.00163358, 0.00238103, 0.00141765, 0.00226138, 0.00144937, 0.00165779, 0.00163434, 0.00192714, 0.00163024, 0.00185941, 0.00152718, 0.00149051, 0.00185892, 0.00202045, 0.00150073, 0.00160096, 0.00155912, 0.00137746, 0.00168696, 0.00140193, 0.00194995, 0.00131858, 0.00268477, 0.00163866, 0.00154365, 0.00142739, 0.00148467, 0.001409, 0.00236448, 0.0021282, 0.00144203, 0.00150643, 0.00163722, 0.00226918, 0.00172294, 0.00211582, 0.001931, 0.00167145, 0.00160453, 0.00147315, 0.00222071, 0.00137172, 0.00187285, 0.00155701, 0.00162234, 0.00168029, 0.00147222, 0.00180387, 0.00150011, 0.00153871, 0.00173878, 0.0014603, 0.0014972, 0.00164913, 0.00157854, 0.00178749, 0.00162135, 0.00236037, 0.00184581, 0.00144952, 0.00158223, 0.00159418, 0.00201456, 0.00153222, 0.00189207, 0.00145206, 0.00150668, 0.00175177, 0.00180822, 0.00155176, 0.00146725, 0.00131615, 0.00237712, 0.00160148, 0.00177738, 0.00139405, 0.00142153, 0.00154836, 0.00169635, 0.00181564, 0.00177088, 0.00152926, 0.00150474, 0.00148856, 0.00160365, 0.00152904, 0.00221751, 0.00151419, 0.00161523, 0.00215053, 0.00164208, 0.00144666, 0.00153813, 0.00228256, 0.00248445, 0.00157912, 0.00155355, 0.00165315, 0.0015803, 0.00217941, 0.00159253, 0.00228343, 0.00197421, 0.00196999, 0.00144699, 0.00147525, 0.00147054, 0.00156226, 0.00241917, 0.00151281, 0.00148914, 0.00148439, 0.00144962, 0.00158566, 0.00163069, 0.00164944, 0.00191995, 0.00202679, 0.00164695, 0.00144164, 0.00161257, 0.00161194, 0.00149263, 0.00154994, 0.00170316, 0.00195346, 0.00163687, 0.0015057, 0.00146444, 0.00149244, 0.00219646, 0.00179593, 0.00155389, 0.00150005, 0.00172955, 0.00168331, 0.00151879, 0.00207471, 0.002151, 0.00152527, 0.0015107, 0.00139855, 0.00144173, 0.00142559, 0.0016363, 0.00157902, 0.00160549, 0.00147043, 0.0014564, 0.00161139, 0.00153212, 0.00182418, 0.00144752, 0.00195389, 0.00165489, 0.00167316, 0.00149579, 0.00154525, 0.00148262, 0.00215447, 0.00169729, 0.00166686, 0.00156835, 0.00244625, 0.00185972, 0.00180695, 0.00158526, 0.00169998, 0.00198942, 0.00170056, 0.00195031, 0.00153267, 0.00191317, 0.00188268, 0.00165097, 0.00193111, 0.00180425, 0.00177601, 0.0015166, 0.00164878, 0.00241774, 0.00173096, 0.00166616, 0.00176148, 0.00155212, 0.00174226, 0.00162385, 0.00236407, 0.00206066, 0.00150335, 0.00155142, 0.00186497, 0.0016537, 0.00161967, 0.0018289, 0.00200102, 0.00159091, 0.00223877, 0.00133995, 0.00168853, 0.00172638, 0.0017465, 0.00163376, 0.00221815, 0.00191291, 0.00213939, 0.00142351, 0.00177336, 0.00168781, 0.00169061, 0.00163958, 0.00171343, 0.0013929, 0.00166281, 0.00178227, 0.00189254, 0.00163879, 0.00234624, 0.00180891, 0.00181262, 0.00239111, 0.00330742, 0.0015464, 0.00165163, 0.00251358, 0.0016512, 0.00157497, 0.00178876, 0.00169542, 0.00173559, 0.0018127, 0.00218203, 0.00165965, 0.00161647, 0.00162972, 0.00130818, 0.00177977, 0.00171378, 0.00198287, 0.00168156, 0.00155344, 0.00190452, 0.00269959, 0.00163199, 0.00148608, 0.00163316, 0.00165657, 0.00179588, 0.00168368, 0.00181669, 0.00164667, 0.00144039, 0.00170345, 0.00242377, 0.00173357, 0.00195074, 0.00187932, 0.00177221, 0.00205035, 0.00170662, 0.00188594, 0.00190418, 0.00179351, 0.00170235, 0.00176681, 0.00171093, 0.0016996, 0.00170927, 0.00243289, 0.00162851, 0.00185742, 0.00164966, 0.00167508, 0.00180523, 0.00178889, 0.00151794, 0.00193499, 0.00180897, 0.00179716, 0.00175932, 0.00181525, 0.00177889, 0.00219723, 0.00196533, 0.00171159, 0.00185819, 0.00182299, 0.00178841, 0.00171927, 0.00176425, 0.00174762, 0.00180948, 0.0024296, 0.00189643, 0.00194662, 0.0020268, 0.00175773, 0.00172136, 0.00176755, 0.00184577, 0.00272324, 0.00171654, 0.00177995, 0.00177849, 0.0024302, 0.00173626, 0.00169919, 0.00238291, 0.00202649, 0.00196142, 0.0018384, 0.00227169, 0.00227411, 0.00193927, 0.0016381, 0.00200411, 0.00170091, 0.00173671, 0.00170022, 0.00178658, 0.0018097, 0.00205566, 0.0017558, 0.00224035, 0.00184775, 0.00213726, 0.0017871, 0.00175099, 0.00196833, 0.00232523, 0.00179931, 0.00180871, 0.00184705, 0.00178667, 0.00157432, 0.00180719, 0.00159341, 0.00238286, 0.00180507, 0.00196955, 0.00168448, 0.00196167, 0.00200076, 0.00205609, 0.00230412, 0.00218659, 0.00212095, 0.00193161, 0.00208514, 0.00220106, 0.00204925, 0.00192335, 0.00247469, 0.00212672, 0.00484555, 0.00167773, 0.00176095, 0.00169993, 0.00188054, 0.00187569, 0.00312619, 0.00234478, 0.00166784, 0.00190388, 0.0018297, 0.00173037, 0.00196092, 0.00244681, 0.00189281, 0.00155773, 0.0017559, 0.0018856, 0.00201623, 0.00380974, 0.00192993, 0.0019667, 0.00181088, 0.00181536, 0.00308067, 0.00192897, 0.00240693, 0.00267688, 0.00223905, 0.00257962, 0.00199246, 0.00210293, 0.00185138, 0.00152142, 0.00179458, 0.0017637, 0.00182272, 0.00273073, 0.00185239, 0.00175958, 0.0016962, 0.00186683, 0.00187971, 0.00169032, 0.00186983, 0.00171442, 0.00167203, 0.00275803, 0.00182419, 0.00183504, 0.00249079, 0.00180885, 0.00173001, 0.00152274, 0.00196239, 0.00197077, 0.00200008, 0.00178105, 0.00187497, 0.00308696, 0.00186154, 0.00191989, 0.00166066, 0.0018679, 0.00173999, 0.00186406, 0.00174777, 0.00185916, 0.00185283, 0.00182915, 0.00176034, 0.00168161, 0.0019203, 0.00247537, 0.00191528, 0.00286919, 0.0018384, 0.00176859, 0.00181725, 0.00174271, 0.00252398, 0.00282665, 0.00183524, 0.00176104, 0.00181931, 0.00164788, 0.00183007, 0.00196797, 0.00166934, 0.0026577, 0.00296535]

n_samples_load_model = len(load_model_time_including_gpu)
n_samples_pred_time = len(prediction_time_including_numpy_preprocess)

average_load_model_time_including_gpu = sum(load_model_time_including_gpu) / n_samples_load_model
average_prediction_time_including_fisrt_val = sum(prediction_time_including_numpy_preprocess) / n_samples_pred_time
average_prediction_time_excluding_first_val = sum(prediction_time_including_numpy_preprocess[1:]) / ( n_samples_pred_time - 1)


print("Average Load Model Time", average_load_model_time_including_gpu)
print("Average Time Prediction including first value", average_prediction_time_including_fisrt_val)
print("Average Time Prediction excluding first value", average_prediction_time_excluding_first_val)

x = [i+1 for i in range(len(prediction_time_including_numpy_preprocess))]
x_load_model = [i+1 for i in range(len(load_model_time_including_gpu))]


#LOAD MODEL TIME
plt.figure(num=1)
plt.plot(x_load_model, load_model_time_including_gpu)
plt.title("Numpy Gefrorene Leistung neuronaler Netze auf einer x86-Prozessarchitektur in Python")
plt.xlabel("Proben")
plt.ylabel("Zeit in Sekunden")
plt.grid(linestyle='-')
plt.hlines(average_load_model_time_including_gpu, 1, n_samples_load_model, color='red')
plt.show()


#INCLUDING THE FIRST VALUE
plt.figure(num=2)
plt.plot(x, prediction_time_including_numpy_preprocess)
plt.title(" Gefrorene Leistung neuronaler Netze auf einer x86-Prozessarchitektur in Python")
plt.xlabel("Iteration")
plt.ylabel("Zeit in Sekunden")
plt.grid(linestyle='-')
plt.hlines(average_prediction_time_including_fisrt_val, 1, n_samples_pred_time, color='red')
plt.xlim(0, 200)
plt.show()

# EXCLUDING THE FIRST VALUE
plt.figure(num=3)
plt.plot(x[:-1], prediction_time_including_numpy_preprocess[1:])
plt.title(" Gefrorene Leistung neuronaler Netze auf einer x86-Prozessarchitektur in Python")
plt.xlabel("Iteration")
plt.ylabel("Zeit in Sekunden")
plt.grid(linestyle='-')
plt.hlines(average_prediction_time_excluding_first_val, 1, n_samples_pred_time-1, color='red')
plt.xlim(0, 200)
plt.show()
