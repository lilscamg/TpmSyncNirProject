import numpy as np
from matplotlib import pyplot as plt

from Enums.TpmType import TpmType
from SyncProcess.start import start
from UpdateRules.update_rules import UpdateRules

'''Основные параметры ДМЧ'''
K, N = 5, 10
L = 5

tpm_types = [TpmType.QueriesNonBinary, TpmType.DefaultNonBinary, TpmType.QueriesBinary, TpmType.DefaultBinary]
tpm_types_names = [TpmType(tpm_type).name for tpm_type in tpm_types]

'''Доп параметры для вариаций алгоритма'''
H = 2.5  # размер локального поля
M = 6  # численный порог значений входного вектора

eve_attacks = False
upd_rule = UpdateRules.Hebbian
logs = False

plt.figure()
plt.title(f'Сравнение алгоритмов. K={K}, N={N}, L={L}, M={M}')
for tpm_type in tpm_types:
    result, eve_result, sync_weights, time_taken = start(tpm_type, K, N, L, eve_attacks=eve_attacks,
                                                         logs=logs, H=H, M=M, upd_rule=upd_rule)
    plt.plot(result["sync_history"])
    plt.xlabel('Количество итераций')
    plt.ylabel('Оценка')

plt.legend(tpm_types_names, loc="lower right")
plt.show()
