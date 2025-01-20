from Enums.TpmType import TpmType
from SyncProcess.start import start_and_show_results
from UpdateRules.update_rules import UpdateRules

# основные параметры ДМЧ
# K, N = 10, 100
# L = 7
K, N = 4, 5
L = 5

# доп параметры для вариаций алгоритма
H = 2.5  # размер локального поля
M = 7  # численный порог значений входного вектора

tpm_type = TpmType.QueriesNonBinary  # QueriesNonBinary (мой), DefaultNonBinary, QueriesBinary, DefaultBinary
eve_attacks = False
upd_rule = UpdateRules.Hebbian
plot_results = False
logs = True

start_and_show_results(tpm_type=tpm_type, K=K, N=N, L=L, eve_attacks=eve_attacks, plot_results=plot_results, H=H, M=M,
                       upd_rule=upd_rule, logs=logs)
