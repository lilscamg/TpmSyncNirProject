import matplotlib.pyplot as plt
from Enums.TpmType import TpmType
from UpdateRules.update_rules import UpdateRules
from SyncProcess.sync_process import sync_process
from SyncProcess.sync_process_with_queries import sync_process_with_queries
from Utils.utils import get_tpm_params

# основные параметры ДМЧ
K, N = 10, 100
L = 7

# доп параметры для вариаций алгоритма
H = 3  # размер локального поля
M = L  # численный порог значений входного вектора


tpm_type = TpmType.DefaultNonBinary  # QueriesNonBinary DefaultNonBinary
eve_attacks = True
upd_rule = UpdateRules.Hebbian
plot_results = False

use_queries, use_binary_inputs = get_tpm_params(tpm_type)
if use_queries:
    result, eve_result, sync_weights, time_taken = sync_process_with_queries(K, N, L, H, eve_attacks=eve_attacks, use_binary_inputs=use_binary_inputs, M=M)
else:
    result, eve_result, sync_weights, time_taken = sync_process(K, N, L, upd_rule, eve_attacks=eve_attacks, use_binary_inputs=use_binary_inputs, M=M)

print(f"{TpmType(tpm_type).name}")
print(f"Параметры ДМЧ: K = {K}, N = {N}, L = {L}")
if not use_queries:
    print(f"Для обновления весов используется правило {UpdateRules(upd_rule).name}")
else:
    print(f'H = {H}')
if not use_binary_inputs:
    print(f'M = {M}')
print(f"Вариаций весов: {2 * L + 1} ^ {K * N}")
print(f'Времени затрачено = {time_taken} секунд, итераций {result["nb_updates"]}')

plt.figure()
plt.plot(result["sync_history"], c='green')
plt.title('Динамика оценки синхронизации весовых коэффициентов двух ДМЧ')
plt.xlabel('Количество итераций')
plt.ylabel('Оценка')

if eve_attacks:
    if eve_result["eve_score"] > 100:
        print("Ева скомпрометировала ключ!")
    else:
        print(f"Машина Евы синхронизирована на {eve_result['eve_score']}% по отношению к Алисе и Бобу, она совершила {eve_result['nb_eve_updates']} обновлений весов")
        plt.plot(eve_result['eve_sync_history'], c='red')

if plot_results:
    plt.show()
