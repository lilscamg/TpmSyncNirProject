import matplotlib.pyplot as plt
from UpdateRules.update_rules import UpdateRules
from SyncProcess.sync_process import sync_process
from SyncProcess.sync_process_with_queries import sync_process_with_queries

K = 10
N = 100
L = 5
plot_results = False
eve_attacks = False
use_binary_inputs = False

# для обычного ДМЧ
upd_rule = UpdateRules.Hebbian
# для запросов
use_queries = True
H = 3

if use_queries:
    result, eve_result, sync_weights, time_taken = sync_process_with_queries(K, N, L, H, upd_rule, eve_attacks=eve_attacks)
else:
    result, eve_result, sync_weights, time_taken = sync_process(K, N, L, upd_rule, eve_attacks=eve_attacks, use_binary_inputs=use_binary_inputs)

print(f"Используются queries с H = {H}" if use_queries else "Обычный ДМЧ (Не используются queries в качестве входов)")
print(f"Параметры ДМЧ: K = {K}, N = {N}, L = {L}")
print(f"Вариаций весов: (2 * {L} + 1) ^ ({K} * {N}) = {(2 * L + 1) ** (K * N)}")
print(f"Для обновления весов используется правило {UpdateRules(upd_rule).name}")
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
