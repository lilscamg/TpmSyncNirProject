from matplotlib import pyplot as plt

from Enums.TpmType import TpmType
from SyncProcess.sync_process import sync_process
from SyncProcess.sync_process_with_queries import sync_process_with_queries
from UpdateRules.update_rules import UpdateRules
from Utils.utils import get_tpm_params


def start(tpm_type, K, N, L, eve_attacks=False, logs=False, H=None, M=None, upd_rule=None):
    use_queries, use_binary_inputs = get_tpm_params(tpm_type)

    if not use_queries and upd_rule is None:
        raise Exception('Default mode is chosen, but upd_rule value is None')
    if use_queries and H is None:
        raise Exception('Queries inputs mode is chosen, but H value is None')
    if use_binary_inputs is False and M is None:
        raise Exception('Binary inputs mode is chosen, but M value is None')

    if use_queries:
        return sync_process_with_queries(K, N, L, H, eve_attacks=eve_attacks,
                                         use_binary_inputs=use_binary_inputs,
                                         M=M,
                                         logs=logs)
    else:
        return sync_process(K, N, L, upd_rule, eve_attacks=eve_attacks,
                            use_binary_inputs=use_binary_inputs, M=M, logs=logs)


def start_and_show_results(tpm_type, K, N, L, eve_attacks=False, plot_results=False, logs=False, H=None, M=None,
                           upd_rule=None):
    use_queries, use_binary_inputs = get_tpm_params(tpm_type)

    if not use_queries and upd_rule is None:
        raise Exception('Default mode is chosen, but upd_rule value is None')
    if use_queries and H is None:
        raise Exception('Queries inputs mode is chosen, but H value is None')
    if use_binary_inputs is False and M is None:
        raise Exception('Binary inputs mode is chosen, but M value is None')

    if use_queries:
        result, eve_result, sync_weights, time_taken = sync_process_with_queries(K, N, L, H, eve_attacks=eve_attacks,
                                                                                 use_binary_inputs=use_binary_inputs,
                                                                                 M=M, logs=logs)
    else:
        result, eve_result, sync_weights, time_taken = sync_process(K, N, L, upd_rule, eve_attacks=eve_attacks,
                                                                    use_binary_inputs=use_binary_inputs, M=M, logs=logs)

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
            print(
                f"Машина Евы синхронизирована на {eve_result['eve_score']}% по отношению к Алисе и Бобу, она совершила {eve_result['nb_eve_updates']} обновлений весов")
            plt.plot(eve_result['eve_sync_history'], c='red')

    if plot_results:
        plt.show()