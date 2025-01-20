import time
import numpy as np
from TPM.tree_parity_machine import TreeParityMachine
from Utils.utils import sync_score, generate_bin_query, generate_nonbin_query


def sync_process_with_queries(K, N, L, H, eve_attacks=True, use_binary_inputs=True, M=None):
    if use_binary_inputs is False and M is None:
        raise Exception('Binary inputs mode is chosen, but M value is None')

    Alice = TreeParityMachine(K=K, N=N, L=L)
    Bob = TreeParityMachine(K=K, N=N, L=L)
    score = 0  # оценка синхронизации у Alice и Bob
    nb_updates = 0  # количество обновлений весов у Alice и Bob
    sync_history = []

    Eve = None
    eve_score = None
    nb_eve_updates = None
    eve_sync_history = None
    eve_result = None
    if eve_attacks:
        Eve = TreeParityMachine(K=K, N=N, L=L)
        eve_score = 0  # оценка синхронизации у Eve с остальными двумя
        nb_eve_updates = 0  # количество обновлений весов у Eve
        eve_sync_history = []

    t = 0
    sync = False  # флаг синхронизации
    start_time = time.time()
    while not sync:
        if t % 2 == 0:
            query = generate_bin_query(weights=Alice.W, H=H, K=K, N=N, L=L) if use_binary_inputs else generate_nonbin_query(weights=Alice.W, H=H, K=K, N=N, L=L, M=M)
        else:
            query = generate_bin_query(weights=Bob.W, H=H, K=K, N=N, L=L) if use_binary_inputs else generate_nonbin_query(weights=Bob.W, H=H, K=K, N=N, L=L, M=M)
        t += 1

        sigma_A, tau_A = Alice.calc_tau(query)
        sigma_B, tau_B = Bob.calc_tau(query)

        # если выходы не совпадают, идем дальше
        if tau_A != tau_B:
            continue

        # обновление весов
        for k in range(K):
            if sigma_A[k] == tau_A:
                Alice.W[k] += tau_A * query[k]
            if sigma_B[k] == tau_B:
                Bob.W[k] += tau_B * query[k]
        Alice.W = np.clip(Alice.W, -L, L)
        Bob.W = np.clip(Bob.W, -L, L)

        nb_updates += 1

        score = 100 * sync_score(Alice, Bob, L)  # Calculate the synchronization of the 2 machines
        sync_history.append(score)  # Add sync score to history, so that we can plot a graph later.

        if eve_attacks:
            sigma_E, tau_E = Eve.calc_tau(query)

            # Ева обновляет веса, если ее выходы совпадают с выходами Алисы и Боба
            if tau_A == tau_B == tau_E:
                # обновление весов Евы
                for k in range(K):
                    if sigma_E[k] == tau_E:
                        Eve.W[k] += tau_E * query[k]
                Eve.W = np.clip(Eve.W, -L, L)
                nb_eve_updates += 1
                eve_score = 100 * sync_score(Alice, Eve, L)
            eve_sync_history.append(eve_score)
            if eve_score >= 100:
                break

        print(f"Синхронизация = {int(score)}%, Итераций = {nb_updates}", f" Итераций с обновлениями у Евы = {nb_eve_updates}" if eve_attacks else "")

        if score >= 100:
            sync = True
            print('Синхронизация завершена\n')

    end_time = time.time()
    time_taken = end_time - start_time

    result = {
        "score": score,
        "nb_updates": nb_updates,
        "sync_history": sync_history
    }
    weights = {
        "alice": Alice.W,
        "bob": Bob.W
    }
    if eve_attacks:
        eve_result = {
            "eve_score": eve_score,
            "nb_eve_updates": nb_eve_updates,
            "eve_sync_history": eve_sync_history
        }
        weights["eve"] = Eve.W

    return result, eve_result, weights, time_taken
