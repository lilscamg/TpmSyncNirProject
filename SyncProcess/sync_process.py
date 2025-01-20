import time

from TPM.tree_parity_machine import TreeParityMachine
from Utils.utils import random_binary_input, sync_score, random_input


def sync_process(K, N, L, update_rule, eve_attacks=True, use_binary_inputs=True, M=None):
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

    sync = False  # флаг синхронизации
    start_time = time.time()
    while not sync:
        X = random_binary_input(K=K, N=N) if use_binary_inputs else random_input(K=K, N=N, M=M)

        _, tauA = Alice.calc_tau(X)
        _, tauB = Bob.calc_tau(X)

        Alice.update(tauB, update_rule)
        Bob.update(tauA, update_rule)

        nb_updates += 1

        score = 100 * sync_score(Alice, Bob, L)  # Calculate the synchronization of the 2 machines
        sync_history.append(score)  # Add sync score to history, so that we can plot a graph later.

        if eve_attacks:
            _, tauE = Eve.calc_tau(X)
            # Ева обновляет веса, если ее выходы совпадают с выходами Алисы и Боба
            if tauA == tauB == tauE:
                Eve.update(tauA, update_rule)
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
