import numpy as np
from Enums.TpmType import TpmType


# Генерация входного вектора из {-1, 1} размерности [K, N]
def random_binary_input(K, N):
    return np.random.choice([-1, 1], size=[K, N])


# Генерация входного вектора из [-L, L] размерности [K, N]
def random_input(K, N, L):
    return np.random.randint(-L, L + 1, size=[K, N])


# Переопределенная функция sgn для массивов
def sgn(x):
    result = np.sign(x)
    return np.where(result == 0, 1, result)


# Переопределенная функция sgn для числа
def sgn_value(x):
    return 1 if x >= 0 else -1


# Функция оценки синхронизации весов двух ДМЧ
def sync_score(m1, m2, L):
    return 1.0 - np.average(1.0 * np.abs(m1.W - m2.W) / (2 * L))


# Расчет количества вхождений l в массив weights
def calc_n(weights, l):
    return (np.abs(weights) == l).sum()


# Получение параметров ДМЧ (use_queries, use_binary_inputs)
def get_tpm_params(tpm_type):
    if tpm_type is TpmType.DefaultBinary:
        return False, True
    elif tpm_type is TpmType.DefaultNonBinary:
        return False, False
    elif tpm_type is TpmType.QueriesBinary:
        return True, True
    elif tpm_type is TpmType.QueriesNonBinary:
        return True, False


# Генерация бинарного запроса размерности (K, N) на основе весов
def generate_bin_query(weights, H, K, N, L):
    query = np.zeros((K, N), dtype=int)

    # проход по всем скрытым нейронам
    for k in range(K):
        weights_k = weights[k]

        sigma_k = np.random.choice([-1, 1])
        h_k = sigma_k * H  # локальное поле для k-ого скрытого нейрона
        c_k = np.zeros(L + 1, dtype=int)  # количеcтва произведений весов и входных значений для k-ого скрытого нейрона

        for l in range(L, 0, -1):
            n_k_l = calc_n(weights_k, l)

            if np.random.rand() < 0.5:
                c_k[l] = int((n_k_l+1)/2 + 1/(2*L) * (h_k*np.sqrt(N) - np.sum(j * (2*c_k[j]-calc_n(weights_k, j)) for j in range(l + 1, L + 1))))
            else:
                c_k[l] = int((n_k_l-1)/2 + 1/(2*L) * (h_k*np.sqrt(N) - np.sum(j * (2*c_k[j]-calc_n(weights_k, j)) for j in range(l + 1, L + 1))))
            c_k[l] = max(0, min(c_k[l], n_k_l))

        for j in range(N):
            # веса со значением 0 не влияют на локальное поле
            if weights_k[j] == 0:
                query[k, j] = np.random.choice([-1, 1])
                continue

            l = np.abs(weights_k[j])
            n_k_l = calc_n(weights_k, l)
            if np.random.rand() < c_k[l] / n_k_l:
                query[k, j] = sgn_value(weights_k[j])
            else:
                query[k, j] = -sgn_value(weights_k[j])
    return query


# Генерация небинарного запроса размерности (K, N) на основе весов
def generate_nonbin_query(weights, H, K, N, L):
    query = generate_bin_query(weights, H, K, N, L)

    # Преобразуем матрицу в одномерный массив
    flattened_matrix = query.flatten()
    # Преобразуем элементы массива в битовое представление
    binary_representation = (flattened_matrix + 1) // 2  # -1 -> 0, 1 -> 1
    # Преобразуем битовое представление в целое число
    number = int(''.join(map(str, binary_representation)), 2) % (2**32 - 1)
    np.random.seed(number)
    return np.random.randint(-L, L + 1, size=(K, N))
