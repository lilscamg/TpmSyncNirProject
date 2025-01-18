# TpmSyncNirProject
 
**Параметры:**\
K - число скрытых нейронов\
N - число входов у скрытых нейронов\
L - числовой порог для весовых коэффициентов\
W_A, W_B - весовые коэффициенты Алисы и Боба (размер KxN, значения [-L, L])\
X - общий входной вектор для ДМЧ Алисы и Боба (размер KxN, значения {-1, 1})\
tau_A, tau_B - выходные биты ДМЧ Алисы и Боба (значения {-1, 1})

**Что известно злоумышленнику:**\
K\
N\
L\
X\
tau_A, tau_b

**Что неизвестно злоумышленнику:**\
W_A, W_B