import numpy as np

# STEP1.生成 DATA
# DATA說明：生成100個由 -1、1 組成的數列(陣列)

# 初始化
arr_shape = (10, 10)
data_count = 3
trng_data = []

# 生成 10 組 10*10 二維陣列，存入List
for count in range(data_count):
    trng_data.append(np.where(np.random.random_sample(arr_shape) > 0.5, 1, -1))

# STEP2.計算權重
# 初始化
data_sum = arr_shape[0] * arr_shape[1]
tij_wt = np.zeros((data_sum, data_sum))

# 張量積
# 將一組DATA的二維陣列轉成一維，分別轉換成行矩陣、列矩陣
# 兩矩陣做張量積存入tij權重，所有DATA執行一次
for i in range(data_count):
    oneArr_data = np.ravel(trng_data[i])
    tij_wt += np.kron(oneArr_data, oneArr_data.reshape(data_sum, 1))

# 因Tij中主對角線為0(Tii=0)，因此生成一主對角線為0其餘為1之矩陣
# 與Tij逐點乘積，得出權重
diag_mp_rev = np.ones((data_sum, data_sum)) - np.eye(data_sum)
tij_wt *= diag_mp_rev

# STEP3.破壞DATA
# 初始化
# 計算破壞的pixel數量
break_pct = 0.2
break_count = int(data_sum * break_pct)

# 隨機選擇DATA
cho_num = np.random.randint(0, data_count)
cho_data = np.ravel(trng_data[cho_num])
corrupt_data = np.copy(cho_data)

# 隨機選擇破壞位置，破壞DATA
rand_list = np.random.randint(data_sum, size=break_count)
corrupt_data[rand_list] = np.where(cho_data[rand_list] >= 1, -1, 1)
# 方便比較DATA
show_corrupt_data = np.reshape(corrupt_data, arr_shape)

# STEP4.修復被破壞的DATA
# 初始化
recall_data = np.zeros(data_sum)

# Tij 與 Xj 相乘後對每一列個別加總
for i in range(data_sum):
    recall_data[i] = np.sum((tij_wt * corrupt_data)[i, :])

# 方便比較DATA
show_recall_data = np.reshape(recall_data, arr_shape)

# 正規化
# > 0 -> = 1
# < 0 -> = -1
# ==0 -> = 原值 XOld
repair_data = np.where(recall_data > 0, 1, np.where(
    recall_data < 0, -1, corrupt_data))
# repair_data[recall_data > 0] = 1
# repair_data[recall_data == 0] = recall_data
# repair_data[recall_data < 0] = -1

show_repair_data = np.reshape(repair_data, arr_shape)
