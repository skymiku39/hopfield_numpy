import numpy as np

# STEP1.生成 DATA
# DATA說明：生成100個由 -1、1 組成的數列(陣列)

# 初始化
array_shape = (10, 10)
data_count = 1
training_data = []

# 生成 10 組 10*10 二維陣列，存入List
for count in range(data_count):
    training_data.append(np.where(np.random.random_sample(array_shape) > 0.5, 1, -1))

# STEP2.計算權重
# 初始化
pixel_quantity = array_shape[0] * array_shape[1]
tij_weights = np.zeros((pixel_quantity, pixel_quantity))

# 張量積
# 將一組DATA的二維陣列轉成一維，分別轉換成行矩陣、列矩陣
# 兩矩陣做張量積存入tij權重，所有DATA執行一次
for i in range(data_count):
    oneArray_data = np.ravel(training_data[i])
    tij_weights = tij_weights + np.kron(oneArray_data, oneArray_data.reshape(pixel_quantity, 1))

# 因Tij中主對角線為0(Tii=0)，因此生成一主對角線為0其餘為1之矩陣
# 與Tij逐點乘積，得出權重
diagonalMatrix_rev = np.ones((pixel_quantity, pixel_quantity)) - np.eye(pixel_quantity)
tij_weights *= diagonalMatrix_rev

# STEP3.破壞DATA
# 初始化
# 計算破壞的pixel數量
break_pct = 0.2
break_count = int(pixel_quantity * break_pct)

# 隨機選擇DATA
twoArray_choice_data = training_data[np.random.randint(0, data_count)]
oneArray_choice_data = np.ravel(twoArray_choice_data)
oneArray_break_data = np.copy(oneArray_choice_data)

# 隨機選擇破壞位置，破壞DATA
choice_position = np.random.randint(pixel_quantity, size=break_count)
oneArray_break_data[choice_position] = np.where(oneArray_choice_data[choice_position] >= 1, -1, 1)
# 方便比較DATA
twoArray_break_data = np.reshape(oneArray_break_data, array_shape)

# STEP4.修復被破壞的DATA
# 初始化
oneArray_result = np.zeros(pixel_quantity)

# Tij 與 Xj 相乘後對每一列個別加總
for i in range(pixel_quantity):
    oneArray_result[i] = np.sum((tij_weights * oneArray_break_data)[i, :])
# for i in range(pixel_quantity):
#     for j in range(pixel_quantity):
#         oneArray_result[i] += tij_weights[i, j] * oneArray_break_data[j]
# 方便比較DATA
twoArray_result = np.reshape(oneArray_result, array_shape)

# 正規化
# > 0 -> = 1
# < 0 -> = -1
# ==0 -> = 原值 XOld
result = np.where(oneArray_result > 0, 1, np.where(
    oneArray_result < 0, -1, oneArray_break_data))
# oneArray_result[oneArray_result > 0] = 1
# oneArray_result[oneArray_result == 0] = oneArray_break_data
# oneArray_result[oneArray_result < 0] = -1


twoArray_result_data = np.reshape(result, array_shape)
