import numpy as np

# STEP1.生成 DATA
# DATA說明：生成100個由 -1、1 組成的數列(陣列)

# 初始化
arr_shape = (10, 10)
data_sum = arr_shape[0] * arr_shape[1]
data_count = 10

# 生成 data_count = 10 組 arr_shape = 10*10 二維陣列，存入List
trng_data = []
for count in range(data_count):
    trng_data.append(np.where(np.random.random_sample(arr_shape) > 0.5, 1, -1))

# 生成 10 組 1*100 一維陣列，併成二維
# 因為計算過程的資料格式也是一維
# 故，生成與計算用一維(省下np.ravel)，顯示轉二維是較便捷的做法
# trng_data = np.where(np.random.random_sample((data_count, data_sum)) > 0.5, 1, -1)

# STEP2.計算權重
# 初始化
tij_wt = np.zeros((data_sum, data_sum))

# 張量積
# 將一組DATA的二維陣列轉成一維，分別轉換成行矩陣、列矩陣
# 兩矩陣做張量積存入tij權重，所有DATA執行一次
for i in range(data_count):
    oneArr_data = np.ravel(trng_data[i])

    # # 用kron，一維向量要轉置
    # tij_wt += np.kron(oneArr_data, oneArr_data.reshape(data_sum, 1))
    # 與上方等效，outer
    tij_wt += np.outer(oneArr_data, oneArr_data)

# 因Tij中主對角線為0(Tii=0)
# # 生成一主對角線為0其餘為1之矩陣與Tij逐點乘積，得出權重
# diag_mp_rev = np.ones((data_sum, data_sum)) - np.eye(data_sum)
# tij_wt *= diag_mp_rev
# 與上方等效原矩陣減去抓出的tij主對角線，得出權重
# diag 在做的事
# 1.內層，抓出原矩陣的主對角線數值，回傳一維資料
# 2.外層，將收到的一維的資料排在主對角線上，回傳二維資料
tij_wt -= np.diag(np.diag(tij_wt))

# STEP3.破壞DATA
# 初始化
# 計算破壞的pixel數量
break_pct = 20
break_count = int(data_sum * break_pct / 100)

# 隨機挑一個DATA
cho_num = np.random.randint(0, data_count)
cho_data = np.ravel(trng_data[cho_num])
corrupt_data = np.copy(cho_data)

# 根據破壞的資料數 break_count，抽選DATA的位置
rand_list = np.random.randint(data_sum, size=break_count)
# 依照對應的位置破壞DATA
corrupt_data[rand_list] = np.where(cho_data[rand_list] >= 1, -1, 1)

# STEP4.修復被破壞的DATA

# # 初始化
# recall_data = np.zeros(data_sum)
# # Tij 與 Xj 相乘後對每一列個別加總
# for i in range(data_sum):
#     recall_data = np.zeros(data_sum)
#     recall_data[i] = np.sum((tij_wt * corrupt_data)[i, :])
# 與上方內容等效
recall_data = np.dot(tij_wt, corrupt_data)

# 正規化
# > 0 -> = 1
# < 0 -> = -1
# ==0 -> = 原值 XOld

# repair_data[recall_data > 0] = 1
# repair_data[recall_data == 0] = recall_data
# repair_data[recall_data < 0] = -1
# 上方內容等效
repair_data = np.where(recall_data > 0, 1, np.where(
    recall_data < 0, -1, corrupt_data))

# 方便比較DATA
show_cho_data = np.reshape(cho_data, arr_shape)
show_corrupt_data = np.reshape(corrupt_data, arr_shape)
show_recall_data = np.reshape(recall_data, arr_shape)
show_repair_data = np.reshape(repair_data, arr_shape)
