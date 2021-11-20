import numpy as np


def hopfield_test():
    # def generate_data():
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

    # def calculate_weight():
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

    # def destroy_data():
    # STEP3.破壞DATA
    # 初始化
    # 計算破壞的pixel數量
    break_pct = 20
    break_count = int(data_sum * break_pct / 100)

    # 隨機挑DATA的寫法不能做統計，因此作廢，保留抽選的寫法
    # # 隨機挑一個DATA
    # cho_num = np.random.randint(0, data_count)
    # cho_data = np.ravel(trng_data[cho_num])
    # corrupt_data = np.copy(cho_data)
    # # 根據破壞的資料數 break_count，抽選DATA的位置
    # rand_list = np.random.randint(data_sum, size=break_count)
    # # 依照對應的位置破壞DATA
    # corrupt_data[rand_list] = np.where(cho_data[rand_list] >= 1, -1, 1)

    # 將所有的DATA，按照break_pct的比例進行破壞
    # 複製原始DATA，並用一維排列
    all_data = np.reshape(trng_data, (data_count, data_sum))
    break_data = np.copy(all_data)
    # rand_list = np.zeros(break_data.shape)
    # 根據破壞的資料數 break_count，抽選DATA的位置
    for i in range(data_count):
        # 抽選位置 (不重複)
        rand_list = np.random.choice(data_sum, size=break_count, replace=False)
        # 對選擇的位置進行破壞
        break_data[i, rand_list] = np.where(break_data[i, rand_list] >= 1, -1, 1)

    # def repair_data():
    # STEP4.修復被破壞的DATA
    # 說明方法：透過 STEP2.的計算，Tij 已經儲存著每一筆資料的記憶
    # trng_data[n] = 記憶、data_count = n = 資料筆數，
    # 輸入 Xj (回憶) ，可以從 Tij (記憶庫) 提取儲存的記憶 <- 個人理解
    # 根據數學推導，Tij 與 Xj 相乘後對每一列個別加總，會得出一串無意義的數值
    # 將這些數值進行分析(正規化)，可以得出接近正確的回憶內容
    # 回憶的次數可能需要超過一次以上，因此需要迭代與收斂

    # 初始化
    input_repair_data = np.copy(break_data)
    output_recall_data = np.copy(break_data)
    loop_count = np.zeros(data_count)

    # 將每一筆資料分別對Tij進行dot運算
    for i in range(data_count):
        # 迭代若無法收斂，一定次數後跳出迴圈
        for loop_count[i] in range(100):
            # 檢查迭代狀態用
            # print("NO", i, "Loop", loop_count[i])
            output_recall_data[i, :] = np.dot(tij_wt, input_repair_data[i, :])
            # 正規化
            output_recall_data[i, :] = np.where(output_recall_data[i, :] > 0, 1, np.where(
                output_recall_data[i, :] < 0, -1, input_repair_data[i, :]))
            # 判斷是否收斂，若收斂，跳下一筆資料
            if np.all(output_recall_data[i, :] == input_repair_data[i, :]):
                break
            # 如果尚未收斂，目前的輸出會取代下次輸入(迭代)
            input_repair_data[i, :] = np.copy(output_recall_data[i, :])

    # 已使用np.dot為最終計算寫法，以下保留撰寫過程
    # # recall_data = np.zeros(data_sum)
    # # # Tij 與 Xj 相乘後對每一列個別加總
    # # for i in range(data_sum):
    # #     recall_data = np.zeros(data_sum)
    # #     recall_data[i] = np.sum((tij_wt * corrupt_data)[i, :])
    # # 與上方內容等效
    # recall_data = np.dot(tij_wt, corrupt_data)

    # 已使用np.where 為最終正規化寫法，以下保留撰寫過程
    # # 正規化
    # # out[x] > 0 -> = 1 ；out[x] < 0 -> = -1 ； out[x] ==0 -> = input[x]
    # # repair_data[recall_data > 0] = 1
    # # repair_data[recall_data == 0] = recall_data
    # # repair_data[recall_data < 0] = -1
    # # 上方內容等效
    # repair_data = np.where(recall_data > 0, 1, np.where(
    #     recall_data < 0, -1, corrupt_data))
