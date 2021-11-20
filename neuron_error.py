import numpy as np
import collections as coll

# STEP5.深入研究！單個神經元儲存器發生錯誤的機率
# 以下為個人見解
# 名詞介紹： (數學符號：KaTeX)
# V = 神經元、 V_i = i個神經元排列(總共N個)、 V_j = 不等於V_i的神經元 (V_i\not=V_j)
# T_{ij} = Vi與Vj兩者間的關聯性量化，同時也是神經元儲存器
# V^s = 神經元的儲存空間的狀態集合(幾筆資料，1、2、3......n)
#
# 試問：Tij在相同的神經元數量N、不同的記憶資料筆數s下，對每個神經元Vi的記憶表現如何？
# 1. 每一個神經元Vi的錯誤機率如何？
# 2. 記憶表現與理論值差異多少？
# 補充說明：神經元儲存器 Tij，會隨著記憶的資料筆數s不同，記憶結果會產生改變
# 進行理論與實驗驗證，理論公式在 Hopfield paper，實驗使用"蒙地卡羅方法"驗證


# 實驗設計(自訂)：
# 1. 神經元數量 N = 100 (不要太小)
#    資料筆數 s = 1、2、3、...n (遞增值與結果自訂)
# 2. 生成 n 筆 N 個隨機 +-1 的 DATA
# 3. 計算不同 s 下需要用到的權重 Tij
#    補：本人使用的方式較吃記憶體(很小，可忽略)，可以不儲存權重一次算
# 4. 輸入原始 DATA 與 Tij 進行計算(回憶)，紀錄無法修復的神經元數量
# 5. 統計無法修復的神經元數量
# 6. 重新執行 2-5 數次(根據個人電腦情況)，以利統計分析

def batch_calc_neuron_error():
    # 初始化
    # 設置神經元數量
    neurons = 100
    # 最大的儲存資料筆數
    sto_cnt = 10
    # 遞增數、執行次數
    inc_val = 5
    inc_cnt = int(sto_cnt / inc_val)
    # 統計計數
    stats_cnt = 100
    # 方便辨識
    data_shape = (sto_cnt, neurons)

    # 統計每筆的神經元個數
    necrotic_neuron_cnt = []
    for i in range(inc_cnt):
        necrotic_neuron_cnt.append(coll.Counter())

    for loop in range(stats_cnt):
        # 當掉與否?
        print(loop)
        # 生成資料，參考STEP1.
        sto_data = np.where(np.random.random_sample(data_shape) > 0.5, 1, -1)

        # 計算權重，參考STEP2.
        wt_list = []
        # 建立 s 遞增迴圈
        for i in range(0, sto_cnt, inc_val):
            # 初始化計算用權重
            wt = np.zeros((neurons, neurons))

            # 計算 s 筆資料的張量積，s 由 i + inc_val 遞增
            for s in range(i + 1):
                wt += np.outer(sto_data[s, :], sto_data[s, :])
            # 主對角線為 0 (Tii=0)
            wt -= np.diag(np.diag(wt))
            # 儲存
            wt_list.append(wt)

        # 輸入原始DATA進行回憶，參考STEP4.
        # wt_list[i] = 目前權重
        # sto_data[s, :] = 目前資料 (要循環i+1次的)

        iter_cnt_list = []
        necrotic_neuron_list = []
        wt_cnt = 0
        for i in range(0, sto_cnt, inc_val):
            # 初始化 in/out
            in_data = np.copy(sto_data)
            out_data = np.copy(sto_data)
            # 初始化 統計變數
            # s 筆資料分別對 Tij 進行 dot 運算
            iter_cnt = np.zeros(i + 1)
            necrotic_neuron = []
            for s in range(i + 1):
                # 初始化迭代計數
                for iter_cnt[s] in range(100):
                    # in_data 對 wt_list(Tij) 進行 dot 運算
                    out_data[s, :] = np.dot(wt_list[wt_cnt], in_data[s, :])
                    # 正規化
                    out_data[s, :] = np.where(out_data[s, :] > 0, 1, np.where(
                        out_data[s, :] < 0, -1, in_data[s, :]))
                    # 判斷是否收斂，若收斂，跳下一筆資料
                    if np.all(out_data[s, :] == in_data[s, :]):
                        break
                    # 如果尚未收斂，目前的輸出會取代下次輸入(迭代)
                    in_data[s, :] = np.copy(out_data[s, :])
                # 如果未收斂，統計未修復的神經元
                necrotic_neuron.append(np.sum((out_data[s, :] == sto_data[s, :]) == 0))
            wt_cnt += 1

            # 儲存迭代計數
            iter_cnt_list.append(iter_cnt)
            # 儲存損壞的神經元個數
            necrotic_neuron_list.append(necrotic_neuron)
        # 統計
        for i in range(inc_cnt):
            necrotic_neuron_cnt[i] += coll.Counter(necrotic_neuron_list[i])
