# Homework 3

## XAI 種類

這次使用的 pkg 為 `eli5`

解釋的方向分成：

1. 全局解釋
1. 局部解釋

因為 LR 本身就是一種有權重的模型，所以 LR 的全局解釋可以直接使用權重來解釋，這邊就不再贅述。至於局部解釋的話，這邊使用 `eli5` 這個套件來做解釋，這個套件背後使用的解釋方法是 LIME（Local Interpretable Model-agnostic Explanations）。

### 全局解釋 V.S. 局部解釋

> **全局解釋**：解釋整個模型的運作規則
> **局部解釋**：解釋單一預測的運作規則

例如：一個預測客戶會不會買保險的模型 $\hat{y} = \sigma(w_1 x_1 + w_2 x_2 + w_3 x_3 + \beta)$

- $x_1$：年齡（平均為 20 歲）
- $x_2$：收入（平均為 30000 元）
- $x_3$：是否有小孩（0, 1）

假設權重如下：

- $w_1 = 0.7$
- $w_2 = 0.1$
- $w_3 = 1.0$

從全局解釋來看：

- 這個模型的運作規則是：有小孩 > 年齡 > 收入

從局部解釋來看（計算程式請看 `example_cal.py`）：

1. Sample 1
   - $x_1 = 1$（20 歲）
   - $x_2 = 1$（30000）
   - $x_3 = 1$（有小孩）
   - $\hat{y} = 0.8581$

2. Sample 2
   - $x_1 = 3$（60 歲）
   - $x_2 = 3$（90000）
   - $x_3 = 0$（沒有小孩）
   - $\hat{y} = 0.9168$

從這兩個 sample 來看，雖然在全局解釋上是有小孩 > 年齡 > 收入，但在 Sample 2 的情況下，Sample 2 的年齡夠大，收入夠高，雖然沒有小孩，對最後的預測影響還是很大的。

因此，全局模型的解釋是有小孩 > 年齡 > 收入，但在 Sample 2 的情況下，局部解釋是年齡 > 收入 > 有小孩。

#### LIME（Local Interpretable Model-agnostic Explanations）

LIME 的運作原理是：

1. 提供一個 sample
2. 在 sample 的周圍隨機產生一些假 sample
3. 用這些假 sample 去預測，查看這些假 sample 的預測結果，究竟是哪些特徵影響了預測結果

## 程式碼

### `eli5` 提供全局解釋的功能，解釋的方式是用權重來解釋

```python
pred = eli5.explain_weights(model, feature_names=list(x_col))
print(eli5.format_as_dataframe(pred))
```

### `eli5` 提供局部解釋的功能，解釋的方式是用 LIME 來解釋

```python
pred = eli5.explain_prediction_sklearn(model, x_test[x_col].iloc[0], feature_names=list(x_col))
print(eli5.format_as_dataframe(pred))
```

### 解釋結果

### Train 1 全局解釋

```python
    target        feature    weight
0        1      PAY_0_DV4  1.990695
1        1      PAY_0_DV2  1.938724
2        1      PAY_0_DV3  1.898042
3        1      PAY_4_DV4  0.902812
4        1      PAY_0_DV5  0.863261
5        1      PAY_0_DV1  0.738016
6        1      PAY_6_DV3  0.637209
7        1      PAY_2_DV7  0.631865
8        1      PAY_0_DV8  0.631865
9        1      PAY_0_DV7  0.608415
10       1      PAY_2_DV5  0.587252
11       1      PAY_5_DV7  0.531920
12       1      PAY_2_DV6  0.515116
13       1     PAY_0_DV-1  0.474384
14       1      PAY_3_DV3  0.474249
15       1      PAY_4_DV5 -0.532001
16       1      PAY_2_DV4 -0.583516
17       1  EDUCATION_DV4 -0.628904
18       1  EDUCATION_DV5 -0.964954
19       1         <BIAS> -1.984627
```

### Train 1 局部解釋 - Sample 1

weights 跟 value 搭配起來看，正號的話，代表這個特徵對預測結果是正向影響（違約機率變高）；負號的話，代表這個特徵對預測結果是負向影響（違約機率變低）。

```python
    target        feature    weight     value
0        0         <BIAS>  1.984627  1.000000
1        0      PAY_0_DV0  0.343696  1.000000
2        0      PAY_6_DV0  0.295438  1.000000
3        0      PAY_4_DV0  0.052758  1.000000
4        0      BILL_AMT3  0.043442 -0.510931
5        0    BILL_AMT3.1  0.043442 -0.510931
6        0      BILL_AMT2  0.037805 -0.549609
7        0    BILL_AMT2.1  0.037805 -0.549609
8        0            AGE  0.027252 -1.137534
9        0          AGE.1  0.027252 -1.137534
10       0      PAY_5_DV0  0.007396  1.000000
11       0      PAY_2_DV0  0.005819  1.000000
12       0       PAY_AMT3  0.000483 -0.211606
13       0     PAY_AMT3.1  0.000483 -0.211606
14       0       PAY_AMT5 -0.001965 -0.215956
15       0     PAY_AMT5.1 -0.001965 -0.215956
16       0       PAY_AMT4 -0.002194 -0.212313
17       0     PAY_AMT4.1 -0.002194 -0.212313
18       0      BILL_AMT6 -0.002283 -0.403736
19       0    BILL_AMT6.1 -0.002283 -0.403736
20       0       PAY_AMT6 -0.005649 -0.180878
21       0     PAY_AMT6.1 -0.005649 -0.180878
22       0      BILL_AMT5 -0.009766 -0.437452
23       0    BILL_AMT5.1 -0.009766 -0.437452
24       0    BILL_AMT4.1 -0.010813 -0.476949
25       0      BILL_AMT4 -0.010813 -0.476949
26       0      BILL_AMT1 -0.017525 -0.575264
27       0    BILL_AMT1.1 -0.017525 -0.575264
28       0     PAY_AMT2.1 -0.019573 -0.170186
29       0       PAY_AMT2 -0.019573 -0.170186
30       0     PAY_AMT1.1 -0.019861 -0.251378
31       0       PAY_AMT1 -0.019861 -0.251378
32       0      PAY_3_DV0 -0.125244  1.000000
33       0    LIMIT_BAL.1 -0.139604 -1.059646
34       0      LIMIT_BAL -0.139604 -1.059646
35       0   MARRIAGE_DV2 -0.228088  1.000000
36       0  EDUCATION_DV2 -0.242259  1.000000
```

## 待想問題

1. 資料做正規化後，會不會讓可解釋性變得沒那麼直觀好懂

## 參考資料

- [[Day 4] LIME vs. SHAP：哪種XAI解釋方法更適合你？](https://andy6804tw.github.io/crazyai-xai/4.LIME%20vs%20SHAP:哪種XAI解釋方法更適合你/)
- [[Day 12] LIME理論：如何用局部線性近似解釋黑箱模型](https://andy6804tw.github.io/crazyai-xai/12.LIME理論:如何用局部線性近似解釋黑箱模型/)
