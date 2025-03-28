import math


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


# 第一筆資料：年齡 20、收入 30000、有小孩 (1)
z1 = 0.7 * 1 + 0.1 * 1 + 1 * 1
y_hat1 = sigmoid(z1)

# 第二筆資料：年齡 60、收入 90000、沒小孩 (0)
z2 = 0.7 * 3 + 0.1 * 3 + 1 * 0
y_hat2 = sigmoid(z2)

print(f"第一筆資料的 y_hat: {y_hat1:.4f}")
print(f"第二筆資料的 y_hat: {y_hat2:.4f}")
