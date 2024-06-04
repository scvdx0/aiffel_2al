#%% 아이펠에서 머신러닝 모델이 어떻게 동작하는지 알 수 있도록, 모델 코드를 풀어서 설명한 코드를 적어둠

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# jupyter notebook 용, 실행한 브라우저에서 바로 그림을 볼 수 있게 해줌
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina' # 더 높은 해상도로 출력한다.

# 함수들
def model(X, W, b):
    predictions = 0
    for i in range(12):
        predictions += X[:, i] * W[i]
    predictions += b
    return predictions

def RMSE(a, b):
    mse = ((a - b) ** 2).mean()  # 두 값의 차이의 제곱의 평균
    rmse = mse ** 0.5        # MSE의 제곱근
    return rmse

def MSE(a, b):
    mse = ((a - b) ** 2).mean()  # 두 값의 차이의 제곱의 평균
    return mse

def loss(x, w, b, y):
    predictions = model(x, w, b)
    L = MSE(predictions, y)
    return L

def gradient(X, W, b, y):
    # N은 데이터 포인트의 개수
    N = len(y)

    # y_pred 준비
    y_pred = model(X, W, b)

    # 공식에 맞게 gradient 계산
    dW = 1 / N * 2 * X.T.dot(y_pred - y)

    # b의 gradient 계산
    db = 2 * (y_pred - y).mean()
    return dW, db

# 데이터 준비하기
sns.get_dataset_names()
tips = sns.load_dataset("tips")
print(tips.shape)
tips.head()

# 범주형 데이터 one-hot encoding
tips = pd.get_dummies(tips, columns=['sex', 'smoker', 'day', 'time'])
tips.head()

# 데이터 준비하기
X = tips[['total_bill', 'size', 'sex_Male', 'sex_Female', 'smoker_Yes', 'smoker_No',
          'day_Thur', 'day_Fri', 'day_Sat', 'day_Sun', 'time_Lunch', 'time_Dinner']].values
y = tips['tip'].values

# 데이터 분할하기
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 데이터 타입 변환 (분할 후)
X_train = X_train.astype(np.float64)
X_test = X_test.astype(np.float64)
y_train = y_train.astype(np.float64)
y_test = y_test.astype(np.float64)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# w, b 를 데이터 만큼 초기화
import numpy as np
W = np.random.rand(12)
b = np.random.rand()

# 학습률
LEARNING_RATE = 0.0001

# 손실 함수 기록
losses = []

for i in range(1, 1001):
    dW, db = gradient(X_train, W, b, y_train)
    W -= LEARNING_RATE * dW
    b -= LEARNING_RATE * db
    L = loss(X_train, W, b, y_train)
    losses.append(L)
    if i % 10 == 0:
        print('Iteration %d : Loss %0.4f' % (i, L))

prediction = model(X_test, W, b)
mse = loss(X_test, W, b, y_test)
print("MSE:", mse)

# 그래프 그리기
plt.scatter(X_test[:, 0], y_test)
plt.scatter(X_test[:, 0], prediction)
#plt.plot()
plt.savefig('plot.png')