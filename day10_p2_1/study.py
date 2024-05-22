# 2-1. 프로젝트 1 : 손수 설계하는 선형회귀, 당뇨병 수치를 맞춰보자!

# 평가기준
# 1. 프로젝트 1의 회귀모델 예측정확도가 기준 이상 높게 나왔는가?
# MSE 손실함수값 3000 이하를 달성
# 시각화 요구사항이 정확하게 이루어졌는가?
# 각 프로젝트 진행 과정에서 요구하고 있는 데이터개수 시각화 및 예측결과 시각화를 모두 진행하였다.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

def model(X, W, b):
    predictions = 0
    for i in range(10):
        predictions += X[:, i] * W[i]
    predictions += b
    return predictions

def MSE(a, b):
    mse = ((a - b) ** 2).mean()  # 두 값의 차이의 제곱의 평균
    return mse

def loss(X, W, b, y):
    predictions = model(X, W, b)
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

# 데이터 확인
# import sweetviz as sv
# db = load_diabetes(as_frame=True)
# print(db.frame)
# report = sv.analyze(db.frame)
# report.show_html("./day10_p2_1/eda.html")

# 데이터 가져오기 #numpy array로 변환된거 쓰는게 편해서 이걸 사용
diabetes=load_diabetes()

print(type(diabetes), "   /   ", diabetes.keys())
print(diabetes.data.shape, diabetes.target.shape)
print(diabetes.feature_names)
print(diabetes.DESCR)
print(diabetes.data_module)

# 프로젝트 진행
# 데이터 가져오기
df_x = diabetes.data
df_y = diabetes.target

print(df_x.shape, df_y.shape)

# 모델에 입력할 데이터 준비하기 numpy array로 변환
X = df_x
y = df_y

# 데이터 분할하기
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1004)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# 학습률
LEARNING_RATE = 0.01
# w, b 를 데이터 만큼 초기화
W = np.random.rand(10)
b = np.random.rand()

# 손실 함수 기록
losses = []

for i in range(1, 90001):
    dW, db = gradient(X_train, W, b, y_train)
    W -= LEARNING_RATE * dW
    b -= LEARNING_RATE * db
    L = loss(X_train, W, b, y_train)
    losses.append(L)
    if i % 10 == 0:
        print('Iteration %d : Loss %0.4f' % (i, L))

# 테스트 데이터에 대한 성능 확인
prediction = model(X_test, W, b)
mse = loss(X_test, W, b, y_test)
print("MSE:", mse)

# 시각화
plt.scatter(X_test[:, 0], y_test)
# plt.scatter(X_test[:, 0], prediction)
plt.savefig('./day10_p2_1/plot.png')


#