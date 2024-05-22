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

diabetes=load_diabetes()

print(type(diabetes), diabetes.keys())
print(diabetes.data.shape, diabetes.target.shape)

diabetes.frame

