import numpy as np

np.random.seed(2020)
data = np.random.randn(100)  # 평균 0, 표준편차 1의 분포에서 100개의 숫자를 샘플링한 데이터 생성
data = np.concatenate((data, np.array([8, 10, -3, -5])))      # [8, 10, -3, -5])를 데이터 뒤에 추가함
data

Q3, Q1 = np.percentile(data, [75 ,25])
IQR = Q3 - Q1
IQR

data[(Q1-1.5*IQR > data)|(Q3+1.5*IQR < data)]

# Q. 사분위 범위수를 이용해서 이상치를 찾는 outlier2() 함수를 구현해보세요.
def outlier2(df, col):
    # [[YOUR CODE]]
    Q3, Q1 = np.percentile(df, [75 ,25])
    IQR = Q3 - Q1
    np.where(df[(Q1-1.5*IQR > df)|(Q3+1.5*IQR < df)])


outlier2(data)






outlier2(trade, '무역수지')


