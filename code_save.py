import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sweetviz as sv # EDA 라이브러리
import missingno as msno # 결측치 시각화 라이브러리

# EDA report 생성
train = pd.read_csv('./0529_kaggle/data/train.csv')

sv.analyze(train).show_html("eda_train.html")

# 결측치 시각화, 결측치가 있으면 흰색으로 표시된다 함
msno.matrix(train)
plt.show()

# 전체 컬럼 라인 분포 보기
fig, ax = plt.subplots(9, 2, figsize=(12, 50))   # 가로스크롤 때문에 그래프 확인이 불편하다면 figsize의 x값을 조절해 보세요. 

# id 변수(count==0인 경우)는 제외하고 분포를 확인합니다.
# 에러나는건 나중에 
count = 1
columns = train.columns
for row in range(9):
    for col in range(2):
        sns.kdeplot(data=train[columns[count]], ax=ax[row][col])
        ax[row][col].set_title(columns[count], fontsize=15)
        count += 1
        if count == 19 :
            break