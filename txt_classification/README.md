🔑 **PRT(Peer Review Template)**
> Reviewer : 이현동

- [ ]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요? (완성도)**
    > - 문제에서 요구하는 최종 결과물이 첨부되었는지 확인
    > - 문제를 해결하는 완성된 코드란 프로젝트 루브릭 3개 중 2개, 퀘스트 문제 요구조건 등을 지칭
        - 해당 조건을 만족하는 부분의 코드 및 결과물을 캡쳐하여 사진으로 첨부  

    - [x] 3가지 이상 모델을 성공적으로 시도  
        ![image](https://github.com/DevHDL/aiffel-L3earl/assets/163500244/5b14b03d-efd2-4a54-9b06-075bf3ed123f)  
    - [ ] gensim의 유사단어 찾기를 활용하여 자체학습한 임베딩과 사전학습 임베딩을 비교 분석  
            ```
            현재 사전학습 임베딩을 불러와 gensim으로 비교 분석을 시도하고 있었음.
            ```
    - [ ] 한국어 Word2Vec을 활용하여 정확도 85% 이상 달성  

- [x]  **2. 프로젝트에서 핵심적인 부분에 대한 설명이 주석(닥스트링) 및 마크다운 형태로 잘 기록되어있나요? (설명)**
    - [x]  모델 선정 이유  
          ![image](https://github.com/DevHDL/aiffel-L3earl/assets/163500244/6b084ae7-8826-483b-9ec9-5c5b59399d8f)  
    - [ ]  Metrics 선정 이유  
    - [ ]  Loss 선정 이유

    - [x] etc  
          > 이외에도 전처리, 모델 선정 이유 등 다양한 주석이 기록되어 있음
        ![image](https://github.com/DevHDL/aiffel-L3earl/assets/163500244/16974c21-5cd1-4fc3-b7c7-c456696c43b5)


- [ ]  **3. 체크리스트에 해당하는 항목들을 모두 수행하였나요? (문제 해결)**
    - [x]  데이터를 분할하여 프로젝트를 진행했나요? (train, validation, test 데이터로 구분)  
          ![image](https://github.com/DevHDL/aiffel-L3earl/assets/163500244/f2f87c51-4bfe-40fd-b9bf-f83c406e7777)  
    - [x]  하이퍼파라미터를 변경해가며 여러 시도를 했나요? (learning rate, dropout rate, unit, batch size, epoch 등)  
            ```
            여러 모델들을 활용했기에 하이퍼파라미터 역시 변경이 됨.
            ```
    - [ ]  각 실험을 시각화하여 비교하였나요?  
    - [x]  모든 실험 결과가 기록되었나요?  
          ![image](https://github.com/DevHDL/aiffel-L3earl/assets/163500244/c75aa019-8260-41f0-bbfb-7604a5ff644e)  

- [ ]  **4. 프로젝트에 대한 회고가 상세히 기록 되어 있나요? (회고, 정리)**
    - [ ]  배운 점
    - [ ]  아쉬운 점
    - [ ]  느낀 점
    - [ ]  어려웠던 점
