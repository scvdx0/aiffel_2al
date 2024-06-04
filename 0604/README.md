🔑 **PRT(Peer Review Template)**

- [x]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요? (완성도)**
    - 문제에서 요구하는 최종 결과물이 첨부되었는지 확인
    - 문제를 해결하는 완성된 코드란 프로젝트 루브릭 3개 중 2개, 
    퀘스트 문제 요구조건 등을 지칭
        - 해당 조건을 만족하는 부분의 코드 및 결과물을 캡쳐하여 사진으로 첨부
1. 자기만의 카메라앱 기능 구현을 완수하였다.
   ![image](https://github.com/L3earl/aiffel/assets/111371565/2f180def-75ea-4801-a37c-e2d7718c13a8)
   > 모델이 인식한 모든 얼굴에 대하여 스티커가 적용되었습니다.

2. 스티커 이미지를 정확한 원본 위치에 반영하였다.
```
# 코의 좌표와 관련된 정보 출력
for dlib_rect, landmarks, img in zip(dlib_rects, list_landmarks, img_show):
    if landmarks:  # 빈 리스트가 아닌 경우에만 실행
        for landmark in landmarks:
            nose_point = landmark[30] # 코의 index는 30 입니다
            x = nose_point[0] # 이미지에서 코 부위의 x값
            y = nose_point[1] - dlib_rect[0].height()//2 # 이미지에서 코 부위의 y값 - 얼굴 영역의 세로를 차지하는 픽셀의 수 // 2
            w = h = dlib_rect[0].width() # 얼굴 영역의 가로를 차지하는 픽셀의 수

            # print(f'코의 좌표: (x, y) = ({x}, {y})')
            # print(f'얼굴 영역의 크기: (w, h) = ({w}, {h})')

            img_sticker = cv2.resize(attach_img, (w,h)) # 스티커 이미지 조정 → w,h는 얼굴 영역의 가로를 차지하는 픽셀의 수(187) // cv2.resize(image객체 행렬, (가로 길이, 세로 길이))
            
            refined_x = x - w // 2 # 437 - (187//2) = 437-93 = 344
            refined_y = y  # 89-187 = -98
            
            # # 음수값 만큼 왕관 이미지(혹은 추후 적용할 스티커 이미지)를 자른다.
            if refined_x < 0: 
                img_sticker = img_sticker[:, -refined_x:]
                refined_x = 0
            # 왕관 이미지를 씌우기 위해 왕관 이미지가 시작할 y좌표 값 조정
            if refined_y < 0:
                img_sticker = img_sticker[-refined_y:, :] # refined_y가 -98이므로, img_sticker[98: , :]가 된다. (187, 187, 3)에서 (89, 187, 3)이 됨 (187개 중에서 98개가 잘려나감)
                refined_y = 0
            # print (f'(x,y) : ({refined_x},{refined_y})')

            sticker_area = img[refined_y:refined_y+img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]]
            
            img[refined_y:refined_y+img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]] = \
            np.where(img_sticker==255,sticker_area,img_sticker).astype(np.uint8)
```
> 모든 이미지에 대해 적절한 위치에 스티커가 붙을 수 있도록 반복문을 적절히 사용했습니다.
   
3. 카메라 스티커앱을 다양한 원본이미지에 적용했을 때의 문제점을 체계적으로 분석하였다.
> 얼굴의 각도에 따른 detection 능력을 실험해보고 싶다고 하셨는데 시간이 없어 구현하지 못한 점이 아쉽습니다. 다만 폴더 내의 이미지를 모두 불러오는 코드를 이미 작성하셨으므로 샘플이미지만 잘 모으면 바로 구현하실 수 있을 거라 생각합니다.
```
# 얼굴 이미지 파일 불러오기
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img_bgr = cv2.imread(img_path)
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # openCV가 이미지를 불러올때는 BGR로 받아오기 때문에, RGB로 변환
        if img is not None:
            images.append(img)
    return images

# 이미지 파일 경로
my_image_path = './images/face/'
attach_img = cv2.imread('./images/cat-whiskers.png')

fc_imgs_rgb = load_images_from_folder(my_image_path)
img_save = fc_imgs_rgb.copy()      # 출력용 이미지를 따로 보관합니다
```

- [X]  **2. 프로젝트에서 핵심적인 부분에 대한 설명이 주석(닥스트링) 및 마크다운 형태로 잘 기록되어있나요? (설명)**
      ![image](https://github.com/L3earl/aiffel/assets/111371565/f4827173-e1e9-4fe2-8b02-f1138a65d347)
    > 전체적인 방식에 대한 추가적인 마크다운이 작성되어 있습니다.

    - [ ]  모델 선정 이유
          > 모델의 한계점은 적어주셨지만 선정 이유는 따로 적혀있지 않았습니다
           ![image](https://github.com/L3earl/aiffel/assets/111371565/a7884292-6a39-4338-859c-cb684c4fcd39)
    - [ ]  Metrics 선정 이유
    - [ ]  Loss 선정 이유
          > 이미 있는 모델을 사용하므로 해당되지 않음

- [ ]  **3. 체크리스트에 해당하는 항목들을 모두 수행하였나요? (문제 해결)**
    - [ ]  데이터를 분할하여 프로젝트를 진행했나요? (train, validation, test 데이터로 구분)
    - [ ]  하이퍼파라미터를 변경해가며 여러 시도를 했나요? (learning rate, dropout rate, unit, batch size, epoch 등)
          > 이미 있는 모델을 사용하므로 해당되지 않음
    - [x]  각 실험을 시각화하여 비교하였나요?  
          > 이미지를subplt을 활용해 한 눈에 들어오도록 한 덕분에 비교가 간편했습니다
           ![image](https://github.com/L3earl/aiffel/assets/111371565/cd65b437-c4d4-4493-b0c7-ec42de2797a6)

    - [ ]  모든 실험 결과가 기록되었나요?
          > 실험에 대한 기록은 적혀있지 않음

- [x]  **4. 프로젝트에 대한 회고가 상세히 기록 되어 있나요? (회고, 정리)**
    - [x]  배운 점
    - [x]  아쉬운 점
    - [x]  느낀 점
    - [x]  어려웠던 점
