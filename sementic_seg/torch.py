import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet101

# GPU 사용 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 이미지 파일 경로
img_path = './sementic_seg/obj_imgs/pexels-esrannuur-129682465-14458837.jpg'

# 임의의 색상 생성 함수
def generate_random_colors(num_colors):
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(num_colors, 3), dtype=np.uint8)
    return colors

# 모델 로드 (DeepLabV3+ with ResNet-101 backbone)
model = deeplabv3_resnet101(pretrained=True).to(device).eval()

# 클래스 이름 정의 (예: PASCAL VOC 데이터셋의 클래스 이름)
class_names = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

# 이미지 로드 및 전처리
original_image = Image.open(img_path).convert('RGB')
preprocess = transforms.Compose([
    transforms.Resize((1024, 1024)),  # 고해상도 사용
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
image_tensor = preprocess(original_image).unsqueeze(0).to(device)

# 모델 예측
with torch.no_grad():
    output = model(image_tensor)['out'][0]
preds = torch.argmax(output, dim=0).byte().cpu().numpy()

# 클래스별 색상 맵 생성
unique_classes = np.unique(preds)
color_map = generate_random_colors(len(unique_classes))

# 클래스 인덱스를 색상으로 변환하는 함수
def label_to_color_image(label, color_map, unique_classes):
    label_colors = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    for idx, class_id in enumerate(unique_classes):
        label_colors[label == class_id] = color_map[idx]
    return label_colors

# 클래스 값을 색상 이미지로 변환
preds_color = label_to_color_image(preds, color_map, unique_classes)

# 마스크 생성 (0 또는 255로 구성된 바이너리 마스크)
mask = (preds != 0).astype(np.uint8) * 255

# 원본 이미지를 배열 형태로 변환
original_image_resized = np.array(original_image.resize((1024, 1024)))

# 마스크 영역을 추출
mask_image = Image.fromarray(mask).convert("L")
mask_image_resized = mask_image.resize((1024, 1024))

# 원본 이미지와 블러 처리된 이미지 생성
base_image = Image.fromarray(original_image_resized)
blurred_image = base_image.filter(ImageFilter.GaussianBlur(radius=15))

# 블렌딩: 마스크가 있는 영역은 원본, 나머지는 블러
blended_image = Image.composite(base_image, blurred_image, mask_image_resized)

# # 결과 시각화
# plt.figure(figsize=(10, 10))
# plt.imshow(blended_image)
# plt.title("Original Image with Segmentation Overlay and Blurred Background")
# plt.axis('off')
# plt.show()

# # 클래스 이름과 색상 매핑 출력
# class_color_mapping = []
# for idx, class_id in enumerate(unique_classes):
#     if class_id < len(class_names):  # ensure the class_id is within the range of class_names
#         class_name = class_names[class_id]
#     else:
#         class_name = f"Unknown class {class_id}"
#     color = color_map[idx]
#     class_color_mapping.append((class_name, class_id, color))

blended_image.save("./sementic_seg/results/blended_image_highres.png")
# class_color_mapping
