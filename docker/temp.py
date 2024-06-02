import tensorflow as tf
import numpy as np

# TensorFlow에서 사용 가능한 GPU 목록을 가져옵니다.
# window에서 tensorflow-gpu를 사용하려면 CUDA와 cuDNN을 설치해야 하지만, 예전 버전밖에 지원하지 않음
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    try:
        # 필요한 경우, 메모리 성장을 허용하도록 설정할 수 있습니다.
        # TensorFlow는 기본적으로 가능한 모든 GPU 메모리를 할당합니다.
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Available GPUs: {gpus}")
    except RuntimeError as e:
        # 메모리 성장을 설정하는 과정에서 발생할 수 있는 오류를 캐치합니다.
        print(e)
else:
    print("CUDA is not available in TensorFlow.")

import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))



import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))

if tf.config.list_physical_devices('GPU'):
    print("TensorFlow is using GPU.")
else:
    print("TensorFlow is not using GPU.")



import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 메모리 할당 설정
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
