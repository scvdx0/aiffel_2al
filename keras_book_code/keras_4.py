#%% 케라스로 배우는 딥러닝 책 4장 분류와 회귀
# 영화 리뷰 분류 IMDB 데이터셋
from keras.datasets import imdb
# num_words=10000은 훈련 데이터에서 가장 자주 나타나는 단어 10,000개만 사용하겠다는 의미, 드물게 나타나는 단어는 무시해야 분석이 됨
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
len(train_data), len(test_data)
len(train_labels), len(test_labels)
len(train_data[0]), len(test_data[0])
train_labels[0], test_labels[0]

# 리뷰를 다시 텍스트로 디코딩
# word_index는 단어와 정수 인덱스를 매핑한 딕셔너리
# i - 3은 인덱스 0, 1, 2가 각각 '패딩', '문서 시작', '사전에 없음'을 위한 인덱스이므로 이를 제외하고 디코딩 -> reverse_word_index의 인덱싱은 0부터 되었는데, 데이터의 인덱싱은 3부터 되었나봄
word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])

# 정수 시퀀스를 이진 행렬로 인코딩 (멀티 핫 인코딩)
import numpy as np

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
y_train = np.asarray(train_labels).astype("float32")
y_test = np.asarray(test_labels).astype("float32")
type(x_train), type(y_train)

# 모델 정의하기
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(16, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# 모델 컴파일하기
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy', 'mean_squared_error'])

# 훈련 검증
# 검증 세트 준비하기
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# 모델 훈련하기
history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))

history_dict = history.history
history_dict.keys()

# 훈련과 검증 손실 그리기
# 그래프를 보면 중간에 과적합되는 것을 볼 수 있음, early stopping을 사용하면 과적합을 방지할 수 있으나, 여기서는 이정도로 마무리
import matplotlib.pyplot as plt
history_dict = history.history
loss_values = history_dict["loss"]
val_loss_values = history_dict["val_loss"]
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, "bo", label="Training loss")
plt.plot(epochs, val_loss_values, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.plot()
plt.savefig('plot.png')

import matplotlib.pyplot as plt
history_dict = history.history
loss_values = history_dict["loss"]
val_loss_values = history_dict["val_loss"]
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, "bo", label="Training loss")
plt.plot(epochs, val_loss_values, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.plot()
plt.savefig('plot2.png')

# 평가
# loss, accuracy, mean_squared_error
results = model.evaluate(x_test, y_test)

# 새로운 데이터에 대해 예측하기
model.predict(x_test)

# 랜덤 분류기로 비교하기
import copy
test_labels_copy = copy.copy(test_labels)
np.random.shuffle(test_labels_copy)
hits_array = np.array(test_labels) == np.array(test_labels_copy)
float(np.sum(hits_array)) / len(test_labels)
hits_array.mean()


#%% 뉴스 기사 분류 : 로이터 데이터셋 : 46개의 토픽 : 다중 분류
from tensorflow.keras.datasets import reuters
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

len(train_data), len(test_data)
train_data[10]
test_data[10]

# 텍스트로 디코딩
word_index = reuters.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_newswire = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])

train_labels[10]

# 데이터 준비하기
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# 레이블을 벡터로 변환 : 원-핫 인코딩 2가지 방법
def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results
y_train = to_one_hot(train_labels)
y_test = to_one_hot(test_labels)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)

# 모델 정의하기
model = keras.Sequential([
    layers.Dense(64, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(46, activation="softmax")
])

# 모델 컴파일하기
model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

# 훈련 검증
x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = y_train[:1000]
partial_y_train = y_train[1000:]

# 모델 훈련하기
history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))

# 평가
results = model.evaluate(x_test, y_test)

# 예측
predictions = model.predict(x_test)

# 과적합에 대한 대응 : early stopping은 넘기고 마무리



#%% 주택 가격 예측 : 보스턴 주택 가격 데이터셋 : 회귀
from tensorflow.keras.datasets import boston_housing
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

train_data.shape, test_data.shape
train_data[0]
train_targets[0]

# 데이터 정규화
# 데이터간 값의 범위가 달라서 정규화했음 : 특성별로 정규화하는 것이 일반적
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std

# 모델 정의하기
def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(1)  # 회귀는 활성화 함수가 없음, 활성화 함수를 쓰면 값의 범위가 제한됨 -> 제한 없이 원하는 값이 나올 수 있도록 함
    ])
    model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
    return model

# k-fold cross validation
k = 4
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []
for i in range(k):
    print(f"#{i}번째 폴드 처리중")
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis=0)
    model = build_model()
    model.fit(partial_train_data, partial_train_targets,
              epochs=num_epochs, batch_size=16, verbose=0)
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)

all_scores
np.mean(all_scores)

# 위 코드처럼 하면, for문이 끝난 후에 model의 mse, mae 평균 값만 알 수 있음
# 각 폴드에서 검증 점수를 로그에 저장하기
num_epochs = 500
all_mae_histories = []
for i in range(k):
    print(f"#{i}번째 폴드 처리중")
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis=0)
    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets,
                        validation_data=(val_data, val_targets), # val 데이털르 알려줘서 검증 점수를 epoch마다 저장하게 함
                        epochs=num_epochs, batch_size=16, verbose=0)
    print(len(history.history["val_mae"]))
    mae_history = history.history["val_mae"]
    all_mae_histories.append(mae_history)

# 모든 폴드에 대해 에포크의 MAE 점수 평균을 계산하기
average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

# 검증 점수 그래프
plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel("Epochs")
plt.ylabel("Validation MAE")
plt.plot()
plt.savefig('plot3.png')


# 처음 10개의 데이터 포인트를 제외한 검증 점수 그리기
truncated_mae_history = average_mae_history[10:]
plt.plot(range(1, len(truncated_mae_history) + 1), truncated_mae_history)
plt.xlabel("Epochs")
plt.ylabel("Validation MAE")
plt.plot()
plt.savefig('plot4.png')

# 최종 모델 훈련하기 결과가 잘 나온 걸로
model = build_model()
model.fit(train_data, train_targets,
          epochs=130, batch_size=16, verbose=0)
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)

test_mae_score

# 평가
# loss, mae
results = model.evaluate(test_data, test_targets)
results

# 예측
predictions = model.predict(test_data)
predictions[0]