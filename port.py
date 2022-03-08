#嫌悪画像のデータ拡張
import numpy as np
import cv2
import glob

#testdataの拡張
path = "face/test/disgust/*.jpg"
file = glob.glob(path)
#ガンマ補正による輝度変換
def create_gamma_img(gamma, img):
    gamma_cvt = np.zeros((256,1), dtype=np.uint8)
    for i in range(256):
        gamma_cvt[i][0] = 255*(float(i)/255)**(1.0/gamma)
    return cv2.LUT(img, gamma_cvt)

lst = [0.5,1,3]
count = 0
#反鏡処理+ガンマ補正による輝度変換
for i in range(len(file)):
    img = cv2.imread(file[i])
    for j in range(len(lst)):
        path = "face/test/disgust2/"+str(count)+".jpg"
        img_gamma = create_gamma_img(lst[j], img)
        cv2.imwrite(path,img_gamma)
        count += 1
        path = "face/test/disgust2/"+str(count)+".jpg"
        img_mirror = cv2.flip(img_gamma,1)
        cv2.imwrite(path,img_mirror)
        count += 1
#元ファイル削除
for file in glob.glob("face/test/disgust/*.jpg"):
    os.remove(file)

#traindataの拡張
path = "face/train/disgust/*.jpg"
file = glob.glob(path)
count = 0

for i in range(len(file)):
    img = cv2.imread(file[i])
    for j in range(len(lst)):
        path = "face/train/disgust2/"+str(count)+".jpg"
        img_gamma = create_gamma_img(lst[j], img)
        cv2.imwrite(path,img_gamma)
        count += 1
        path = "face/train/disgust2/"+str(count)+".jpg"
        img_mirror = cv2.flip(img_gamma,1)
        cv2.imwrite(path,img_mirror)
        count += 1

for file in glob.glob("face/train/disgust/*.jpg"):
    os.remove(file)

#validationdataの拡張
count = 0

for i in range(len(file)):
    img = cv2.imread(file[i])
    for j in range(len(lst)):
        path = "face/validation/disgust2/"+str(count)+".jpg"
        img_gamma = create_gamma_img(lst[j], img)
        cv2.imwrite(path,img_gamma)
        count += 1
        path = "face/validation/disgust2/"+str(count)+".jpg"
        img_mirror = cv2.flip(img_gamma,1)
        cv2.imwrite(path,img_mirror)
        count += 1

for file in glob.glob("face/validation/disgust/*.jpg"):
    os.remove(file)

#拡張後のデータを移動元の枠へ
#cp　-r drive/MyDrive/face/test/disgust face/test/
#cp　-r drive/MyDrive/face/train/disgust face/train/
#cp　-r drive/MyDrive/face/validation/disgust face/validation/


#-------------------------------------------------------------------------------


import numpy as np
import os
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Activation, Dropout, Flatten, Dense
from tensorflow.keras import optimizers
import pickle
import matplotlib.pyplot as plt

#acc/lossの推移グラフ
def plot_acc(history,
                fig_size_width,
                fig_size_height,
                lim_font_size):

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    epochs = range(len(acc))

    plt.figure(figsize=(fig_size_width, fig_size_height))
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = lim_font_size

    plt.plot(epochs, acc, color = "blue", linestyle = "solid", label = 'train acc')
    plt.plot(epochs, val_acc, color = "green", linestyle = "solid", label= 'valid acc')
    plt.title('Training and Validation acc')
    plt.xlabel("accuracy")
    plt.ylabel("epochs")
    plt.grid()
    plt.legend()
    plt.show()
    plt.close()

def plot_loss(history,
                fig_size_width,
                fig_size_height,
                lim_font_size):

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    plt.figure(figsize=(fig_size_width, fig_size_height))
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = lim_font_size

    plt.plot(epochs, loss, color = "red", linestyle = "solid" ,label = 'train loss')
    plt.plot(epochs, val_loss, color = "orange", linestyle = "solid" , label= 'valid loss')
    plt.title('Training and Validation loss')
    plt.xlabel("loss")
    plt.ylabel("epochs")
    plt.legend()
    plt.grid()
    plt.show()
    plt.close()


#-------------------------------------------------------------------------------


#modelの構成（VGG16のモデル）
def build_model(num_classes,
                img_width=48,
                img_height=48):

    input_tensor = Input(shape=(img_width, img_height, 3))

    vgg16 = VGG16(include_top=False, weights="imagenet", input_tensor=input_tensor)

    model = Sequential()

    model.add(Flatten(input_shape=vgg16.output_shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(num_classes, activation='softmax'))

    new_model = Model(vgg16.input, model(vgg16.output))

    return new_model


#-------------------------------------------------------------------------------


#画像データを取得（リスケール等）
def img_generator(classes,
                train_path,
                valid_path,
                test_path,
                batch_size=20,
                img_width=32,
                img_height=32):


    train_gen = ImageDataGenerator(rescale=1.0 / 255, zoom_range=0.2, horizontal_flip=True)

    valid_gen = ImageDataGenerator(rescale=1.0 / 255)

    test_gen = ImageDataGenerator(rescale=1.0 / 255)


    train_datas = train_gen.flow_from_directory(train_path,
                target_size=(img_width, img_height),
                color_mode='rgb',
                classes=classes,
                class_mode='categorical',
                batch_size=batch_size,
                shuffle=True)
    valid_datas = valid_gen.flow_from_directory(
                valid_path,
                target_size=(img_width, img_height),
                color_mode='rgb',
                classes=classes,
                class_mode='categorical',
                batch_size=batch_size,
                shuffle=True)

    test_datas = test_gen.flow_from_directory(
                test_path,
                target_size=(img_width, img_height),
                color_mode='rgb',
                classes=classes,
                class_mode='categorical',
                batch_size=batch_size,
                shuffle=False)

    return train_datas,valid_datas, test_datas


#-------------------------------------------------------------------------------


#パラメータの調節
num_epoch = 50
batch_size = 128
num_classes = 7
classes = ['angry','disgust','fear','happy','neutral','sad','surprise']

img_width = 48
img_height = 48

FIG_SIZE_WIDTH = 12
FIG_SIZE_HEIGHT = 10
FIG_FONT_SIZE = 25

SAVE_DATA_DIR_PATH = 'face/'
os.makedirs(SAVE_DATA_DIR_PATH, exist_ok=True)

#modelの構築
model = build_model(num_classes = num_classes,
                        img_width = img_width,
                        img_height = img_height)

#VGG16モデルの14層を凍結
for layer in model.layers[:15]:
  layer.trainable = False

#modelのコンパイル
model.compile(loss = 'categorical_crossentropy',
          optimizer = optimizers.SGD(lr=1e-3, momentum=0.9),
          metrics=['accuracy'])

#各画像データの挿入
train_datas,valid_datas, test_datas = img_generator(classes = classes,
                            train_path = SAVE_DATA_DIR_PATH + "train/",
                            valid_path = SAVE_DATA_DIR_PATH + "validation/",
                            test_path = SAVE_DATA_DIR_PATH + "test/",
                            batch_size = batch_size,
                            img_width = img_width,
                            img_height = img_height)

#モデルの学習
history = model.fit(train_datas,epochs = num_epoch,validation_data = valid_datas)


#-------------------------------------------------------------------------------


#モデルのacc/loss画像の表示
plot_acc(history,
                fig_size_width = FIG_SIZE_WIDTH,
                fig_size_height = FIG_SIZE_HEIGHT,
                lim_font_size = FIG_FONT_SIZE)
plot_loss(history,
                fig_size_width = FIG_SIZE_WIDTH,
                fig_size_height = FIG_SIZE_HEIGHT,
                lim_font_size = FIG_FONT_SIZE)


#-------------------------------------------------------------------------------


#testdataでの結果を挿入
predicts = model.predict(test_datas)

#modelの精度の確認
import copy

test_lst = [958,111,1024,1774,1233,1247,831]
test_copy = copy.copy(test_lst)

for i in range(1,num_classes):
    test_lst[i] = test_lst[i-1] + test_lst[i]

result_lst = [[0]*7 for _ in range(7)]
acc_lst = [[0]*7 for _ in range(7)]
miss_lst = []

start = 0

for i in range(num_classes):
    for j in range(start,test_lst[i]):
        result_lst[i][np.argmax(predicts[j])] = result_lst[i][np.argmax(predicts[j])] + 1
        if np.argmax(predicts[j]) != i:
            miss_lst.append(j)
    start = test_lst[i]
    acc_lst[i] = result_lst[i][i]

for i in range(num_classes):
  acc_lst[i] = acc_lst[i]*100//test_copy[i]

#分布の表示
for i in range(num_classes):
  print(result_lst[i])



#各表情の正答率のグラフ表示
def bar_graph(classes,
          lst,
          fig_size_width = FIG_SIZE_WIDTH,
          fig_size_height = FIG_SIZE_HEIGHT,
          lim_font_size = FIG_FONT_SIZE):
    left = [i+1 for i in range(len(lst))]
    plt.figure(figsize=(fig_size_width, fig_size_height))
    plt.title('Test acc')
    plt.ylabel("%")
    plt.bar(left,lst,width=0.5,color='#0096c8',edgecolor='b',linewidth=2,tick_label=classes)
    plt.show()

bar_graph(classes,acc_lst,FIG_SIZE_WIDTH,FIG_SIZE_HEIGHT,FIG_FONT_SIZE)
