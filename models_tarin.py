import os

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.resnet_v2 import ResNet50V2
from keras.applications.vgg16 import VGG16
from keras.applications.xception import Xception
import warnings

warnings.filterwarnings("ignore")

# set proxy to download model
proxy = 'http://proxy.kavosh.org:808'
os.environ['http_proxy'] = proxy
os.environ['HTTP_PROXY'] = proxy
os.environ['https_proxy'] = proxy
os.environ['HTTPS_PROXY'] = proxy


class ColonDataGen(tf.keras.utils.Sequence):

    def __init__(self, df, X_col, y_col,
                 batch_size,
                 model_name,
                 input_size=(640, 640, 3),
                 shuffle=True):

        self.model_name = model_name
        self.df = df.copy()
        self.X_col = X_col
        self.y_col = y_col
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle

        self.num_classes = df[y_col].nunique()
        self.n = len(self.df)

    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def __get_input(self, path, target_size):

        image = tf.keras.preprocessing.image.load_img(path)
        image_arr = tf.keras.preprocessing.image.img_to_array(image)
        image_arr = tf.image.resize(image_arr, (target_size[0], target_size[1])).numpy()
        if self.model_name == "Xception":
            final_image = tf.keras.applications.xception.preprocess_input(image_arr)
        elif self.model_name == "ResNet50V2":
            final_image = tf.keras.applications.resnet_v2.preprocess_input(image_arr)
        elif self.model_name == "MobileNetV2":
            final_image = tf.keras.applications.mobilenet_v2.preprocess_input(image_arr)
        elif self.model_name == "VGG16":
            final_image = tf.keras.applications.vgg16.preprocess_input(image_arr)
        else:
            final_image = image_arr / 255.
        return final_image

    def __get_output(self, label, num_classes):
        return tf.keras.utils.to_categorical(label, num_classes=num_classes)

    def __get_data(self, batches):
        path_batch = batches[self.X_col]
        label_batch = batches[self.y_col]

        X_batch = np.asarray([self.__get_input(x, self.input_size) for x in path_batch])
        # y_batch = np.asarray([self.__get_output(y, self.num_classes) for y in label_batch])
        y_batch = np.asarray([y for y in label_batch])

        return X_batch, y_batch

    def __getitem__(self, index):

        batches = self.df[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(batches)
        return X, y

    def __len__(self):
        return self.n // self.batch_size


def model_network(model_name, train_data):
    #  define models
    if model_name == "Xception":
        model_net = Xception(
            input_shape=(640, 640, 3),
            include_top=False,
            weights="imagenet"
        )
    elif model_name == "ResNet50V2":
        model_net = ResNet50V2(
            input_shape=(640, 640, 3),
            include_top=False,
            weights="imagenet"
        )
    elif model_name == "MobileNetV2":
        model_net = MobileNetV2(
            input_shape=(640, 640, 3),
            include_top=False,
            weights="imagenet"
        )
    elif model_name == "VGG16":
        model_net = VGG16(
            include_top=False,
            weights="imagenet",
            input_shape=(640, 640, 3),
        )
    # freeze models layer to not train
    for layer in model_net.layers:
        layer.trainable = False

    # define Dense layer to finetune
    trans_model = keras.layers.GlobalAveragePooling2D()(model_net.output)
    trans_model = keras.layers.Dropout(.2)(trans_model)
    trans_model = keras.layers.Dense(1024, activation="relu")(trans_model)
    trans_model = keras.layers.Dropout(.2)(trans_model)
    trans_model = keras.layers.Dense(1, activation="sigmoid")(trans_model)

    model = keras.Model(inputs=model_net.input, outputs=trans_model)

    # print(model.summary())
    #  set Config
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=10)
    reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=5, min_lr=0.0001)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=f'./output/{model_name}/', histogram_freq=0,
                                                          write_graph=True, write_images=False)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(f"./output/{model_name}/Best_model.hdf5", monitor="loss",
                                                             save_best_only=True)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy", "mse"])

    # Fit models
    model.fit(train_data, epochs=50,
              callbacks=[checkpoint_callback, tensorboard_callback, early_stopping_callback, reduce_lr_callback])



# "EfficientNetB0", not ok
# models name list "Xception", "ResNet50V2" "ResNet50V2",,
model_name_list = [  "Xception", "ResNet50V2"]
for model_name in model_name_list:
    print(35 * "*")
    print(model_name)
    BATCH_SIZE = 32

    # read data info
    train_df = pd.read_csv("data/data_aug/train/info.csv", quoting=1)
    test_df = pd.read_csv("data/data_aug/test/info.csv", quoting=1)

    # Data Generator
    traingen = ColonDataGen(train_df, model_name=model_name, X_col='image', y_col='label', batch_size=BATCH_SIZE,
                            input_size=(640, 640, 3))

    model_network(model_name=model_name, train_data=traingen)
