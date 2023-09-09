import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import metrics

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


# load test data csv
test_df = pd.read_csv("data/data_aug/test/info.csv", quoting=1)

# variables
# models name list "Xception",
model_name_list = ["Xception","ResNet50V2","VGG16", "MobileNetV2" ]
BATCH_SIZE = 207

for model_name in model_name_list:
    # Load the model
    loaded_model = tf.keras.models.load_model(f"output/{model_name}/Best_model.hdf5")

    # data Generator for test data
    valdata = ColonDataGen(test_df, model_name=model_name, X_col='image', y_col='label', batch_size=BATCH_SIZE,
                          input_size=(640, 640, 3))
    x, y = valdata.__getitem__(0)


    # predict data
    y_pred = loaded_model.predict(x)

    y_pred_prob = np.array([inner for outer in y_pred for inner in outer])
    y_pred_label = [1 if inner > 0.5 else 0 for outer in y_pred for inner in outer]

    addres_name = f"report/{model_name}"
    if not os.path.exists(addres_name):
        os.makedirs(addres_name)
    # plot AUC metric
    fpr, tpr, _ = metrics.roc_curve(np.array(list(test_df["label"])), y_pred_prob)
    auc = metrics.roc_auc_score(np.array(list(test_df["label"])), y_pred_prob)
    print("AUC == ", auc)
    plt.plot(fpr, tpr, label=" auc=" + str(auc))
    plt.legend(loc=4)
    plt.savefig(f"{addres_name}/AUC_curve.png")
    plt.clf()

    with open(f'{addres_name}/{model_name}_eval.txt', 'w') as f:
        f.write(f"********* {model_name} Outputs *********** \n")
        f.write(f"bath number == 32 \n")
        f.write(f"epoch == 50 \n")
        f.write(f"AUC == {str(auc)} \n")
    # calculate Accuracy
        acc = metrics.accuracy_score(np.array(list(test_df["label"])), y_pred_label)
        print("Accuracy == ", acc)
        f.write(f"Accuracy == {str(acc)} \n")
        # calculate F-1 score
        f1 = metrics.f1_score(np.array(list(test_df["label"])), y_pred_label)
        print("f1_score == ", f1)
        f.write(f"f1_score == {str(f1)} \n")
        # classification report
        report = metrics.classification_report(np.array(list(test_df["label"])), y_pred_label)
        print("\n", report)
        f.write("     **** Overall Report **** \n")
        f.write(report)

        f.close()


    # plot Confusion Matrix
    CM = metrics.confusion_matrix(np.array(list(test_df["label"])), y_pred_label)
    disp = metrics.ConfusionMatrixDisplay(CM, display_labels=["Normal", "cancer"])
    disp.plot()
    plt.savefig(f"{addres_name}/Confusion_Matrix.png")
    plt.clf()

    # build model graph as image
    tf.keras.utils.plot_model(
        loaded_model,
        to_file=f"{addres_name}/{model_name}_graph.png",
        show_shapes=False,
        show_dtype=False,
        show_layer_names=True,
        rankdir="TB",
        expand_nested=False,
        dpi=96,
        layer_range=None,
        show_layer_activations=False

    )
