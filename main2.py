#!/usr/bin/python3
import time
import tkinter
from tkinter import filedialog
import cv2
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np


def train_dec_model(max_depth, decision_tree_model, time_label: tkinter.Label):
    features_dataset_path = filedialog.askopenfilename(title="Select The train FEATURES(X) Dataset", filetypes=[("CSV files", "*.csv")])
    labels_dataset_path = filedialog.askopenfilename(title="Select The train LABELS(y) Dataset", filetypes=[("CSV files", "*.csv")])
    X_train_raw = pd.read_csv(features_dataset_path, header=None)
    y_train_raw = pd.read_csv(labels_dataset_path, header=None)
    X_train = X_train_raw.to_numpy()
    X_train = X_train.reshape(X_train_raw.shape[0], 28, 28)
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    y_train = np.ravel(y_train_raw.to_numpy())

    for i in range(X_train.shape[0]):
        X_train[i] = np.rot90(X_train[i], k=3)
        X_train[i] = np.flip(X_train[i], axis=1)
    if len(max_depth) == 0:
        decision_tree_model.max_depth = None
    else:
        decision_tree_model.max_depth = int(max_depth)
    decision_tree_start_train_time = time.time()
    decision_tree_model.fit(X_train_flat, y_train)
    decision_tree_end_train_time = time.time()
    decision_tree_total_train_time = decision_tree_end_train_time - decision_tree_start_train_time
    time_label.config(text=f"Train time = {round(decision_tree_total_train_time)} sec", font=18)


def predict_image_using_decision_tree(decision_tree_model: DecisionTreeClassifier, prediction_label: tkinter.Label):
    file_path = filedialog.askopenfilename(title="Select the image",
                                           filetypes=[("Image files", "*.png")])
    if file_path:
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (28, 28))
        image = np.ravel(image)
        image = image.reshape(1, -1)
        prediction_label.config(text=f"Number is: {decision_tree_model.predict(image)}", font=18)


def dt_model_csv(decision_tree_model: DecisionTreeClassifier, accuracy_label: tkinter.Label, compute_acc):

    features_dataset_path = filedialog.askopenfilename(title="Select The test FEATURES(X) Dataset",
                                                       filetypes=[("CSV files", "*.csv")])
    labels_dataset_path = ''
    if compute_acc: labels_dataset_path = filedialog.askopenfilename(title="Select The train LABELS(y) Dataset",
                                                                     filetypes=[("CSV files", "*.csv")])

    X_test_raw = pd.read_csv(features_dataset_path, header=None).to_numpy()
    X_test = X_test_raw.reshape(X_test_raw.shape[0], 28, 28)
    for i in range(X_test.shape[0]):
        X_test[i] = np.rot90(X_test[i], k=3)
        X_test[i] = np.flip(X_test[i], axis=1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    y_pred = decision_tree_model.predict(X_test_flat)
    pd.DataFrame(y_pred).to_csv("predictions.csv", index=False, header=False)
    print("Predicted")
    if compute_acc:
        y_test = np.ravel(pd.read_csv(labels_dataset_path, header=None).to_numpy())
        test_accuracy = accuracy_score(y_test, y_pred)
        accuracy_label.config(text=f"Accuracy: {round(test_accuracy *100, 3)}%", font=18)


def train_rnd_forest_model(max_depth, n_selectors, random_forest_model,
                           time_label: tkinter.Label):
    features_dataset_path = filedialog.askopenfilename(title="Select The train FEATURES(X) Dataset",
                                                       filetypes=[("CSV files", "*.csv")])
    labels_dataset_path = filedialog.askopenfilename(title="Select The train LABELS(y) Dataset",
                                                     filetypes=[("CSV files", "*.csv")])
    X_train_raw = pd.read_csv(features_dataset_path, header=None)
    y_train_raw = pd.read_csv(labels_dataset_path, header=None)
    X_train = X_train_raw.to_numpy()
    X_train = X_train.reshape(X_train_raw.shape[0], 28, 28)
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    y_train = np.ravel(y_train_raw.to_numpy())

    for i in range(X_train.shape[0]):
        X_train[i] = np.rot90(X_train[i], k=3)
        X_train[i] = np.flip(X_train[i], axis=1)
    if len(max_depth) == 0:
        random_forest_model.max_depth = None
    else:
        random_forest_model.max_depth = int(max_depth)
    if len(n_selectors) == 0:
        random_forest_model.n_estimators = 10
    else:
        random_forest_model.n_estimators = int(n_selectors)
    random_forest_start_train_time = time.time()
    random_forest_model.fit(X_train_flat, y_train)
    random_forest_end_train_time = time.time()
    random_forest_total_train_time = random_forest_end_train_time - random_forest_start_train_time
    time_label.config(text=f"Train time = {round(random_forest_total_train_time)} sec", font=18)


def predict_image_using_random_forest(random_forest_model: RandomForestClassifier, prediction_label: tkinter.Label):
    image = cv2.imread(filedialog.askopenfilename(title="Select the image",
                                                  filetypes=[("Image files", "*.png")]),
                       cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (28, 28))
    image = np.ravel(image)
    image = image.reshape(1, -1)
    prediction_label.config(text=f"Number is: {random_forest_model.predict(image)}", font=18)

def predict_csv_using_random_forest(random_forest_model: RandomForestClassifier, accuracy_label: tkinter.Label,
                                    compute_acc):
    features_dataset_path = filedialog.askopenfilename(title="Select The test FEATURES(X) Dataset",
                                                       filetypes=[("CSV files", "*.csv")])
    labels_dataset_path = ''
    if compute_acc: labels_dataset_path = filedialog.askopenfilename(title="Select The train LABELS(y) Dataset",
                                                                     filetypes=[("CSV files", "*.csv")])

    X_test_raw = pd.read_csv(features_dataset_path, header=None).to_numpy()
    X_test = X_test_raw.reshape(X_test_raw.shape[0], 28, 28)
    for i in range(X_test.shape[0]):
        X_test[i] = np.rot90(X_test[i], k=3)
        X_test[i] = np.flip(X_test[i], axis=1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    y_pred = random_forest_model.predict(X_test_flat)
    pd.DataFrame(y_pred).to_csv("preds.csv", index=False, header=False)
    if compute_acc:
        y_test = np.ravel(pd.read_csv(labels_dataset_path, header=None).to_numpy())
        test_accuracy = accuracy_score(y_test, y_pred)
        accuracy_label.config(text=f"Accuracy: {round(test_accuracy * 100, 3)}%", font=18)



def validate_input(char):
    return char.isdigit()


dec_model = DecisionTreeClassifier()

rnd_forest_model = RandomForestClassifier()

window = tkinter.Tk()
window.config(background="#222222")

decArea = tkinter.Frame(window, bg="#111111")
decArea.grid(row=0, column=0, padx=(50, 50), pady=(15, 15))

deslabel = tkinter.Label(decArea, text="Decision tree", background="#111111", fg="white", font=(18))
deslabel.grid(row=0, column=0, columnspan=2, pady=(15, 50))

dec_max_depth_label = tkinter.Label(decArea, text="Max depth:", background="#111111", fg="white",
                                         font=(18))
dec_max_depth_label.grid(row=1, column=0, padx=(10, 20))

validate_func = window.register(validate_input)

dec_max_entry = tkinter.Entry(decArea, width=20, font=(18), bg="#333333", validate="key",
                                   validatecommand=(validate_func, "%S"))
dec_max_entry.grid(row=1, column=1, padx=(10, 10))

dec_train_button = tkinter.Button(decArea,
                                       command=lambda: train_dec_model(dec_max_entry.get(),
                                                                            dec_model, dec_train_time_label),
                                       bg="#222222", text="Train decision tree model", fg="#FFFFFF", font=(18))
dec_train_button.grid(row=2, column=0, padx=(50, 50), pady=(50, 15))

dec_train_time_label = tkinter.Label(decArea, bg="#111111", fg="white")
dec_train_time_label.grid(row=2, column=1, padx=(50, 50), pady=(50, 15))

dec_predict_image_button = tkinter.Button(decArea,
                                               command=lambda: predict_image_using_decision_tree(dec_model,
                                                                                                 dec_predict_image_result),
                                               bg="#222222", text="Predict image", fg="#FFFFFF", font=(18))
dec_predict_image_button.grid(row=3, column=0, padx=(50, 50), pady=(50, 15))
dec_predict_image_result = tkinter.Label(decArea, bg="#111111", fg="white")
dec_predict_image_result.grid(row=3, column=1, padx=(50, 50), pady=(50, 15))

dec_pre_button = tkinter.Button(decArea, command=lambda: dt_model_csv(dec_model,dec_predict_acc, dec_check_var.get() == 1) 
                                         , bg="#222222", text="Predict CSV", fg="#FFFFFF", font=(18))
dec_pre_button.grid(row=4, column=0, padx=(50, 0), pady=(50, 0))

dec_predict_label = tkinter.Label(decArea, text="Compute accuracy", font=18, bg="#111111", fg="white")
dec_predict_label.grid(row=5, column=0, padx=(50, 0), pady=(0, 15))
dec_check_var = tkinter.IntVar()
dec_predict_check = tkinter.Checkbutton(decArea, bg="#111111", variable=dec_check_var)
dec_predict_check.grid(row=5, column=1, padx=(0, 50), pady=(0, 15))

dec_predict_acc = tkinter.Label(decArea, text="", font=18, bg="#111111", fg="white")
dec_predict_acc.grid(row=6, column=0, columnspan=2)

########## Start Random forest ##########

ranForArea = tkinter.Frame(window, bg="#111111")
ranForArea.grid(row=0, column=1, padx=(50, 50), pady=(15, 15))

random_forest_label = tkinter.Label(ranForArea, text="Random forest", bg="#111111", fg="white", font=(18))
random_forest_label.grid(row=0, column=0, columnspan=2, pady=(15, 50))

rnd_forest_max_depth_label = tkinter.Label(ranForArea, text="Max depth:", background="#111111", fg="white",
                                           font=(18))
rnd_forest_max_depth_label.grid(row=1, column=0, padx=(10, 20))

validate_func = window.register(validate_input)

rnd_forest_max_entry = tkinter.Entry(ranForArea, width=20, font=(18), bg="#333333", validate="key",
                                     validatecommand=(validate_func, "%S"))
rnd_forest_max_entry.grid(row=1, column=1, padx=(10, 10))

rnd_forest_selectors_label = tkinter.Label(ranForArea, text="Number of estimators:", background="#111111",
                                           fg="white",
                                           font=(18))

rnd_forest_selectors_label.grid(row=2, column=0, padx=(10, 20))

rnd_forest_selectors_entry = tkinter.Entry(ranForArea, width=20,
                                           font=(18), bg="#333333", validate="key",
                                           validatecommand=(validate_func, "%S"))

rnd_forest_selectors_entry.grid(row=2, column=1, padx=(10, 10))

rnd_forest_train_button = tkinter.Button(ranForArea, 
                                            command=lambda: train_rnd_forest_model(rnd_forest_max_entry.get(),
                                            rnd_forest_selectors_entry. get(), rnd_forest_model, 
                                            rnd_forest_train_time_label), bg="#222222", text="Train random forest model", fg="#FFFFFF", font=(18))


rnd_forest_train_button.grid(row=3, column=0, padx=(50, 50), pady=(50, 15))

rnd_forest_train_time_label = tkinter.Label(ranForArea, bg="#111111", fg="white")
rnd_forest_train_time_label.grid(row=3, column=1, padx=(50, 50), pady=(50, 15))

rnd_forest_predict_image_button = tkinter.Button(ranForArea,
                                                 command=lambda: predict_image_using_random_forest(rnd_forest_model,
                                                                                                   rnd_forest_predict_image_result),
                                                 bg="#222222", text="Predict image", fg="#FFFFFF", font=(18))
rnd_forest_predict_image_button.grid(row=4, column=0, padx=(50, 50), pady=(50, 15))
rnd_forest_predict_image_result = tkinter.Label(ranForArea, bg="#111111", fg="white")
rnd_forest_predict_image_result.grid(row=4, column=1, padx=(50, 50), pady=(50, 15))

rnd_forest_predict_button = tkinter.Button(ranForArea,
                                           command=lambda: predict_csv_using_random_forest(rnd_forest_model,
                                                                                           rnd_forest_predict_acc,
                                                                                           rnd_forest_check_var.get() == 1)
                                           , bg="#222222", text="Predict CSV", fg="#FFFFFF", font=(18))
rnd_forest_predict_button.grid(row=5, column=0, padx=(50, 0), pady=(50, 0))

rnd_forest_predict_label = tkinter.Label(ranForArea, text="Compute accuracy", font=18, bg="#111111", fg="white")
rnd_forest_predict_label.grid(row=6, column=0, padx=(50, 0), pady=(0, 15))
rnd_forest_check_var = tkinter.IntVar()
rnd_forest_predict_check = tkinter.Checkbutton(ranForArea, bg="#111111", variable=rnd_forest_check_var)
rnd_forest_predict_check.grid(row=6, column=1, padx=(0, 50), pady=(0, 15))

rnd_forest_predict_acc = tkinter.Label(ranForArea, text="", font=18, bg="#111111", fg="white")
rnd_forest_predict_acc.grid(row=7, column=0, columnspan=2)

window.mainloop()
