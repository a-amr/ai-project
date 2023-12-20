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

from PIL import Image, ImageDraw




class DrawingApp:
    def __init__(self, master, callback):
        self.master = master
        self.master.title("Drawing Window")
        self.canvas = tkinter.Canvas(self.master, width=300, height=300, bg="white")
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.draw)
        self.predict_button = tkinter.Button(self.master, text="Predict", command=callback)
        self.predict_button.pack()
        self.clear_button = tkinter.Button(self.master, text="Clear", command=self.clear_canvas)
        self.clear_button.pack()
        self.image = Image.new("L", (300, 300), color=255)
        self.draw = ImageDraw.Draw(self.image)

    def draw(self, event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.canvas.create_oval(x1, y1, x2, y2, fill="black", width=28)
        self.draw.line([x1, y1, x2, y2], fill="black", width=28)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (300, 300), color=255)
        self.draw = ImageDraw.Draw(self.image)

    def preprocess_image(self):
        img = self.image.resize((28, 28))
        img_array = np.array(img)
        img_array = 255 - img_array
        return np.ravel(img_array).reshape(1, -1)





drawing_app = None



def predict_draw_using_random_forest():
    img = drawing_app.preprocess_image()
    prediction = rf_model.predict(img)
    predicted_class = prediction[0]
    tkinter.messagebox.showinfo("Prediction", f"Predicted letter: {predicted_class}")

def predict_draw_using_dec_tree():
    img = drawing_app.preprocess_image()
    prediction = dt_model.predict(img)
    predicted_class = prediction[0]
    tkinter.messagebox.showinfo("Prediction", f"Predicted letter: {predicted_class}")



def rnd_forest_drawing_window(root):
    drawing_window = tkinter.Toplevel(root)
    global drawing_app
    drawing_app = DrawingApp(drawing_window, predict_draw_using_random_forest)


def dec_tree_drawing_window(root):
    drawing_window = tkinter.Toplevel(root)
    global drawing_app
    drawing_app = DrawingApp(drawing_window, predict_draw_using_dec_tree)















































def train_dt_model(max_depth, dt_model, time_label: tkinter.Label):
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
        dt_model.max_depth = None
    else:
        dt_model.max_depth = int(max_depth)
    dt_starf_train_time = time.time()
    dt_model.fit(X_train_flat, y_train)
    dt_end_train_time = time.time()
    dt_total_train_time = dt_end_train_time - dt_starf_train_time
    time_label.config(text=f"Train time = {round(dt_total_train_time)} sec", font=18)


def pred_image_using_dt(dt_model: DecisionTreeClassifier, predion_label: tkinter.Label):
    file_path = filedialog.askopenfilename(title="Select the image",
                                           filetypes=[("Image files", "*.png")])
    if file_path:
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (28, 28))
        image = np.ravel(image)
        image = image.reshape(1, -1)
        predion_label.config(text=f"Number is: {dt_model.predict(image)}", font=18)


def dt_model_csv(dt_model: DecisionTreeClassifier, accuracy_label: tkinter.Label, compute_acc):

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
    y_pred = dt_model.predict(X_test_flat)
    pd.DataFrame(y_pred).to_csv("predions.csv", index=False, header=False)
    if compute_acc:
        y_test = np.ravel(pd.read_csv(labels_dataset_path, header=None).to_numpy())
        test_accuracy = accuracy_score(y_test, y_pred)
        accuracy_label.config(text=f"Accuracy: {round(test_accuracy *100, 3)}%", font=18)


def train_rf_model(max_depth, n_selectors, rf_model,
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
        rf_model.max_depth = None
    else:
        rf_model.max_depth = int(max_depth)
    if len(n_selectors) == 0:
        rf_model.n_estimators = 10
    else:
        rf_model.n_estimators = int(n_selectors)
    rf_starf_train_time = time.time()
    rf_model.fit(X_train_flat, y_train)
    rf_end_train_time = time.time()
    rf_total_train_time = rf_end_train_time - rf_starf_train_time
    time_label.config(text=f"Train time = {round(rf_total_train_time)} sec", font=18)


def pred_image_using_rf(rf_model: RandomForestClassifier, predion_label: tkinter.Label):
    image = cv2.imread(filedialog.askopenfilename(title="Select the image",
                     filetypes=[("Image files", "*.png")]),
                       cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (28, 28))
    image = np.ravel(image)
    image = image.reshape(1, -1)
    predion_label.config(text=f"Number is: {rf_model.predict(image)}", font=18)

def pred_csv_using_rf(rf_model: RandomForestClassifier, accuracy_label: tkinter.Label,
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
    y_pred = rf_model.predict(X_test_flat)
    pd.DataFrame(y_pred).to_csv("preds.csv", index=False, header=False)
    if compute_acc:
        y_test = np.ravel(pd.read_csv(labels_dataset_path, header=None).to_numpy())
        test_accuracy = accuracy_score(y_test, y_pred)
        accuracy_label.config(text=f"Accuracy: {round(test_accuracy * 100, 3)}%", font=18)



def validate_input(char):
    return char.isdigit()


dt_model = DecisionTreeClassifier()

rf_model = RandomForestClassifier()

window = tkinter.Tk()
window.config(background="#222222")

dtArea = tkinter.Frame(window, bg="#111111")
dtArea.grid(row=0, column=0, padx=(50, 50), pady=(15, 15))

dtlabel = tkinter.Label(dtArea, text="Decision tree", background="#111111", fg="white", font=(18))
dtlabel.grid(row=0, column=0, columnspan=2, pady=(15, 50))

dt_max_depth_label = tkinter.Label(dtArea, text="Max depth:", background="#111111", fg="white",
                  font=(18))
dt_max_depth_label.grid(row=1, column=0, padx=(10, 20))

validate_func = window.register(validate_input)

dt_max_entry = tkinter.Entry(dtArea, width=20, font=(18), bg="#333333", validate="key",
                  validatecommand=(validate_func, "%S"))
dt_max_entry.grid(row=1, column=1, padx=(10, 10))

dt_train_button = tkinter.Button(dtArea,
                  command=lambda: train_dt_model(dt_max_entry.get() , 
                  dt_model, dt_train_time_label),
                  bg="#222222", text="Train decision tree model", fg="#FFFFFF", font=(18))
dt_train_button.grid(row=2, column=0, padx=(50, 50), pady=(50, 15))

dt_train_time_label = tkinter.Label(dtArea, bg="#111111", fg="white")
dt_train_time_label.grid(row=2, column=1, padx=(50, 50), pady=(50, 15))

dt_pred_image_button = tkinter.Button(dtArea,
                  command=lambda: pred_image_using_dt(dt_model,
                  dt_pred_image_result),
                  bg="#222222", text="pred image", fg="#FFFFFF", font=(18))
dt_pred_image_button.grid(row=3, column=0, padx=(50, 50), pady=(50, 15))
dt_pred_image_result = tkinter.Label(dtArea, bg="#111111", fg="white")
dt_pred_image_result.grid(row=3, column=1, padx=(50, 50), pady=(50, 15))

dt_pre_button = tkinter.Button(dtArea, command=lambda: dt_model_csv(dt_model,dt_pred_acc, dt_check_var.get() == 1), bg="#222222", text="pred CSV", fg="#FFFFFF", font=(18))
dt_pre_button.grid(row=4, column=0, padx=(50, 0), pady=(50, 0))

dt_pred_label = tkinter.Label(dtArea, text="Compute accuracy", font=18, bg="#111111", fg="white")
dt_pred_label.grid(row=5, column=0, padx=(50, 0), pady=(0, 15))
dt_check_var = tkinter.IntVar()
dt_pred_check = tkinter.Checkbutton(dtArea, bg="#111111", variable=dt_check_var)
dt_pred_check.grid(row=5, column=1, padx=(0, 50), pady=(0, 15))

dt_pred_acc = tkinter.Label(dtArea, text="", font=18, bg="#111111", fg="white")
dt_pred_acc.grid(row=6, column=0, columnspan=2)





dec_tree_draw_button = tkinter.Button(dtArea, bg="#4444FF", text="Predict From Image", fg="#FFFFFF", font=(18),
                                   command=lambda: dec_tree_drawing_window(window)
                                   )
dec_tree_draw_button.grid(row=7, column=0, columnspan=2, pady=(0, 15))






########## Starf Random forest ##########

ranForArea = tkinter.Frame(window, bg="#111111")
ranForArea.grid(row=0, column=1, padx=(50, 50), pady=(15, 15))

rf_label = tkinter.Label(ranForArea, text="Random forest", bg="#111111", fg="white", font=(18))
rf_label.grid(row=0, column=0, columnspan=2, pady=(15, 50))

rf_max_depth_label = tkinter.Label(ranForArea, text="Max depth:", background="#111111", fg="white",
                  font=(18))
rf_max_depth_label.grid(row=1, column=0, padx=(10, 20))

validate_func = window.register(validate_input)

rf_max_entry = tkinter.Entry(ranForArea, width=20, font=(18), bg="#333333", validate="key",
                  validatecommand=(validate_func, "%S"))
rf_max_entry.grid(row=1, column=1, padx=(10, 10))

rf_selectors_label = tkinter.Label(ranForArea, text="Number of estimators:", background="#111111",
                  fg="white",
                  font=(18))

rf_selectors_label.grid(row=2, column=0, padx=(10, 20))

rf_selectors_entry = tkinter.Entry(ranForArea, width=20,
                  font=(18), bg="#333333", validate="key",
                  validatecommand=(validate_func, "%S"))

rf_selectors_entry.grid(row=2, column=1, padx=(10, 10))

rf_train_button = tkinter.Button(ranForArea, 
                  command=lambda: train_rf_model(rf_max_entry.get(),
                  rf_selectors_entry. get(), rf_model, 
                  rf_train_time_label), bg="#222222", text="Train random forest model", fg="#FFFFFF", font=(18))


rf_train_button.grid(row=3, column=0, padx=(50, 50), pady=(50, 15))

rf_train_time_label = tkinter.Label(ranForArea, bg="#111111", fg="white")
rf_train_time_label.grid(row=3, column=1, padx=(50, 50), pady=(50, 15))

rf_pred_image_button = tkinter.Button(ranForArea,
                  command=lambda: pred_image_using_rf(rf_model,
                  rf_pred_image_result),
                  bg="#222222", text="pred image", fg="#FFFFFF", font=(18))
rf_pred_image_button.grid(row=4, column=0, padx=(50, 50), pady=(50, 15))
rf_pred_image_result = tkinter.Label(ranForArea, bg="#111111", fg="white")
rf_pred_image_result.grid(row=4, column=1, padx=(50, 50), pady=(50, 15))

rf_pred_button = tkinter.Button(ranForArea,
                  command=lambda: pred_csv_using_rf(rf_model,
                  rf_pred_acc,
                  rf_check_var.get() == 1)
                  , bg="#222222", text="pred CSV", fg="#FFFFFF", font=(18))
rf_pred_button.grid(row=5, column=0, padx=(50, 0), pady=(50, 0))

rf_pred_label = tkinter.Label(ranForArea, text="Compute accuracy", font=18, bg="#111111", fg="white")
rf_pred_label.grid(row=6, column=0, padx=(50, 0), pady=(0, 15))
rf_check_var = tkinter.IntVar()
rf_pred_check = tkinter.Checkbutton(ranForArea, bg="#111111", variable=rf_check_var)
rf_pred_check.grid(row=6, column=1, padx=(0, 50), pady=(0, 15))

rf_pred_acc = tkinter.Label(ranForArea, text="", font=18, bg="#111111", fg="white")
rf_pred_acc.grid(row=7, column=0, columnspan=2)






rnd_forest_draw_button = tkinter.Button(ranForArea, bg="#FF4444", text="Predict From Image", fg="#FFFFFF", font=(18),
                                   command=lambda: rnd_forest_drawing_window(window)
                                   )
rnd_forest_draw_button.grid(row=8, column=0, columnspan=2, pady=(0, 15))





window.mainloop()
