import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
import tkinter as tk
from tkinter import filedialog, messagebox

class StockPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Stock Price Prediction")
        
        # Frame to contain input elements
        self.input_frame = tk.Frame(root, bg="pink", pady=20)
        self.input_frame.pack()

        # Button to open file dialog
        self.open_button = tk.Button(self.input_frame, text="Open CSV", command=self.open_csv, bg="pink", fg="white", font=("Arial", 16))
        self.open_button.grid(row=0, column=0, padx=10)

        # Label to display selected file path
        self.file_label = tk.Label(self.input_frame, text="", bg="pink", font=("Arial", 12))
        self.file_label.grid(row=0, column=1, padx=10)

        # Button to start prediction
        self.predict_button = tk.Button(self.input_frame, text="Predict", command=self.predict, bg="pink", fg="white", font=("Arial", 16))
        self.predict_button.grid(row=0, column=2, padx=10)

        # Frame to contain plot
        self.plot_frame = tk.Frame(root)
        self.plot_frame.pack(pady=20)

        # Canvas to display plot
        self.canvas = plt.Figure()
        self.canvas_widget = FigureCanvasTkAgg(self.canvas, master=self.plot_frame)
        self.canvas_widget.get_tk_widget().pack()

    def open_csv(self):
        filename = filedialog.askopenfilename(initialdir="/", title="Select file",
                                              filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
        self.file_label.config(text=filename)

    def predict(self):
        filename = self.file_label.cget("text")
        if not filename:
            messagebox.showerror("Error", "Please select a CSV file first.")
            return

        try:
            df = pd.read_csv(filename)
            data = df.filter(['Close'])

            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data)

            train_size = int(len(scaled_data) * 0.8)
            train_data = scaled_data[0:train_size, :]
            test_data = scaled_data[train_size:, :]

            X_train = []
            y_train = []
            for i in range(60, len(train_data)):
                X_train.append(train_data[i-60:i, 0])
                y_train.append(train_data[i, 0])

            X_train, y_train = np.array(X_train), np.array(y_train)

            model = SVR(kernel='rbf', C=1e3, gamma=0.1)
            model.fit(X_train, y_train)

            X_test = []
            y_test = data[train_size+60:].values
            for i in range(60, len(test_data)):
                X_test.append(test_data[i-60:i, 0])

            X_test = np.array(X_test)

            predictions = model.predict(X_test)
            predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()

            mse = np.mean((y_test - predictions)*0.09)
            print('MSE:', mse)

            self.canvas.clear()
            ax = self.canvas.add_subplot(111)
            ax.plot(y_test, label='Actual')
            ax.plot(predictions, label='Predicted')
            ax.legend()
            self.canvas_widget.draw()

            MAE_numpy = np.mean(np.abs(np.subtract(y_test,predictions)))
            print ("MAE using Numpy: % ", MAE_numpy)

            MAE_sci = mean_absolute_error(y_test,predictions)
            print ("MAE using Sklearn: % ", MAE_sci)

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

root = tk.Tk()
app = StockPredictionApp(root)
root.mainloop()
