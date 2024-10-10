import tkinter as tk
from tkinter import ttk, filedialog
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from PIL import Image, ImageTk

class WeatherPredictionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Weather Prediction using Logistic Regression")
        self.master.attributes('-fullscreen', True)

        self.model = None
        self.scaler = None
        self.features = []
        self.label_encoder = None

        self.create_widgets()

    def create_widgets(self):
        self.load_background_image()

        # Place main_frame on top of the canvas (background image)
        self.main_frame = ttk.Frame(self.canvas, style='TFrame')
        self.canvas.create_window(self.master.winfo_screenwidth() // 2, self.master.winfo_screenheight() // 2,
                                  window=self.main_frame, anchor="center")

        title_label = ttk.Label(self.main_frame, text="Weather Prediction", font=("Arial", 24), background='white')
        title_label.pack(pady=20)

        load_button = ttk.Button(self.main_frame, text="Load Dataset", command=self.load_dataset, style='TButton')
        load_button.pack(pady=10)

        self.status_label = ttk.Label(self.main_frame, text="", font=("Arial", 12), background='white')
        self.status_label.pack(pady=10)

        self.input_frame = ttk.Frame(self.main_frame, style='TFrame')
        self.input_frame.pack(pady=20)

        self.predict_button = ttk.Button(self.main_frame, text="Predict", command=self.predict_weather, state="disabled", style='TButton')
        self.predict_button.pack(pady=20)

        self.result_label = ttk.Label(self.main_frame, text="", font=("Arial", 14), background='white', wraplength=800)
        self.result_label.pack()

        exit_button = ttk.Button(self.main_frame, text="Exit", command=self.master.quit, style='TButton')
        exit_button.pack(pady=20)

    def load_background_image(self):
        # Create a canvas to hold the background image
        self.canvas = tk.Canvas(self.master)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Load and resize the background image
        image = Image.open("D:/ML MINI PROJECT/Weather_Prediction/bgpic.png")  # Replace with your image path
        image = image.resize((self.master.winfo_screenwidth(), self.master.winfo_screenheight()), Image.LANCZOS)
        self.bg_image = ImageTk.PhotoImage(image)

        # Display the background image on the canvas
        self.canvas.create_image(0, 0, image=self.bg_image, anchor="nw")

    def load_dataset(self):
        file_path = "D:/ML MINI PROJECT/Weather_Prediction/seattle-weather.csv"
        if file_path:
            try:
                df = pd.read_csv(file_path)
                df = df.drop(columns=['date'])  # Drop 'date' column
                
                # Encoding the target variable 'weather'
                self.label_encoder = LabelEncoder()
                df['weather'] = self.label_encoder.fit_transform(df['weather'])
                
                self.preprocess_data(df)
                self.create_input_fields()
                self.status_label.config(text="Dataset loaded and model trained successfully!")
                self.predict_button.config(state="normal")
            except Exception as e:
                self.status_label.config(text=f"Error: {str(e)}")

    def preprocess_data(self, df):
        # Features and target separation
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        self.features = X.columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.model = LogisticRegression()
        self.model.fit(X_train_scaled, y_train)

        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model accuracy: {accuracy:.2f}")

    def create_input_fields(self):
        for widget in self.input_frame.winfo_children():
            widget.destroy()

        self.entries = {}
        for feature in self.features:
            frame = ttk.Frame(self.input_frame, style='TFrame')
            frame.pack(pady=5)
            
            label = ttk.Label(frame, text=f"{feature}:", font=("Arial", 12), background='white')
            label.pack(side='left', padx=10)
            
            entry = ttk.Entry(frame, font=("Arial", 12), justify='center', width=20)
            entry.pack(side='left')
            
            self.entries[feature] = entry

    def predict_weather(self):
        try:
            input_data = [float(self.entries[feature].get()) for feature in self.features]
            input_scaled = self.scaler.transform([input_data])
            prediction = self.model.predict(input_scaled)
            probability = self.model.predict_proba(input_scaled)[0][1]

            weather_prediction = self.label_encoder.inverse_transform(prediction)[0]
            
            result = f"Prediction: {weather_prediction}\n"
            result += f"Probability: {probability:.2f}\n\n"
            result += "Weather predictions:\n"
            for i, weather_type in enumerate(self.label_encoder.classes_):
                result += f"{i}: {weather_type}\n"

            self.result_label.config(text=result)
        except ValueError:
            self.result_label.config(text="Error: Please enter valid numeric values for all fields.")

if __name__ == "__main__":
    root = tk.Tk()

    # Style settings
    style = ttk.Style(root)
    style.configure('TFrame', background='white')  # Frames will have a transparent background over the image
    style.configure('TButton', font=("Arial", 12))
    style.configure('TLabel', background='white')

    app = WeatherPredictionApp(root)
    root.mainloop()
