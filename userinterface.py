import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from mqtt_client import CustomMQTTClient
from database_storage import SensorDatabaseHandler as Database
from predict import WeightPredictionModel


class SimpleApp(tk.Tk):
    def __init__(self, db_file):
        super().__init__()
        self.title("Automatisierung Dashboard")
        self.geometry("600x600")
        self.db = Database(db_file)

        self.mqtt_settings = {
            "broker": "158.180.44.197",
            "port": 1883,
            "topic": "aut/GroJuTu/fillLevel/1",
            "username": "bobm",
            "password": "letmein"
        }

        self.frames = {}
        for Page in (HomePage, MQTTSettings, DataPlot, ModelPredictor, ClassificationPredictor):
            frame = Page(self, self)
            self.frames[Page.__name__] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_page("HomePage")

    def show_page(self, name):
        frame = self.frames[name]
        frame.tkraise()
        if name == "DataPlot":
            frame.load_table_names()

    def update_mqtt(self, broker, port, topic, username, password):
        self.mqtt_settings.update({
            "broker": broker,
            "port": int(port),
            "topic": topic,
            "username": username,
            "password": password
        })

    def on_disconnect(self, client=None, userdata=None, rc=0):
        messagebox.showerror("Error", "MQTT disconnected unexpectedly!")

    def on_timeout(self):
        messagebox.showerror("Timeout", "No MQTT messages for 20 seconds.")


class HomePage(tk.Frame):
    def __init__(self, root, app):
        super().__init__(root)
        self.app = app
        tk.Label(self, text="Welcome!", font=("Arial", 16)).pack(pady=10)

        tk.Button(self, text="MQTT Setup", command=lambda: app.show_page("MQTTSettings")).pack(pady=5)
        tk.Button(self, text="Plot Data", command=lambda: app.show_page("DataPlot")).pack(pady=5)
        tk.Button(self, text="Run Prediction", command=lambda: app.show_page("ModelPredictor")).pack(pady=5)
        tk.Button(self, text="Classify Bottles", command=lambda: app.show_page("ClassificationPredictor")).pack(pady=5)

        self.rec_btn = tk.Button(self, text="Start MQTT", command=self.toggle_record)
        self.rec_btn.pack(pady=20)
        self.recording = False

        tk.Button(self, text="Prepare Data", command=self.prepare_data).pack(pady=5)

    def prepare_data(self):
    # formatted_data.csv & formatted_data_plot.json
        Database("data.json").export_and_prepare_data()
        messagebox.showinfo("Datenbereitstellung", "Dateien wurden erstellt:\n- formatted_data.csv\n- formatted_data_plot.json")

    def toggle_record(self):
        self.recording = not self.recording
        if self.recording:
            self.rec_btn.config(text="Stop MQTT")
            mqtt = CustomMQTTClient(**self.app.mqtt_settings)
            mqtt.register_timeout_callback(self.app.on_timeout)
            mqtt.client.on_disconnect = self.app.on_disconnect
            self.mqtt = mqtt
            self.thread = threading.Thread(target=mqtt.begin)
            self.thread.start()
        else:
            self.rec_btn.config(text="Start MQTT")
            self.mqtt.terminate()
            self.thread.join()


class MQTTSettings(tk.Frame):
    def __init__(self, root, app):
        super().__init__(root)
        self.app = app
        tk.Label(self, text="MQTT Configuration", font=("Arial", 14)).pack(pady=10)

        self.entries = {}
        for field in ["broker", "port", "topic", "username", "password"]:
            tk.Label(self, text=field.capitalize()).pack()
            entry = tk.Entry(self, show="*" if field == "password" else None)
            entry.insert(0, str(app.mqtt_settings[field]))
            entry.pack()
            self.entries[field] = entry

        tk.Button(self, text="Save", command=self.save).pack(pady=10)
        tk.Button(self, text="Back", command=lambda: app.show_page("HomePage")).pack()

    def save(self):
        values = {key: entry.get() for key, entry in self.entries.items()}
        self.app.update_mqtt(**values)
        messagebox.showinfo("Saved", "MQTT settings updated!")


class DataPlot(tk.Frame):
    def __init__(self, root, app):
        super().__init__(root)
        self.app = app
        self.table_var = tk.StringVar()
        self.plot_type = tk.StringVar(value="line")
        self.canvas = None

        tk.Label(self, text="Choose Table").pack()
        self.table_menu = tk.OptionMenu(self, self.table_var, "")
        self.table_menu.pack()

        tk.Label(self, text="Plot Type").pack()
        tk.OptionMenu(self, self.plot_type, "line", "bar", "scatter").pack()

        tk.Button(self, text="Plot", command=self.draw_plot).pack(pady=10)
        tk.Button(self, text="Back", command=lambda: app.show_page("HomePage")).pack()

    def load_table_names(self):
        tables = self.app.db.list_all_tables()
        menu = self.table_menu["menu"]
        menu.delete(0, 'end')

        if not tables:
            self.table_var.set("")  
            return

        for table in tables:
            menu.add_command(label=table, command=lambda v=table: self.table_var.set(v))

        self.table_var.set(tables[0])

    def draw_plot(self):
        df = self.app.db.fetch_table_data(self.table_var.get()).tail(50)

        if 'datetime' not in df.columns or 'value' not in df.columns:
            messagebox.showerror("Error", "Selected table doesn't contain 'datetime' and 'value' columns.")
            return

        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        df.dropna(subset=['datetime'], inplace=True)

        fig, ax = plt.subplots(figsize=(6, 4))
        plot_type = self.plot_type.get()

        if plot_type == "line":
            ax.plot(df['datetime'], df['value'])
        elif plot_type == "bar":
            ax.bar(df['datetime'], df['value'])
        elif plot_type == "scatter":
            ax.scatter(df['datetime'], df['value'])

        ax.set_title("Data Plot")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.grid(True)

        if self.canvas:
            self.canvas.get_tk_widget().destroy()

        self.canvas = FigureCanvasTkAgg(fig, self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(pady=20)



class ModelPredictor(tk.Frame):
    def __init__(self, root, app):
        super().__init__(root)
        self.model = None
        self.train_path = None
        self.predict_path = None

        tk.Label(self, text="Prediction Module", font=("Arial", 14)).pack(pady=10)

        tk.Button(self, text="Load Training CSV", command=self.load_train_csv).pack(pady=5)
        tk.Button(self, text="Train Model", command=self.train_model).pack(pady=5)
        self.status = tk.Label(self, text="")
        self.status.pack()

        tk.Button(self, text="Load Data for Prediction", command=self.load_predict_csv).pack(pady=5)
        tk.Button(self, text="Run Prediction", command=self.run_prediction).pack(pady=5)
        tk.Button(self, text="Save Output", command=self.save_output).pack(pady=5)

        self.result = tk.Label(self, text="")
        self.result.pack(pady=10)

        tk.Button(self, text="Back", command=lambda: app.show_page("HomePage")).pack()

    def load_train_csv(self):
        file = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file:
            self.train_path = file
            self.model = WeightPredictionModel(file)
            self.status.config(text="Training file loaded.")

    def train_model(self):
        if self.model:
            self.model.fit()
            train_mse, test_mse, _ = self.model.get_performance_metrics()
            self.status.config(text=f"Model ready. Train MSE: {train_mse:.2f}, Test MSE: {test_mse:.2f}")

    def load_predict_csv(self):
        file = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file:
            self.predict_path = file
            self.result.config(text="Prediction file ready.")

    def run_prediction(self):
        if self.model and self.predict_path:
            self.model.make_prediction(self.predict_path)
            self.result.config(text="Prediction completed.")

    def save_output(self):
        matriculation_numbers = ["662200914", "52111901", "52216068"]  # <-- Matrikelnummern

        if not hasattr(self.model, "predicted_dataset"):
            self.model.make_prediction(self.predict_path)

        filename = self.model.export_predictions(matriculation_numbers)
        messagebox.showinfo("Export erfolgreich", f"Datei gespeichert: {filename}")


class ClassificationPredictor(tk.Frame):
    def __init__(self, root, app):
        super().__init__(root)
        self.app = app
        self.data_path = None

        tk.Label(self, text="Bottle Defect Classification", font=("Arial", 14)).pack(pady=10)

        tk.Button(self, text="Load CSV for Classification", command=self.load_csv).pack(pady=5)
        tk.Button(self, text="Run Classification", command=self.run_classification).pack(pady=5)

        self.status = tk.Label(self, text="")
        self.status.pack(pady=5)

        self.result_table = tk.Text(self, height=15, width=75)
        self.result_table.pack(pady=10)

        tk.Button(self, text="Back", command=lambda: app.show_page("HomePage")).pack()

    def load_csv(self):
        file = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file:
            self.data_path = file
            self.status.config(text="Classification data loaded.")

    def run_classification(self):
        if not self.data_path:
            messagebox.showerror("Fehler", "Bitte zuerst eine CSV-Datei laden.")
            return

        import pandas as pd
        import os
        import matplotlib.pyplot as plt
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay

        df = pd.read_csv(self.data_path)

        # Umbenennen, falls 'value' vorhanden ist (von drop_vibration.csv)
        if 'value' in df.columns:
            df.rename(columns={"value": "drop_vibration"}, inplace=True)

        # Stelle sicher, dass drop_vibration und is_cracked existieren
        if 'drop_vibration' not in df.columns or 'is_cracked' not in df.columns:
            messagebox.showerror("Fehler", "CSV benötigt Spalten: 'drop_vibration' und 'is_cracked'")
            return

        # Konvertiere Spalten
        df['drop_vibration'] = pd.to_numeric(df['drop_vibration'], errors='coerce')
        df['is_cracked'] = pd.to_numeric(df['is_cracked'], errors='coerce')

        df.dropna(subset=['drop_vibration', 'is_cracked'], inplace=True)

        # Zusätzliche Features berechnen
        df['std_vibration'] = df['drop_vibration'].rolling(window=3).std()
        df['lag_1'] = df['drop_vibration'].shift(1)
        df.dropna(inplace=True)

        if len(df) < 10:
            messagebox.showerror("Fehler", "Nicht genug Datenpunkte (mind. 10) nach Bereinigung.")
            return

        y = df['is_cracked']

        feature_sets = {
            'mean()': ['drop_vibration'],
            'mean(), std()': ['drop_vibration', 'std_vibration'],
            'mean(), lag_1': ['drop_vibration', 'lag_1'],
        }

        models = {
            'Logistic Regression': LogisticRegression(max_iter=200),
            'kNN': KNeighborsClassifier(),
            'Decision Tree': DecisionTreeClassifier()
        }

        results = []
        os.makedirs("confusion_matrices", exist_ok=True)

        for feat_label, features in feature_sets.items():
            X = df[features]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            for model_name, model in models.items():
                model.fit(X_train, y_train)
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)

                train_f1 = f1_score(y_train, train_pred)
                test_f1 = f1_score(y_test, test_pred)
                cm = confusion_matrix(y_test, test_pred)

                fig, ax = plt.subplots()
                disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                disp.plot(ax=ax)
                ax.set_title(f"{model_name} ({feat_label})")
                filename = f"confusion_matrices/cm_{model_name.replace(' ', '_')}_{feat_label.replace(',', '').replace('()', '').replace(' ', '_')}.png"
                plt.savefig(filename)
                plt.close()

                results.append({
                    "Features": feat_label,
                    "Model": model_name,
                    "F1-Train": round(train_f1, 2),
                    "F1-Test": round(test_f1, 2)
                })

        # Ergebnisse speichern
        result_df = pd.DataFrame(results)
        result_df.to_csv("classification_results.csv", index=False)

        # Im UI anzeigen
        self.result_table.delete(1.0, tk.END)
        self.result_table.insert(tk.END, f"{'Features':<20} {'Model':<20} {'F1-Train':<10} {'F1-Test':<10}\n")
        self.result_table.insert(tk.END, "-" * 65 + "\n")
        for r in results:
            self.result_table.insert(tk.END, f"{r['Features']:<20} {r['Model']:<20} {r['F1-Train']:<10} {r['F1-Test']:<10}\n")

        self.status.config(text="Ergebnisse gespeichert in classification_results.csv")
        messagebox.showinfo("Fertig", "Klassifikation abgeschlossen.\nErgebnisse: classification_results.csv")




if __name__ == "__main__":
    SimpleApp("formatted_data_plot.json").mainloop()

class BottleDefectClassifier:
    def __init__(self, filepath):
        import numpy as np
        self.df = pd.read_csv(filepath)

        # Neue Feature-Spalte aus gegebenen Vibrationsspalten erzeugen
        self.df['drop_vibration'] = self.df[[
            'vibration_index_red', 'vibration_index_green', 'vibration_index_blue'
        ]].mean(axis=1)

        # Label generieren (z. B. Schwellenwert für Defekt anpassen)
        self.df['defect_label'] = (self.df['drop_vibration'] > 0.3).astype(int)

        # Feature Engineering
        self.df['std_vibration'] = self.df['drop_vibration'].rolling(window=3).std()
        self.df['lag_1'] = self.df['drop_vibration'].shift(1)
        self.df.dropna(inplace=True)

        self.feature_set = ['drop_vibration', 'std_vibration', 'lag_1']
        self.X = self.df[self.feature_set]
        self.y = self.df['defect_label']

    def train_and_evaluate(self):
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import f1_score, confusion_matrix

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        model = LogisticRegression(max_iter=200)
        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        train_f1 = f1_score(y_train, train_pred)
        test_f1 = f1_score(y_test, test_pred)
        cm = confusion_matrix(y_test, test_pred)

        return {
            "model": model,
            "train_f1": train_f1,
            "test_f1": test_f1,
            "confusion_matrix": cm
        }
