import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

class BottleDefectClassifier:
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath)

        self.df.rename(columns={"value": "drop_vibration"}, inplace=True)
        

        # Feature 
        self.df['std_vibration'] = self.df['drop_vibration'].rolling(window=3).std()
        self.df['lag_1'] = self.df['drop_vibration'].shift(1)
        self.df.dropna(inplace=True)

        self.y = self.df['defect_label']

    def train_models(self):
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

        for feature_label, features in feature_sets.items():
            X = self.df[features]
            X_train, X_test, y_train, y_test = train_test_split(X, self.y, test_size=0.2, random_state=42)

            for model_name, model in models.items():
                model.fit(X_train, y_train)
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)

                train_f1 = f1_score(y_train, train_pred)
                test_f1 = f1_score(y_test, test_pred)
                cm = confusion_matrix(y_test, test_pred)

                # Confusion Matrix
                fig, ax = plt.subplots()
                disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                disp.plot(ax=ax)
                ax.set_title(f"CM: {model_name} ({feature_label})")
                filename = f"confusion_matrices/confmat_{model_name}_{feature_label.replace(',', '').replace('()','').replace(' ', '_')}.png"
                plt.savefig(filename)
                plt.close()

                results.append({
                    'Features': feature_label,
                    'Model': model_name,
                    'F1 Train': round(train_f1, 2),
                    'F1 Test': round(test_f1, 2)
                })

        results_df = pd.DataFrame(results)
        results_df.to_csv("classification_results.csv", index=False)
        print("classification_results.csv gespeichert.")
        print("Confusion-Matrix-Bilder in 'confusion_matrices/' erstellt.")

if __name__ == "__main__":
    clf = BottleDefectClassifier("X.csv")  
    clf.train_models()

