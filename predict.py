import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class WeightPredictionModel:
    def __init__(self, file_path):
        # Load dataset and remove unwanted columns
        self.dataset = pd.read_csv(file_path).dropna()
        self.dataset = self.dataset.loc[:, ~self.dataset.columns.str.startswith('Unnamed')]
        
        self.chosen_model = None
        self.model_type = None
        self.model_coeffs = None
        self.model_intercept = None

    def fit(self):
        # Split features and target variable
        target = self.dataset['final_weight_grams']
        features = self.dataset.drop(columns=['final_weight_grams'])
        self.features = features
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        return self._compare_models()

    def _compare_models(self):
        algorithms = {
            "LinearRegression": LinearRegression(),
            "RidgeRegression": Ridge(),
            "SupportVectorRegression": SVR()
        }

        best_mse = float("inf")
        results_summary = []

        for name, model in algorithms.items():
            model.fit(self.X_train, self.y_train)
            train_predictions = model.predict(self.X_train)
            test_predictions = model.predict(self.X_test)

            train_error = mean_squared_error(self.y_train, train_predictions)
            test_error = mean_squared_error(self.y_test, test_predictions)

            results_summary.append({
                "Model": name,
                "Train MSE": train_error,
                "Test MSE": test_error
            })

            if test_error < best_mse:
                best_mse = test_error
                self.model_type = name
                self.chosen_model = model

        if hasattr(self.chosen_model, 'coef_'):
            self.model_coeffs = self.chosen_model.coef_
        if hasattr(self.chosen_model, 'intercept_'):
            self.model_intercept = self.chosen_model.intercept_

        return results_summary, self.model_type, best_mse

    def get_performance_metrics(self):
        if not self.chosen_model:
            raise RuntimeError("No model has been trained yet.")
        train_err, test_err = self._calculate_mse()
        mse_df = self._featurewise_mse()
        return train_err, test_err, mse_df

    def _calculate_mse(self):
        y_train_pred = self.chosen_model.predict(self.X_train)
        y_test_pred = self.chosen_model.predict(self.X_test)
        train_error = mean_squared_error(self.y_train, y_train_pred)
        test_error = mean_squared_error(self.y_test, y_test_pred)
        return train_error, test_error

    def _featurewise_mse(self):
        mse_scores = {}
        target = self.dataset['final_weight_grams']
        
        for feature in self.dataset.columns.drop('final_weight_grams'):
            X = self.dataset[[feature]]
            y = target
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mse_scores[feature] = mse

        mse_df = pd.DataFrame(mse_scores.items(), columns=["Feature", "MSE"])
        return mse_df

    def make_prediction(self, new_data_path):
        new_data = pd.read_csv(new_data_path)
        new_data = new_data.loc[:, ~new_data.columns.str.startswith('Unnamed')]

        rename_map = {
            'aut_GroJuTu_temperature_blue': 'temperature_blue',
            'aut_GroJuTu_fill_level_grams_red': 'fill_level_grams_red',
            'aut_GroJuTu_fill_level_grams_green': 'fill_level_grams_green',
            'aut_GroJuTu_fill_level_grams_blue': 'fill_level_grams_blue',
            'aut_GroJuTu_vibration-index_red': 'vibration_red',
            'aut_GroJuTu_vibration-index_green': 'vibration_green',
            'aut_GroJuTu_vibration-index_blue': 'vibration_blue',
        }
        new_data = new_data.rename(columns=rename_map)

        expected_cols = self.features.columns.tolist()
        missing = set(expected_cols) - set(new_data.columns)

        if missing:
            raise ValueError(f"Missing columns in prediction data: {missing}")

        # correct order
        new_data = new_data[expected_cols]

        preds = self.chosen_model.predict(new_data)
        self.predicted_dataset = new_data.copy()
        self.predicted_dataset['predicted_final_weight'] = preds

    def export_predictions(self, student_ids):
        if not hasattr(self, 'predicted_dataset'):
            raise RuntimeError("No predictions to export. Run make_prediction() first.")
        
        output_file = f"reg_{'-'.join(student_ids)}.csv"
        self.predicted_dataset.to_csv(output_file, index=False)
        return output_file


    def display_model_equation(self):
        if self.model_coeffs is None or self.model_intercept is None:
            raise RuntimeError("Model coefficients not found. Train the model first.")

        terms = [f"{coef:.4f}*x{i}" for i, coef in enumerate(self.model_coeffs, start=1)]
        equation = " + ".join(terms) + f" + {self.model_intercept:.4f}"
        return f"Prediction Equation: y = {equation}"
