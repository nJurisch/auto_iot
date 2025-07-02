import os
import pandas as pd
from tinydb import TinyDB
from datetime import datetime, timezone

class SensorDatabaseHandler:
    def __init__(self, file_path):
        self.db_instance = TinyDB(file_path)

    def insert_entry(self, data_entry, topic_label):
        print(f"Inserting into '{topic_label}': {data_entry}")
        self.db_instance.table(topic_label).insert(data_entry)

    def fetch_table_data(self, table_key):
        raw_table = self.db_instance.table(table_key)
        all_entries = []
        for record in raw_table.all():
            for k, val in record.items():
                timestamp, reading = val
                all_entries.append({'datetime': timestamp, 'value': reading})
        return pd.DataFrame(all_entries)

    def list_all_tables(self):
        return self.db_instance.tables()

    def store_data(self, data, topic_name):
        if 'time' not in data and 'datetime' not in data:
            data['time'] = datetime.now(timezone.utc).isoformat()
        print(f"Storing data in topic {topic_name}: {data}")
        self.db_instance.table(topic_name).insert(data)

    def export_and_prepare_data(self):
        output_dir = 'data'
        os.makedirs(output_dir, exist_ok=True)
        table_dataframes = {}

        # 1. Export all TinyDB tables
        for table in self.db_instance.tables():
            safe_name = "_".join(table.split("/"))
            df = pd.DataFrame(self.db_instance.table(table).all())
            csv_path = os.path.join(output_dir, f"{safe_name}.csv")
            print(f"Exporting: {table}")
            df.to_csv(csv_path, index=False)

            try:
                table_dataframes[safe_name] = pd.read_csv(csv_path)
            except Exception as e:
                print(f"Skipping {csv_path}: {e}")

        # 2. Mapping fuer regression
        mapping = {
            'aut_GroJuTu_fill_level_grams_red': 'fill_level_grams_red',
            'aut_GroJuTu_fill_level_grams_green': 'fill_level_grams_green',
            'aut_GroJuTu_fill_level_grams_blue': 'fill_level_grams_blue',
            'aut_GroJuTu_vibration-index_red': 'vibration_index_red',
            'aut_GroJuTu_vibration-index_green': 'vibration_index_green',
            'aut_GroJuTu_vibration-index_blue': 'vibration_index_blue',
            'aut_GroJuTu_temperature_blue': 'temperature_blue',
            'aut_GroJuTu_scale_final_weight': 'final_weight_grams',
        }

        # 3. Drop oscillation speichern
        drop_df = table_dataframes.get('iot1_teaching_machine_drop_oscillation')
        if drop_df is not None and 'value' in drop_df.columns:
            drop_df[['value']].to_csv('drop_vibration.csv', index=False)
            print("Saved to drop_vibration.csv")

        # 4. Kombiniertes Label-Dataset
        ground_df = table_dataframes.get('iot1_teaching_machine_ground_truth')
        if drop_df is not None and ground_df is not None:
            drop_expanded = []
            label_expanded = []

            for drop_row, gt_row in zip(drop_df.itertuples(), ground_df.itertuples()):
                drop_val = drop_row.value
                gt_val = gt_row.value
                try:
                    if isinstance(drop_val, str) and drop_val.startswith("["):
                        values = eval(drop_val)
                    elif isinstance(drop_val, list):
                        values = drop_val
                    else:
                        values = [drop_val]
                    numeric_values = [float(v) for v in values]
                    label = int(gt_val)
                    drop_expanded.extend(numeric_values)
                    label_expanded.extend([label] * len(numeric_values))
                except Exception as e:
                    print(f"Fehler beim Verarbeiten eines Eintrags: {e}")

            if drop_expanded and label_expanded:
                combined_df = pd.DataFrame({
                    'drop_vibration': drop_expanded,
                    'is_cracked': label_expanded
                })
                combined_df.to_csv('vibration_with_labels.csv', index=False)
                print(f"Saved vibration_with_labels.csv mit {len(combined_df)} Zeilen.")
            else:
                print("Keine gültigen Einträge für vibration_with_labels.csv erzeugt.")
        else:
            print("drop_oscillation oder ground_truth Tabelle fehlt oder leer.")


        # 5. Regressionstabelle
        prepared_df = pd.DataFrame()
        for topic_key, column_name in mapping.items():
            df = table_dataframes.get(topic_key)
            if df is not None and 'value' in df.columns:
                values = df['value'].reset_index(drop=True)
                prepared_df[column_name] = values
            else:
                print(f"Missing data for: {column_name}")
                prepared_df[column_name] = pd.NA

        prepared_df = prepared_df[list(mapping.values())]
        prepared_df.to_csv('formatted_data.csv', index=False)
        print("Saved to formatted_data.csv")

        # 6. Zeitreihenstruktur für Plot
        plot_json_path = 'formatted_data_plot.json'
        formatted_db = TinyDB(plot_json_path)

        for topic_key, df in table_dataframes.items():
            time_col = None
            for col in ['time', 'datetime']:
                if col in df.columns:
                    time_col = col
                    break

            if time_col and 'value' in df.columns:
                df[time_col] = pd.to_datetime(df[time_col], errors='coerce').astype(str)
                records = list(zip(df[time_col], df['value']))
                reformatted = [{topic_key: record} for record in records]
                formatted_db.table(topic_key).insert_multiple(reformatted)
            else:
                print(f"Skipping {topic_key}: missing time and/or value column")

        formatted_db.close()
        print("Saved plot-compatible data to formatted_data_plot.json")


