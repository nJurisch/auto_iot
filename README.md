# Automatisierung_iot_final

Dies ist ein Python-basiertes IoT-System zur Erfassung, Speicherung, Visualisierung und Analyse von Füllstand- und Vibrationsdaten. Es umfasst MQTT-Kommunikation, Datenbankintegration, GUI-Anzeige, Vorhersagemodelle (Regression) und Klassifikation (Defekterkennung). 
├── data/ # Empfangene MQTT-Daten (CSV) 
├── confusion_matrices/ # Confusion-Matrix-Bilder 
├── pictures/ # Beispielplots 
├── userinterface.py # GUI-Anwendung 
├── simulate_sender.py # MQTT-Testsender 
|── train_classifier.py # Klassifikationstraining 
├── run_bottle_classifier.py # Klassifikation ausführen 
├── predict.py # Regressionsmodell 
├── formatted_data.csv/json # Datenbereitstellung 
├── classification_results.csv # Klassifikationsergebnisse 
├── reg_.csv # Regressionsvorhersagen
├── vibration_with_labels.csv #Klassifikationsgrundlage

Repository Clonen
"git clone https://github.com/nJurisch/auto_iot"

Voraussetzungen
-Python -Paketliste in "requirements.txt"

Installation: pip install -r requirements.txt

Ausführen
Aufgabe 1: python iot1.py [Senden von einfachen Werten als Retain] Userinterface: python userinterface.py [Hauptanwendung]

Andere Aufgaben werden wie in "Projektbericht_GroJuTu.md" gelöst mithilfe von userinterface.py

Notiz:
folgende Dateien sind noch vorhanden, da signifikante Arbeitszeit in diese geflossen ist, um das ausfallen von "iot1/teaching_factory" zu kompensieren.
MQTT-Datensimulation: python simulate_sender.py [Zufalls generierte Werte, da iot1/teaching_factory nicht gesendet hat] 
Klassifikation seperat trainieren: python train_classifier.py [kann alleinstehend genutzt werden]
