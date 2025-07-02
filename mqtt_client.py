import paho.mqtt.client as mqtt
from database_storage import SensorDatabaseHandler as Database
import json
import threading
import time

class CustomMQTTClient:
    def __init__(self, broker, port, topic, username=None, password=None):
        self.broker = broker
        self.port = port
        self.topic = topic
        self.username = username
        self.password = password

        self.db = Database('data.json')
        self.client = mqtt.Client()

        if self.username and self.password:
            self.client.username_pw_set(self.username, self.password)

        self.client.on_connect = self._handle_connect
        self.client.on_message = self._handle_message
        self.client.on_disconnect = self._handle_disconnect

        self.timeout_limit = 20  # seconds
        self.last_received_time = time.time()
        self.timeout_monitor = threading.Thread(target=self._monitor_timeout)
        self.timeout_callback = None

    def _handle_connect(self, client, userdata, flags, result_code):
        if result_code == 0:
            print("MQTT connection established.")
            client.subscribe(self.topic)
        else:
            print(f"Connection error, code: {result_code}")

    def _handle_disconnect(self, client, userdata, result_code):
        print("Disconnected from broker.")
        if result_code != 0:
            print("Unexpected disconnect detected.")

    def _handle_message(self, client, userdata, message):
        decoded_payload = message.payload.decode().strip()
        print("New message received:")
        print(decoded_payload, "\n")

        self.last_received_time = time.time()

        if not decoded_payload or not decoded_payload.startswith("{"):
            print("Nicht-JSON Nachricht – wird ignoriert.")
            return

        try:
            parsed = json.loads(decoded_payload)

            topic = message.topic

            # --- Topic-bezogene Extraktion ---
            if topic.endswith("dispenser_red"):
                self.db.store_data({"value": parsed["fill_level_grams"]}, "aut_GroJuTu_fill_level_grams_red")
                self.db.store_data({"value": parsed["vibration-index"]}, "aut_GroJuTu_vibration-index_red")

            elif topic.endswith("dispenser_green"):
                self.db.store_data({"value": parsed["fill_level_grams"]}, "aut_GroJuTu_fill_level_grams_green")
                self.db.store_data({"value": parsed["vibration-index"]}, "aut_GroJuTu_vibration-index_green")

            elif topic.endswith("dispenser_blue"):
                self.db.store_data({"value": parsed["fill_level_grams"]}, "aut_GroJuTu_fill_level_grams_blue")
                self.db.store_data({"value": parsed["vibration-index"]}, "aut_GroJuTu_vibration-index_blue")

            elif topic.endswith("temperature"):
                self.db.store_data({"value": parsed["temperature_C"]}, "aut_GroJuTu_temperature_blue")

            elif topic.endswith("final_weight"):
                self.db.store_data({"value": parsed["final_weight"]}, "aut_GroJuTu_scale_final_weight")
            elif topic.endswith("ground_truth"):
                self.db.store_data({"value": parsed["is_cracked"]}, "iot1_teaching_machine_ground_truth")
            elif topic.endswith("drop_oscillation"):
                self.db.store_data({"value": parsed["drop_oscillation"]}, "iot1_teaching_machine_drop_oscillation")
            else:
                print(f"Topic nicht erkannt: {topic}")

        except json.JSONDecodeError as err:
            print(f"JSON decoding failed: {err}")
        except Exception as e:
            print(f"Fehler bei der Verarbeitung: {e}")

    def begin(self):
        try:
            self.client.connect(self.broker, self.port, keepalive=60)
            self.timeout_monitor.start()
            self.client.loop_forever()
        except Exception as error:
            print(f"Unable to connect to MQTT broker: {error}")

    def terminate(self):
        self.client.disconnect()

    def _monitor_timeout(self):
        while True:
            elapsed = time.time() - self.last_received_time
            if elapsed > self.timeout_limit:
                if self.timeout_callback:
                    self.timeout_callback()
                break
            time.sleep(1)

    def register_timeout_callback(self, callback_function):
        self.timeout_callback = callback_function

