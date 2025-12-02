"""
Hardware Interface Module for Real-time Sensor Data Collection
Supports multiple sensor types including ultrasonic, gas, temperature, humidity, etc.
Compatible with Arduino, Raspberry Pi, and direct sensor connections
"""

import serial
import time
import json
import threading
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Callable
import logging
from abc import ABC, abstractmethod

# NodeMCU/ESP8266/ESP32 focused - no Raspberry Pi dependencies
# All sensor interfacing will be done through Arduino/NodeMCU via Serial/WiFi
NODEMCU_AVAILABLE = True  # Always available for NodeMCU development

from database import SensorDatabase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SensorInterface(ABC):
    """Abstract base class for sensor interfaces"""
    
    @abstractmethod
    def read_data(self) -> Dict:
        """Read data from sensor and return as dictionary"""
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if sensor is connected and responding"""
        pass

class UltrasonicSensor(SensorInterface):
    """
    Ultrasonic sensor interface (HC-SR04 compatible) for NodeMCU
    Measures distance using ultrasonic waves via NodeMCU/Arduino
    """
    
    def __init__(self, trigger_pin: int = 5, echo_pin: int = 4, 
                 max_distance: float = 400.0, name: str = "ultrasonic_1"):
        """
        Initialize ultrasonic sensor for NodeMCU
        
        Args:
            trigger_pin: NodeMCU GPIO pin for trigger (e.g., D1 = GPIO5)
            echo_pin: NodeMCU GPIO pin for echo (e.g., D2 = GPIO4)
            max_distance: Maximum measurement distance in cm
            name: Sensor identifier name
        """
        self.trigger_pin = trigger_pin
        self.echo_pin = echo_pin
        self.max_distance = max_distance
        self.name = name
        self.connected = True  # Assume connected via NodeMCU
        
        logger.info(f"Ultrasonic sensor {name} configured for NodeMCU pins D{self._gpio_to_d_pin(trigger_pin)}/D{self._gpio_to_d_pin(echo_pin)}")
    
    def _gpio_to_d_pin(self, gpio_num):
        """Convert GPIO number to NodeMCU D pin for user reference"""
        gpio_to_d = {16: 0, 5: 1, 4: 2, 0: 3, 2: 4, 14: 5, 12: 6, 13: 7, 15: 8}
        return gpio_to_d.get(gpio_num, gpio_num)
    
    def read_data(self) -> Dict:
        """Read distance measurement (simulation for NodeMCU interface)"""
        # Simulation mode - NodeMCU will provide real data via Serial/WiFi
        # This generates realistic ultrasonic data for testing
        distance = np.random.normal(50, 5)  # 50cm ± 5cm
        distance = max(2, min(self.max_distance, distance))  # Clamp to valid range
        
        return {
            'sensor_name': self.name,
            'sensor_type': 'ultrasonic', 
            'sensor_model': 'HC-SR04',
            'distance_cm': round(distance, 2),
            'trigger_pin': f"D{self._gpio_to_d_pin(self.trigger_pin)}",
            'echo_pin': f"D{self._gpio_to_d_pin(self.echo_pin)}",
            'timestamp': datetime.now().isoformat(),
            'status': 'nodemcu_simulation'
        }
    
    def is_connected(self) -> bool:
        """Check sensor connection"""
        return self.connected

class GasSensor(SensorInterface):
    """
    Gas sensor interface (MQ series compatible)
    Supports various gas types: CO, CO2, methane, alcohol, etc.
    """
    
    def __init__(self, analog_pin: str = "A0", sensor_type: str = "MQ-2", 
                 gas_type: str = "general", name: str = "gas_1",
                 calibration_factor: float = 1.0):
        """
        Initialize gas sensor for NodeMCU
        
        Args:
            analog_pin: NodeMCU analog pin ("A0" for NodeMCU v1.0, "A0" for ESP32)
            sensor_type: Type of MQ sensor (MQ-2, MQ-7, MQ-135, etc.)
            gas_type: Type of gas being measured
            name: Sensor identifier name
            calibration_factor: Calibration factor for concentration calculation
        """
        self.analog_pin = analog_pin
        self.sensor_type = sensor_type
        self.gas_type = gas_type
        self.name = name
        self.calibration_factor = calibration_factor
        self.connected = True  # Assume connected via NodeMCU
        
        # Gas sensor characteristics for MQ series
        self.characteristics = {
            'MQ-2': {'range': (200, 10000), 'unit': 'ppm', 'gases': ['LPG', 'methane', 'hydrogen']},
            'MQ-7': {'range': (10, 500), 'unit': 'ppm', 'gases': ['carbon_monoxide']},
            'MQ-135': {'range': (10, 1000), 'unit': 'ppm', 'gases': ['CO2', 'ammonia', 'benzene']},
            'MQ-9': {'range': (10, 1000), 'unit': 'ppm', 'gases': ['carbon_monoxide', 'methane']},
            'MQ-3': {'range': (10, 500), 'unit': 'ppm', 'gases': ['alcohol', 'ethanol']},
            'MQ-4': {'range': (200, 10000), 'unit': 'ppm', 'gases': ['methane', 'natural_gas']},
            'MQ-6': {'range': (200, 10000), 'unit': 'ppm', 'gases': ['butane', 'LPG']},
            'MQ-8': {'range': (100, 10000), 'unit': 'ppm', 'gases': ['hydrogen']}
        }
        
        logger.info(f"Gas sensor {name} ({sensor_type}) configured for NodeMCU pin {analog_pin}")
    
    def read_data(self) -> Dict:
        """Read gas concentration (simulation for NodeMCU interface)"""
        # Simulation mode - NodeMCU will provide real data via Serial/WiFi
        # Generate realistic gas sensor data based on sensor type
        
        if self.sensor_type in self.characteristics:
            char = self.characteristics[self.sensor_type]
            min_range, max_range = char['range']
            # Generate concentration within sensor range
            base_ppm = min_range + (max_range - min_range) * 0.1  # 10% of range as baseline
            concentration = np.random.normal(base_ppm, base_ppm * 0.2)
            concentration = max(0, min(max_range, concentration))
        else:
            concentration = np.random.normal(50, 10)
            concentration = max(0, concentration)
        
        # Simulate ADC reading (NodeMCU: 0-1024 for 3.3V)
        voltage = np.random.uniform(0.5, 3.0)  # NodeMCU 3.3V system
        raw_adc = int((voltage / 3.3) * 1024)  # 10-bit ADC
        
        return {
            'sensor_name': self.name,
            'sensor_type': 'gas',
            'gas_sensor_model': self.sensor_type,
            'gas_type': self.gas_type,
            'concentration_ppm': round(concentration * self.calibration_factor, 2),
            'voltage': round(voltage, 3),
            'raw_adc': raw_adc,
            'analog_pin': self.analog_pin,
            'timestamp': datetime.now().isoformat(),
            'status': 'nodemcu_simulation'
        }
    
    def is_connected(self) -> bool:
        """Check sensor connection"""
        return self.connected
    
    def calibrate(self, known_concentration: float, readings: List[float] = None):
        """
        Calibrate the sensor with known gas concentration
        
        Args:
            known_concentration: Known concentration in ppm
            readings: List of voltage readings (if None, takes current reading)
        """
        if readings is None:
            # Take multiple readings for better calibration
            readings = []
            for _ in range(10):
                data = self.read_data()
                if data['status'] == 'ok':
                    readings.append(data['voltage'])
                time.sleep(0.5)
        
        if readings:
            avg_voltage = np.mean(readings)
            if avg_voltage > 0:
                # Update calibration factor
                current_calc = avg_voltage * 100 * self.calibration_factor
                if current_calc > 0:
                    self.calibration_factor = (known_concentration / current_calc) * self.calibration_factor
                    logger.info(f"Calibrated {self.name} with factor {self.calibration_factor:.3f}")

class NodeMCUInterface:
    """
    Interface for NodeMCU (ESP8266/ESP32) sensor systems
    Communicates via Serial/USB or WiFi connection
    """
    
    def __init__(self, port: str = '/dev/ttyUSB0', baudrate: int = 115200, 
                 timeout: float = 1.0, wifi_ip: str = None):
        """
        Initialize NodeMCU interface
        
        Args:
            port: Serial port (e.g., 'COM3' on Windows, '/dev/ttyUSB0' on Linux)
            baudrate: Communication speed (115200 typical for NodeMCU)
            timeout: Serial timeout in seconds
            wifi_ip: WiFi IP address for wireless communication (optional)
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.wifi_ip = wifi_ip
        self.serial_conn = None
        self.connected = False
        self.connection_type = 'serial'  # 'serial' or 'wifi'
        
        self.connect()
    
    def connect(self):
        """Establish connection with NodeMCU (Serial or WiFi)"""
        if self.wifi_ip:
            # TODO: Implement WiFi connection
            self.connection_type = 'wifi'
            logger.info(f"WiFi connection to NodeMCU not yet implemented: {self.wifi_ip}")
            self.connected = False
        else:
            try:
                self.serial_conn = serial.Serial(
                    port=self.port,
                    baudrate=self.baudrate,
                    timeout=self.timeout
                )
                time.sleep(3)  # Wait for NodeMCU to reset (ESP8266 needs more time)
                self.connected = True
                self.connection_type = 'serial'
                logger.info(f"Connected to NodeMCU on {self.port} at {self.baudrate} baud")
            except Exception as e:
                logger.error(f"Failed to connect to NodeMCU: {e}")
                self.connected = False
    
    def read_sensors(self) -> Dict:
        """Read all sensors connected to NodeMCU"""
        if not self.connected:
            return {'error': 'NodeMCU not connected'}
        
        if self.connection_type == 'wifi':
            # TODO: Implement WiFi reading
            return {'error': 'WiFi reading not yet implemented'}
        
        try:
            # Send request command
            self.serial_conn.write(b'READ_SENSORS\n')
            
            # Read response
            response = self.serial_conn.readline().decode().strip()
            
            if response:
                # Parse JSON response from NodeMCU
                try:
                    data = json.loads(response)
                    data['timestamp'] = datetime.now().isoformat()
                    data['source'] = 'nodemcu'
                    return data
                except json.JSONDecodeError:
                    # Handle simple comma-separated values
                    values = response.split(',')
                    return {
                        'ultrasonic_distance': float(values[0]) if len(values) > 0 else -1,
                        'gas_concentration': float(values[1]) if len(values) > 1 else -1,
                        'temperature': float(values[2]) if len(values) > 2 else -1,
                        'humidity': float(values[3]) if len(values) > 3 else -1,
                        'wifi_signal': int(values[4]) if len(values) > 4 else -1,
                        'timestamp': datetime.now().isoformat(),
                        'source': 'nodemcu'
                    }
            else:
                return {'error': 'No response from NodeMCU'}
                
        except Exception as e:
            logger.error(f"Error reading from NodeMCU: {e}")
            return {'error': str(e)}
    
    def send_command(self, command: str) -> str:
        """Send command to NodeMCU and get response"""
        if not self.connected:
            return "NodeMCU not connected"
        
        try:
            self.serial_conn.write(f"{command}\n".encode())
            response = self.serial_conn.readline().decode().strip()
            return response
        except Exception as e:
            logger.error(f"Error sending command to NodeMCU: {e}")
            return f"Error: {e}"

class SensorDataCollector:
    """
    Main class for collecting data from multiple sensors
    Handles real-time data collection, processing, and storage
    """
    
    def __init__(self, database_path: str = 'sensor_data.db'):
        """
        Initialize sensor data collector
        
        Args:
            database_path: Path to SQLite database
        """
        self.db = SensorDatabase(database_path)
        self.sensors: Dict[str, SensorInterface] = {}
        self.arduino = None
        self.collecting = False
        self.collection_thread = None
        self.collection_interval = 1.0  # seconds
        self.data_callbacks: List[Callable] = []
    
    def add_sensor(self, sensor: SensorInterface):
        """Add a sensor to the collection system"""
        self.sensors[sensor.name] = sensor
        logger.info(f"Added sensor: {sensor.name}")
    
    def add_nodemcu(self, port: str = '/dev/ttyUSB0', baudrate: int = 115200, wifi_ip: str = None):
        """Add NodeMCU interface"""
        self.nodemcu = NodeMCUInterface(port, baudrate, wifi_ip=wifi_ip)
        if self.nodemcu.connected:
            logger.info("NodeMCU interface added successfully")
        else:
            logger.warning("NodeMCU interface added but not connected")
    
    # Keep Arduino method for backwards compatibility
    def add_arduino(self, port: str = '/dev/ttyUSB0', baudrate: int = 115200):
        """Add Arduino/NodeMCU interface (alias for add_nodemcu)"""
        self.add_nodemcu(port, baudrate)
    
    def add_ultrasonic_sensor(self, trigger_pin: int = 18, echo_pin: int = 24, 
                            name: str = "ultrasonic_1"):
        """Convenience method to add ultrasonic sensor"""
        sensor = UltrasonicSensor(trigger_pin, echo_pin, name=name)
        self.add_sensor(sensor)
        return sensor
    
    def add_gas_sensor(self, analog_pin: int = 0, sensor_type: str = "MQ-2",
                      gas_type: str = "general", name: str = "gas_1"):
        """Convenience method to add gas sensor"""
        sensor = GasSensor(analog_pin, sensor_type, gas_type, name)
        self.add_sensor(sensor)
        return sensor
    
    def add_data_callback(self, callback: Callable):
        """Add callback function to be called when new data is collected"""
        self.data_callbacks.append(callback)
    
    def collect_single_reading(self) -> Dict:
        """Collect a single reading from all sensors"""
        readings = {}
        timestamp = datetime.now().isoformat()
        
        # Collect from individual sensors
        for sensor_name, sensor in self.sensors.items():
            try:
                data = sensor.read_data()
                readings[sensor_name] = data
            except Exception as e:
                logger.error(f"Error reading sensor {sensor_name}: {e}")
                readings[sensor_name] = {
                    'sensor_name': sensor_name,
                    'error': str(e),
                    'timestamp': timestamp
                }
        
        # Collect from NodeMCU if available
        if hasattr(self, 'nodemcu') and self.nodemcu and self.nodemcu.connected:
            try:
                nodemcu_data = self.nodemcu.read_sensors()
                if 'error' not in nodemcu_data:
                    readings['nodemcu'] = nodemcu_data
            except Exception as e:
                logger.error(f"Error reading NodeMCU: {e}")
        # Backwards compatibility with Arduino
        elif hasattr(self, 'arduino') and self.arduino and self.arduino.connected:
            try:
                arduino_data = self.arduino.read_sensors()
                if 'error' not in arduino_data:
                    readings['arduino'] = arduino_data
            except Exception as e:
                logger.error(f"Error reading Arduino: {e}")
        
        return readings
    
    def start_continuous_collection(self, interval: float = 1.0):
        """Start continuous data collection in background thread"""
        if self.collecting:
            logger.warning("Collection already running")
            return
        
        self.collection_interval = interval
        self.collecting = True
        self.collection_thread = threading.Thread(target=self._collection_loop)
        self.collection_thread.daemon = True
        self.collection_thread.start()
        logger.info(f"Started continuous collection (interval: {interval}s)")
    
    def stop_continuous_collection(self):
        """Stop continuous data collection"""
        if not self.collecting:
            return
        
        self.collecting = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
        logger.info("Stopped continuous collection")
    
    def _collection_loop(self):
        """Main collection loop running in background thread"""
        while self.collecting:
            try:
                readings = self.collect_single_reading()
                
                # Store in database
                self._store_readings(readings)
                
                # Call callbacks
                for callback in self.data_callbacks:
                    try:
                        callback(readings)
                    except Exception as e:
                        logger.error(f"Error in data callback: {e}")
                
                time.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Error in collection loop: {e}")
                time.sleep(self.collection_interval)
    
    def _store_readings(self, readings: Dict):
        """Store readings in database"""
        for sensor_name, data in readings.items():
            if 'error' in data:
                continue
            
            try:
                # Extract common fields
                timestamp = data.get('timestamp', datetime.now().isoformat())
                sensor_type = data.get('sensor_type', 'unknown')
                
                # Store based on sensor type
                if sensor_type == 'ultrasonic':
                    self.db.insert_sensor_reading(
                        timestamp=timestamp,
                        sensor_name=sensor_name,
                        original_reading=data.get('distance_cm', 0),
                        corrected_reading=data.get('distance_cm', 0),  # No correction yet
                        drift_estimate=0.0,
                        drift_magnitude=0.0
                    )
                elif sensor_type == 'gas':
                    self.db.insert_sensor_reading(
                        timestamp=timestamp,
                        sensor_name=sensor_name,
                        original_reading=data.get('concentration_ppm', 0),
                        corrected_reading=data.get('concentration_ppm', 0),  # No correction yet
                        drift_estimate=0.0,
                        drift_magnitude=0.0
                    )
                elif sensor_name == 'arduino':
                    # Handle Arduino multi-sensor data
                    for key, value in data.items():
                        if key not in ['timestamp', 'source'] and isinstance(value, (int, float)):
                            self.db.insert_sensor_reading(
                                timestamp=timestamp,
                                sensor_name=f"arduino_{key}",
                                original_reading=value,
                                corrected_reading=value,
                                drift_estimate=0.0,
                                drift_magnitude=0.0
                            )
            except Exception as e:
                logger.error(f"Error storing reading for {sensor_name}: {e}")
    
    def get_recent_readings(self, sensor_name: str = None, limit: int = 100) -> pd.DataFrame:
        """Get recent readings from database"""
        return self.db.get_recent_readings(sensor_name=sensor_name, limit=limit)
    
    def cleanup(self):
        """Cleanup resources"""
        self.stop_continuous_collection()
        
        # No GPIO cleanup needed for NodeMCU interface
        pass
        
        # Close NodeMCU connection
        if hasattr(self, 'nodemcu') and self.nodemcu and self.nodemcu.serial_conn:
            try:
                self.nodemcu.serial_conn.close()
            except:
                pass
        
        # Close Arduino connection (backwards compatibility)
        if hasattr(self, 'arduino') and self.arduino and self.arduino.serial_conn:
            try:
                self.arduino.serial_conn.close()
            except:
                pass

# NodeMCU (ESP8266/ESP32) sketch code for Arduino IDE
NODEMCU_SKETCH = """
/*
NodeMCU Sensor Interface Sketch
Supports ultrasonic (HC-SR04) and gas sensors (MQ series)
Compatible with ESP8266 and ESP32

NodeMCU Pin Mapping:
D0 = GPIO16    D1 = GPIO5     D2 = GPIO4     D3 = GPIO0
D4 = GPIO2     D5 = GPIO14    D6 = GPIO12    D7 = GPIO13
D8 = GPIO15    A0 = Analog Input

Connections:
- Ultrasonic Trigger: D1 (GPIO5)
- Ultrasonic Echo: D2 (GPIO4)  
- Gas Sensor: A0
- Optional: DHT22 Temperature/Humidity: D4 (GPIO2)
*/

#define TRIGGER_PIN D1  // GPIO5
#define ECHO_PIN D2     // GPIO4
#define GAS_PIN A0      // Analog input
#define DHT_PIN D4      // GPIO2 for DHT22

// Uncomment if using DHT sensor
// #include "DHT.h"
// #define DHT_TYPE DHT22
// DHT dht(DHT_PIN, DHT_TYPE);

void setup() {
  Serial.begin(115200);  // Higher baud rate for NodeMCU
  pinMode(TRIGGER_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  
  // dht.begin();  // Uncomment if using DHT sensor
  
  Serial.println("NodeMCU Sensor Interface Ready");
  Serial.println("Commands: READ_SENSORS, STATUS, WIFI_STATUS");
}

void loop() {
  if (Serial.available()) {
    String command = Serial.readStringUntil('\\n');
    command.trim();
    
    if (command == "READ_SENSORS") {
      readAllSensors();
    } else if (command == "STATUS") {
      sendStatus();
    } else if (command == "WIFI_STATUS") {
      sendWiFiStatus();
    }
  }
  
  delay(100);
}

void readAllSensors() {
  float distance = readUltrasonic();
  float gasLevel = readGasSensor();
  float temperature = readTemperature();
  float humidity = readHumidity();
  int wifiSignal = WiFi.RSSI();
  
  // Send JSON response
  Serial.print("{");
  Serial.print("\\"ultrasonic_distance\\": "); Serial.print(distance);
  Serial.print(", \\"gas_concentration\\": "); Serial.print(gasLevel);
  Serial.print(", \\"temperature\\": "); Serial.print(temperature);
  Serial.print(", \\"humidity\\": "); Serial.print(humidity);
  Serial.print(", \\"wifi_signal\\": "); Serial.print(wifiSignal);
  Serial.print(", \\"free_heap\\": "); Serial.print(ESP.getFreeHeap());
  Serial.print(", \\"uptime\\": "); Serial.print(millis());
  Serial.println("}");
}

float readUltrasonic() {
  digitalWrite(TRIGGER_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIGGER_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIGGER_PIN, LOW);
  
  long duration = pulseIn(ECHO_PIN, HIGH, 30000);
  if (duration == 0) return -1;
  
  float distance = (duration * 0.034) / 2;
  return distance;
}

float readGasSensor() {
  int rawValue = analogRead(GAS_PIN);
  float voltage = rawValue * (3.3 / 1024.0);  // NodeMCU uses 3.3V
  float concentration = voltage * 100; // Simple conversion, calibrate as needed
  return concentration;
}

float readTemperature() {
  // Using DHT sensor - uncomment if available
  // return dht.readTemperature();
  
  // Placeholder - replace with actual sensor reading
  return 25.0 + random(-5, 5); // Simulated temperature
}

float readHumidity() {
  // Using DHT sensor - uncomment if available  
  // return dht.readHumidity();
  
  // Placeholder - replace with actual sensor reading
  return 50.0 + random(-10, 10); // Simulated humidity
}

void sendStatus() {
  Serial.print("{");
  Serial.print("\\"device\\": \\"NodeMCU\\", ");
  Serial.print("\\"chip_id\\": "); Serial.print(ESP.getChipId());
  Serial.print(", \\"free_heap\\": "); Serial.print(ESP.getFreeHeap());
  Serial.print(", \\"uptime\\": "); Serial.print(millis());
  Serial.println("}");
}

void sendWiFiStatus() {
  Serial.print("{");
  Serial.print("\\"wifi_connected\\": "); Serial.print(WiFi.status() == WL_CONNECTED);
  Serial.print(", \\"ssid\\": \\""); Serial.print(WiFi.SSID());
  Serial.print("\\", \\"ip\\": \\""); Serial.print(WiFi.localIP());
  Serial.print("\\", \\"rssi\\": "); Serial.print(WiFi.RSSI());
  Serial.println("}");
}

/*
Optional: WiFi setup for wireless communication
Add this to setup() if you want WiFi connectivity:

#include <ESP8266WiFi.h>  // For ESP8266
// #include <WiFi.h>      // For ESP32

const char* ssid = "YourWiFiNetwork";
const char* password = "YourWiFiPassword";

void setupWiFi() {
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("WiFi connected");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());
}
*/
"""

if __name__ == "__main__":
    # Example usage with NodeMCU
    collector = SensorDataCollector()
    
    # Add sensors (configured for NodeMCU pins)
    ultrasonic = collector.add_ultrasonic_sensor(
        trigger_pin=5,  # D1 on NodeMCU
        echo_pin=4,     # D2 on NodeMCU
        name="distance_sensor_1"
    )
    gas = collector.add_gas_sensor(
        analog_pin="A0", 
        sensor_type="MQ-2", 
        gas_type="LPG", 
        name="lpg_sensor_1"
    )
    
    # Add NodeMCU interface
    collector.add_nodemcu('/dev/ttyUSB0', baudrate=115200)
    
    # Add data callback for real-time processing
    def on_new_data(readings):
        print(f"📊 New readings from {len(readings)} sensors")
        for sensor_name, data in readings.items():
            if 'distance_cm' in data:
                print(f"  📏 {sensor_name}: {data['distance_cm']} cm")
            elif 'concentration_ppm' in data:
                print(f"  🌬️ {sensor_name}: {data['concentration_ppm']} ppm")
            elif sensor_name == 'nodemcu':
                print(f"  🔧 NodeMCU: {data}")
    
    collector.add_data_callback(on_new_data)
    
    try:
        print("🔧 NodeMCU Sensor Interface Demo")
        print("=" * 40)
        
        # Collect single reading
        print("\n📈 Single reading:")
        readings = collector.collect_single_reading()
        for sensor, data in readings.items():
            status = data.get('status', 'unknown')
            print(f"  {sensor}: Status={status}")
        
        # Start continuous collection
        print("\n▶️ Starting continuous collection (press Ctrl+C to stop)...")
        collector.start_continuous_collection(interval=2.0)
        
        # Keep running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n⏹️ Stopping collection...")
    finally:
        collector.cleanup()
        print("✅ Cleanup complete")