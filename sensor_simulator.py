#!/usr/bin/env python3
"""
Sensor Data Simulator
Generates realistic sensor data without requiring physical hardware
Perfect for testing and development
"""

import time
import random
import math
import json
import threading
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

@dataclass
class SensorReading:
    """Structure for sensor readings"""
    sensor_type: str
    sensor_id: str
    value: float
    unit: str
    timestamp: datetime
    status: str = "ok"
    metadata: Dict = None

class UltrasonicSimulator:
    """Simulates HC-SR04 ultrasonic distance sensor"""
    
    def __init__(self, sensor_id: str = "ultrasonic_1", min_distance: float = 2.0, max_distance: float = 400.0):
        self.sensor_id = sensor_id
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.base_distance = 50.0  # Base distance in cm
        self.noise_level = 2.0     # Noise amplitude
        self.trend_speed = 0.1     # Speed of distance changes
        self.time_offset = random.uniform(0, 2 * math.pi)
        
    def read(self) -> SensorReading:
        """Generate realistic distance reading"""
        current_time = time.time()
        
        # Simulate realistic distance changes with trends and noise
        trend = math.sin(current_time * self.trend_speed + self.time_offset) * 20
        noise = random.gauss(0, self.noise_level)
        daily_pattern = math.sin(current_time / 3600 * 2 * math.pi) * 10  # 24-hour cycle
        
        distance = self.base_distance + trend + noise + daily_pattern
        distance = max(self.min_distance, min(self.max_distance, distance))
        
        # Simulate occasional sensor issues
        status = "ok"
        if random.random() < 0.05:  # 5% chance of issues
            if random.random() < 0.5:
                status = "warning"
                distance += random.uniform(-10, 10)
            else:
                status = "error"
                distance = -1
        
        return SensorReading(
            sensor_type="ultrasonic",
            sensor_id=self.sensor_id,
            value=round(distance, 2),
            unit="cm",
            timestamp=datetime.now(),
            status=status,
            metadata={
                "pin_trigger": "D1",
                "pin_echo": "D2",
                "max_range": self.max_distance,
                "temperature_compensation": True
            }
        )

class GasSensorSimulator:
    """Simulates MQ-series gas sensors"""
    
    def __init__(self, sensor_id: str = "gas_1", gas_type: str = "LPG", 
                 baseline_ppm: float = 50.0, sensor_type: str = "MQ-2"):
        self.sensor_id = sensor_id
        self.gas_type = gas_type
        self.sensor_type = sensor_type
        self.baseline_ppm = baseline_ppm
        self.noise_level = 5.0
        self.drift_rate = 0.001  # Gradual sensor drift
        self.start_time = time.time()
        self.warmup_time = 300   # 5 minutes warmup
        
    def read(self) -> SensorReading:
        """Generate realistic gas concentration reading"""
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        # Simulate sensor warmup period
        if elapsed < self.warmup_time:
            warmup_factor = elapsed / self.warmup_time
            status = "warming_up"
        else:
            warmup_factor = 1.0
            status = "ok"
        
        # Base concentration with environmental variations
        base_concentration = self.baseline_ppm * warmup_factor
        
        # Add realistic variations
        environmental_variation = math.sin(current_time / 1800) * 10  # 30-minute cycle
        noise = random.gauss(0, self.noise_level)
        drift = elapsed * self.drift_rate
        
        # Simulate gas events (spikes)
        if random.random() < 0.02:  # 2% chance of gas event
            spike = random.uniform(100, 500)
            base_concentration += spike
            
        concentration = base_concentration + environmental_variation + noise + drift
        concentration = max(0, concentration)
        
        # Convert to raw sensor values for realism
        raw_value = int(concentration * 10.24)  # Simulate ADC reading
        voltage = (raw_value / 1024.0) * 3.3
        
        # Detect dangerous levels
        if concentration > 1000:
            status = "danger"
        elif concentration > 500:
            status = "warning"
        
        return SensorReading(
            sensor_type="gas",
            sensor_id=self.sensor_id,
            value=round(concentration, 1),
            unit="ppm",
            timestamp=datetime.now(),
            status=status,
            metadata={
                "gas_type": self.gas_type,
                "sensor_model": self.sensor_type,
                "raw_value": raw_value,
                "voltage": round(voltage, 3),
                "pin": "A0",
                "warmup_time": self.warmup_time
            }
        )

class TemperatureSimulator:
    """Simulates temperature sensor (bonus sensor)"""
    
    def __init__(self, sensor_id: str = "temp_1"):
        self.sensor_id = sensor_id
        self.base_temp = 25.0  # Base temperature in Celsius
        
    def read(self) -> SensorReading:
        current_time = time.time()
        
        # Daily temperature cycle
        daily_cycle = math.sin((current_time / 86400) * 2 * math.pi - math.pi/2) * 8
        noise = random.gauss(0, 0.5)
        
        temperature = self.base_temp + daily_cycle + noise
        
        status = "ok"
        if temperature > 40:
            status = "warning"
        elif temperature > 50:
            status = "danger"
            
        return SensorReading(
            sensor_type="temperature",
            sensor_id=self.sensor_id,
            value=round(temperature, 1),
            unit="°C",
            timestamp=datetime.now(),
            status=status,
            metadata={"sensor_model": "DHT22", "pin": "D3"}
        )

class HumiditySimulator:
    """Simulates humidity sensor (bonus sensor)"""
    
    def __init__(self, sensor_id: str = "humidity_1"):
        self.sensor_id = sensor_id
        self.base_humidity = 45.0
        
    def read(self) -> SensorReading:
        current_time = time.time()
        
        # Humidity variations
        cycle = math.sin((current_time / 7200) * 2 * math.pi) * 15  # 2-hour cycle
        noise = random.gauss(0, 2.0)
        
        humidity = self.base_humidity + cycle + noise
        humidity = max(10, min(95, humidity))  # Realistic range
        
        status = "ok"
        if humidity > 80:
            status = "warning"
            
        return SensorReading(
            sensor_type="humidity",
            sensor_id=self.sensor_id,
            value=round(humidity, 1),
            unit="%RH",
            timestamp=datetime.now(),
            status=status,
            metadata={"sensor_model": "DHT22", "pin": "D3"}
        )

class SensorNetworkSimulator:
    """Manages multiple simulated sensors"""
    
    def __init__(self):
        self.sensors = {}
        self.running = False
        self.data_callbacks = []
        self.collection_thread = None
        self.interval = 2.0
        
        # Initialize default sensors
        self.add_sensor("distance_1", UltrasonicSimulator("distance_1", min_distance=5, max_distance=200))
        self.add_sensor("lpg_sensor", GasSensorSimulator("lpg_sensor", "LPG", baseline_ppm=30))
        self.add_sensor("co_sensor", GasSensorSimulator("co_sensor", "CO", baseline_ppm=15, sensor_type="MQ-7"))
        self.add_sensor("temperature", TemperatureSimulator("temperature"))
        self.add_sensor("humidity", HumiditySimulator("humidity"))
        
    def add_sensor(self, name: str, simulator):
        """Add a sensor to the network"""
        self.sensors[name] = simulator
        
    def remove_sensor(self, name: str):
        """Remove a sensor from the network"""
        if name in self.sensors:
            del self.sensors[name]
            
    def read_all_sensors(self) -> Dict[str, SensorReading]:
        """Read all sensors once"""
        readings = {}
        for name, sensor in self.sensors.items():
            try:
                reading = sensor.read()
                readings[name] = reading
            except Exception as e:
                # Create error reading
                readings[name] = SensorReading(
                    sensor_type="error",
                    sensor_id=name,
                    value=-1,
                    unit="error",
                    timestamp=datetime.now(),
                    status="error",
                    metadata={"error": str(e)}
                )
        return readings
    
    def add_data_callback(self, callback):
        """Add callback for new data"""
        self.data_callbacks.append(callback)
        
    def _collection_loop(self):
        """Background data collection loop"""
        while self.running:
            readings = self.read_all_sensors()
            
            # Notify callbacks
            for callback in self.data_callbacks:
                try:
                    callback(readings)
                except Exception as e:
                    print(f"Callback error: {e}")
                    
            time.sleep(self.interval)
    
    def start_continuous_collection(self, interval: float = 2.0):
        """Start continuous sensor reading"""
        self.interval = interval
        self.running = True
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        
    def stop_continuous_collection(self):
        """Stop continuous sensor reading"""
        self.running = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5.0)
            
    def get_sensor_status(self) -> Dict[str, str]:
        """Get status of all sensors"""
        status = {}
        for name, sensor in self.sensors.items():
            reading = sensor.read()
            status[name] = {
                "type": reading.sensor_type,
                "status": reading.status,
                "last_reading": reading.value,
                "unit": reading.unit
            }
        return status

def format_reading_for_display(reading: SensorReading) -> str:
    """Format sensor reading for console display"""
    status_emoji = {
        "ok": "🟢",
        "warning": "🟡", 
        "danger": "🔴",
        "error": "❌",
        "warming_up": "⏳"
    }
    
    emoji = status_emoji.get(reading.status, "❓")
    timestamp = reading.timestamp.strftime("%H:%M:%S")
    
    return f"{emoji} [{timestamp}] {reading.sensor_id}: {reading.value} {reading.unit} ({reading.status})"

def main():
    """Demo of the sensor simulator"""
    print("🎭 Sensor Network Simulator")
    print("=" * 50)
    
    simulator = SensorNetworkSimulator()
    
    # Show sensor status
    print("\n📡 Available Sensors:")
    status = simulator.get_sensor_status()
    for name, info in status.items():
        print(f"  • {name}: {info['type']} ({info['status']})")
    
    print("\n📊 Single Reading Test:")
    readings = simulator.read_all_sensors()
    for name, reading in readings.items():
        print(f"  {format_reading_for_display(reading)}")
    
    print(f"\n▶️ Starting continuous simulation (press Ctrl+C to stop)...")
    
    # Add display callback
    def display_readings(readings):
        print(f"\n📈 Live Data - {datetime.now().strftime('%H:%M:%S')}")
        for name, reading in readings.items():
            print(f"  {format_reading_for_display(reading)}")
    
    simulator.add_data_callback(display_readings)
    
    try:
        simulator.start_continuous_collection(interval=3.0)
        
        # Keep running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n⏹️ Stopping simulation...")
        simulator.stop_continuous_collection()
        print("✅ Simulation stopped")

if __name__ == "__main__":
    main()