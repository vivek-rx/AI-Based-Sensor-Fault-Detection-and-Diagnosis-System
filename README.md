# Industrial Sensor Monitoring System

A streamlined real-time sensor monitoring system with hardware integration and simulation capabilities.

## 🎯 Project Overview

This system provides:
- **Real-time sensor monitoring** with live dashboard
- **Hardware integration** with NodeMCU and sensors
- **Realistic sensor simulation** for development and testing  
- **Industrial-grade database** with predictive maintenance
- **Interactive web dashboard** built with Streamlit

## 📂 Project Structure

```
Major__project/
├── app.py                   # Main Streamlit dashboard
├── database.py              # SQLite database management
├── hardware_interface.py    # Hardware integration system
├── sensor_simulator.py      # Realistic sensor simulation
├── requirements.txt         # Python dependencies
├── combined_sensor_data.csv # Sample sensor data
├── sensor_data.db          # SQLite database (auto-generated)
└── README.md               # This documentation
```

## 🚀 Quick Start

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Launch dashboard
streamlit run app.py
```

### Usage
1. **Open browser**: Navigate to `http://localhost:8501`
2. **Explore dashboard**: Multiple pages for monitoring and analysis
3. **Hardware Monitor**: View live sensor data (simulated or real)
4. **Data Export**: Export sensor readings in various formats

## 🎛️ Dashboard Features

### Core Pages
- **Overview**: System status and key metrics
- **Sensor Readings**: Historical data analysis and visualization
- **Hardware Monitor**: Live sensor data with real-time updates
- **Sensor Comparison**: Cross-sensor analysis and correlation
- **Performance Metrics**: System health and efficiency tracking
- **Data Export**: Export capabilities for analysis and reporting

### Key Features
- ✅ **Real-time visualization** with Plotly charts
- ✅ **Sensor simulation** for testing without hardware
- ✅ **Hardware integration** ready for NodeMCU and sensors
- ✅ **Predictive maintenance** with AI-powered alerts
- ✅ **Cross-sensor analytics** for anomaly detection
- ✅ **Data export** in multiple formats (CSV, JSON, Excel)

## 🔧 Hardware Integration

### Supported Hardware
- **NodeMCU ESP8266/ESP32** for wireless communication
- **HC-SR04 Ultrasonic Sensor** for distance measurement
- **MQ-2/MQ-7 Gas Sensors** for air quality monitoring

### Connection (when using real hardware)
```
NodeMCU Pin  →  Sensor Connection
D1 (GPIO5)   →  HC-SR04 Trigger Pin
D2 (GPIO4)   →  HC-SR04 Echo Pin
A0 (ADC)     →  Gas Sensor Analog Output
3.3V         →  Sensor VCC
GND          →  Sensor GND
```

### Simulation Mode
The system automatically runs in simulation mode when hardware isn't connected, providing:
- Realistic sensor data patterns
- Environmental variations and noise
- Occasional sensor errors for testing
- Multiple sensor types (distance, gas, temperature, humidity)

## 🏗️ Architecture

### Core Components

**1. Database (`database.py`)**
- SQLite database with industrial-grade schema
- Stores sensor readings, alerts, and performance metrics
- Supports predictive maintenance and cost analysis
- Handles sensor health monitoring and environmental factors

**2. Hardware Interface (`hardware_interface.py`)**  
- Abstracted sensor interface for multiple hardware types
- NodeMCU communication via serial protocol
- Automatic fallback to simulation mode
- Support for ultrasonic and gas sensors

**3. Sensor Simulator (`sensor_simulator.py`)**
- Realistic sensor behavior simulation
- Multiple sensor types with proper physics models
- Environmental variations and realistic noise
- Configurable parameters for different scenarios

**4. Dashboard (`app.py`)**
- Modern Streamlit interface with multiple pages
- Real-time data visualization and monitoring
- Interactive controls and configuration options
- Export capabilities and performance analytics

## 📊 Database Schema

The system uses an SQLite database with these key tables:
- `sensor_data` - Raw and processed sensor readings
- `sensor_health` - Health monitoring and diagnostics
- `maintenance_logs` - Maintenance history and scheduling
- `alerts` - System notifications and warnings
- `cross_sensor_anomalies` - Multi-sensor correlation analysis
- `cost_analysis` - ROI tracking and operational costs
- `performance_metrics` - System efficiency metrics
- `environmental_factors` - Environmental correlation data

## 🎮 Testing & Development

### Simulation Mode
```bash
# The system automatically uses simulation when no hardware is connected
streamlit run app.py
# Navigate to "Hardware Monitor" page for live simulated data
```

### Development Features
- **No hardware required** - full simulation capabilities
- **Realistic data patterns** - proper sensor physics and noise
- **Multiple sensor types** - distance, gas, temperature, humidity
- **Environmental variations** - daily cycles and trends
- **Error simulation** - occasional sensor faults for robust testing

## 🏭 Industrial Applications

- **Manufacturing**: Production line monitoring and quality control
- **Environmental**: Air quality and emissions tracking
- **Safety**: Gas leak detection and proximity sensing  
- **Maintenance**: Predictive maintenance and asset health
- **Research**: Sensor network development and testing

## ⚙️ Configuration

### Settings
The dashboard includes configurable settings for:
- Alert thresholds and notification preferences
- Auto-refresh intervals for real-time monitoring
- Hardware connection parameters
- Data export formats and scheduling

### Customization
- Add new sensor types by extending the hardware interface
- Modify simulation parameters for different scenarios
- Configure database schema for specific requirements
- Customize dashboard pages for specific use cases

## 🔧 Troubleshooting

### Common Issues
- **Import errors**: Run `pip install -r requirements.txt`
- **Database errors**: Delete `sensor_data.db` to reset
- **Hardware connection**: System automatically falls back to simulation
- **Port issues**: Check serial port permissions and connections

### Debug Mode
Enable detailed logging by setting debug flags in the hardware interface.

## 📄 License

This project is for educational and research purposes.

---

**🚀 Ready for industrial IoT monitoring with full simulation capabilities!**



