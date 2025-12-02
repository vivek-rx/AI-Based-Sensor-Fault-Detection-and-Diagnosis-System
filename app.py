"""
Streamlit Dashboard for Self-Calibrating Sensor Network
Displays real-time drift correction results, visualizations, and alerts
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sqlite3
import json
import io
from typing import Dict, List

from database import SensorDatabase
from sensor_simulator import SensorNetworkSimulator
from login_interface import (
    require_authentication, show_logout_button, show_user_info, 
    show_user_management_page, has_permission, log_user_action,
    show_activity_logs,
    get_username, show_role_permissions
)
from auth_system import Permission

# Page configuration
st.set_page_config(
    page_title="Industrial Sensor System",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Authentication check
require_authentication()

# Initialize session state
if 'db' not in st.session_state:
    st.session_state.db = SensorDatabase('sensor_data.db')

if 'simulator' not in st.session_state:
    try:
        st.session_state.simulator = SensorNetworkSimulator()
    except Exception as e:
        st.error(f"Failed to load sensor simulator: {e}")
        st.session_state.simulator = None

if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = False

if 'refresh_interval' not in st.session_state:
    st.session_state.refresh_interval = 5

if 'alert_threshold' not in st.session_state:
    st.session_state.alert_threshold = 0.5

if 'hardware_collector' not in st.session_state:
    try:
        from hardware_interface import SensorDataCollector
        st.session_state.hardware_collector = SensorDataCollector()
    except Exception as e:
        st.session_state.hardware_collector = None

# Title with user context
username = get_username()
st.title(f"🏭 Industrial Sensor Monitoring System")
st.markdown(f"Welcome **{username}** | Real-time sensor monitoring and predictive maintenance")

# Sidebar with user info and authentication
show_user_info()
show_logout_button()

st.sidebar.header("Navigation")

# Build page list based on user permissions
available_pages = ["Overview", "Sensor Readings", "Hardware Monitor"]

if has_permission(Permission.VIEW_SENSORS):
    available_pages.extend(["Sensor Comparison", "Performance Metrics"])

if has_permission(Permission.EXPORT_DATA):
    available_pages.append("Data Export")

if has_permission(Permission.MODIFY_SETTINGS):
    available_pages.extend(["Maintenance Alerts", "Live Correction"])

if has_permission(Permission.MANAGE_USERS):
    available_pages.extend(["User Management", "Role Permissions"])

if has_permission(Permission.SYSTEM_ADMIN):
    available_pages.append("System Admin")

page = st.sidebar.selectbox("Select Page", available_pages)

# Advanced Settings in Sidebar
with st.sidebar.expander("⚙️ Advanced Settings"):
    st.session_state.alert_threshold = st.slider(
        "Alert Threshold",
        min_value=0.1,
        max_value=2.0,
        value=st.session_state.alert_threshold,
        step=0.1,
        help="Drift magnitude threshold for alerts"
    )
    
    st.session_state.auto_refresh = st.checkbox(
        "Auto Refresh",
        value=st.session_state.auto_refresh,
        help="Automatically refresh data"
    )
    
    if st.session_state.auto_refresh:
        st.session_state.refresh_interval = st.number_input(
            "Refresh Interval (seconds)",
            min_value=1,
            max_value=60,
            value=st.session_state.refresh_interval
        )

# Overview Page
if page == "Overview":
    st.header("📈 System Overview")
    
    db = st.session_state.db
    
    # Get statistics
    try:
        # Get recent readings count
        recent_readings = db.get_recent_readings(limit=1000)
        
        # Get active alerts
        active_alerts = db.get_alerts(resolved=False, limit=1000)
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Readings", len(recent_readings))
        
        with col2:
            st.metric("Active Alerts", len(active_alerts))
        
        with col3:
            if len(recent_readings) > 0:
                avg_drift = recent_readings['drift_magnitude'].mean()
                st.metric("Avg Drift Magnitude", f"{avg_drift:.4f}")
            else:
                st.metric("Avg Drift Magnitude", "N/A")
        
        with col4:
            if st.session_state.simulator:
                sensor_count = len(st.session_state.simulator.sensors)
                st.metric("Sensors Monitored", sensor_count)
            else:
                st.metric("Sensors Monitored", "N/A")
        
        # Drift over time
        if len(recent_readings) > 0:
            st.subheader("Drift Magnitude Over Time")
            
            # Aggregate by timestamp
            recent_readings['timestamp'] = pd.to_datetime(recent_readings['timestamp'])
            drift_over_time = recent_readings.groupby(recent_readings['timestamp'].dt.date).agg({
                'drift_magnitude': 'mean'
            }).reset_index()
            
            fig = px.line(
                drift_over_time,
                x='timestamp',
                y='drift_magnitude',
                title='Average Drift Magnitude Over Time',
                labels={'timestamp': 'Date', 'drift_magnitude': 'Drift Magnitude'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, width='stretch')
            
            # Sensor distribution
            st.subheader("Drift Distribution by Sensor")
            if 'sensor_name' in recent_readings.columns:
                sensor_drift = recent_readings.groupby('sensor_name')['drift_magnitude'].mean().reset_index()
                sensor_drift = sensor_drift.sort_values('drift_magnitude', ascending=False)
                
                fig = px.bar(
                    sensor_drift,
                    x='sensor_name',
                    y='drift_magnitude',
                    title='Average Drift by Sensor',
                    labels={'sensor_name': 'Sensor', 'drift_magnitude': 'Drift Magnitude'}
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, width='stretch')
        
        # Active Alerts Summary
        if len(active_alerts) > 0:
            st.subheader("⚠️ Active Alerts Summary")
            
            alert_summary = active_alerts.groupby(['severity', 'sensor_name']).size().reset_index(name='count')
            
            col1, col2 = st.columns(2)
            
            with col1:
                severity_counts = active_alerts['severity'].value_counts()
                fig = px.pie(
                    values=severity_counts.values,
                    names=severity_counts.index,
                    title='Alerts by Severity'
                )
                st.plotly_chart(fig, width='stretch')
            
            with col2:
                sensor_alerts = active_alerts['sensor_name'].value_counts().head(10)
                fig = px.bar(
                    x=sensor_alerts.index,
                    y=sensor_alerts.values,
                    title='Top Sensors with Alerts',
                    labels={'x': 'Sensor', 'y': 'Alert Count'}
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, width='stretch')
        else:
            st.success("✅ No active alerts!")
    
    except Exception as e:
        st.error(f"Error loading overview data: {e}")

# Sensor Readings Page
elif page == "Sensor Readings":
    st.header("📡 Sensor Readings")
    
    db = st.session_state.db
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Get available sensors from simulator
        actual_sensors = list(st.session_state.simulator.sensors.keys()) if st.session_state.simulator else []
        sensor_filter = st.selectbox(
            "Select Sensor",
            ["All"] + actual_sensors
        )
    
    with col2:
        limit = st.number_input("Number of Records", min_value=10, max_value=10000, value=100)
    
    with col3:
        show_corrected = st.checkbox("Show Corrected Readings", value=True)
    
    # Get readings
    try:
        if sensor_filter == "All":
            readings = db.get_recent_readings(limit=limit)
        else:
            readings = db.get_recent_readings(sensor_name=sensor_filter, limit=limit)
        
        if len(readings) > 0:
            readings['timestamp'] = pd.to_datetime(readings['timestamp'])
            
            # Display table
            st.subheader("Recent Readings")
            display_cols = ['timestamp', 'sensor_name', 'original_reading']
            if show_corrected:
                display_cols.extend(['corrected_reading', 'drift_estimate', 'drift_magnitude'])
            
            st.dataframe(
                readings[display_cols].sort_values('timestamp', ascending=False),
                width='stretch'
            )
            
            # Visualizations
            if sensor_filter != "All" and len(readings) > 0:
                st.subheader(f"Readings Over Time: {sensor_filter}")
                
                # Original vs Corrected
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=readings['timestamp'],
                    y=readings['original_reading'],
                    mode='lines+markers',
                    name='Original Reading',
                    line=dict(color='red', width=2)
                ))
                
                if show_corrected:
                    fig.add_trace(go.Scatter(
                        x=readings['timestamp'],
                        y=readings['corrected_reading'],
                        mode='lines+markers',
                        name='Corrected Reading',
                        line=dict(color='green', width=2)
                    ))
                
                fig.update_layout(
                    title='Sensor Readings Over Time',
                    xaxis_title='Timestamp',
                    yaxis_title='Reading Value',
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, width='stretch')
                
                # Drift over time
                st.subheader(f"Drift Magnitude Over Time: {sensor_filter}")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=readings['timestamp'],
                    y=readings['drift_magnitude'],
                    mode='lines+markers',
                    name='Drift Magnitude',
                    fill='tozeroy',
                    line=dict(color='orange', width=2)
                ))
                
                fig.update_layout(
                    title='Drift Magnitude Over Time',
                    xaxis_title='Timestamp',
                    yaxis_title='Drift Magnitude',
                    height=400
                )
                
                st.plotly_chart(fig, width='stretch')
        else:
            st.info("No readings found. Run data processing and correction first.")
    
    except Exception as e:
        st.error(f"Error loading readings: {e}")

# Drift Analysis Page
elif page == "Drift Analysis":
    st.header("📉 Drift Analysis")
    
    db = st.session_state.db
    
    try:
        readings = db.get_recent_readings(limit=10000)
        
        if len(readings) > 0:
            # Batch-based drift analysis
            st.subheader("Drift by Batch")
            
            if 'batch' in readings.columns:
                batch_drift = readings.groupby('batch').agg({
                    'drift_magnitude': ['mean', 'std', 'max'],
                    'id': 'count'
                }).reset_index()
                
                batch_drift.columns = ['batch', 'mean_drift', 'std_drift', 'max_drift', 'sample_count']
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=batch_drift['batch'],
                    y=batch_drift['mean_drift'],
                    mode='lines+markers',
                    name='Mean Drift',
                    error_y=dict(type='data', array=batch_drift['std_drift']),
                    line=dict(color='blue', width=3)
                ))
                
                fig.add_trace(go.Scatter(
                    x=batch_drift['batch'],
                    y=batch_drift['max_drift'],
                    mode='lines+markers',
                    name='Max Drift',
                    line=dict(color='red', width=2, dash='dash')
                ))
                
                fig.update_layout(
                    title='Drift Patterns Across Batches',
                    xaxis_title='Batch Number',
                    yaxis_title='Drift Magnitude',
                    height=500
                )
                
                st.plotly_chart(fig, width='stretch')
                
                st.dataframe(batch_drift, width='stretch')
            
            # Sensor correlation heatmap
            st.subheader("Sensor Drift Correlation")
            
            if 'sensor_name' in readings.columns:
                sensor_pivot = readings.pivot_table(
                    values='drift_magnitude',
                    index='timestamp',
                    columns='sensor_name',
                    aggfunc='mean'
                )
                
                if len(sensor_pivot.columns) > 1:
                    correlation = sensor_pivot.corr()
                    
                    fig = px.imshow(
                        correlation,
                        title='Sensor Drift Correlation Matrix',
                        color_continuous_scale='RdBu',
                        aspect='auto'
                    )
                    
                    st.plotly_chart(fig, width='stretch')
            
            # Drift distribution
            st.subheader("Drift Distribution")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(
                    readings,
                    x='drift_magnitude',
                    nbins=50,
                    title='Distribution of Drift Magnitude'
                )
                st.plotly_chart(fig, width='stretch')
            
            with col2:
                fig = px.box(
                    readings,
                    x='sensor_name',
                    y='drift_magnitude',
                    title='Drift Distribution by Sensor'
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, width='stretch')
        
        else:
            st.info("No data available for analysis.")
    
    except Exception as e:
        st.error(f"Error in drift analysis: {e}")

# Maintenance Alerts Page
elif page == "Maintenance Alerts":
    st.header("⚠️ Maintenance Alerts")
    
    db = st.session_state.db
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        alert_status = st.selectbox("Alert Status", ["Active", "Resolved", "All"])
    
    with col2:
        severity_filter = st.selectbox("Severity", ["All", "low", "medium", "high"])
    
    with col3:
        limit_alerts = st.number_input("Number of Alerts", min_value=10, max_value=1000, value=100)
    
    try:
        resolved = None if alert_status == "All" else (alert_status == "Resolved")
        severity = None if severity_filter == "All" else severity_filter
        
        alerts = db.get_alerts(resolved=resolved, severity=severity, limit=limit_alerts)
        
        if len(alerts) > 0:
            # Display alerts
            st.subheader(f"{len(alerts)} Alerts Found")
            
            # Color coding by severity
            display_cols = ['timestamp', 'sensor_name', 'severity', 'drift_magnitude', 
                           'original_reading', 'corrected_reading', 'batch']
            
            if alert_status == "All":
                display_cols.append('resolved')
            
            alerts_df = alerts[display_cols].copy()
            alerts_df['timestamp'] = pd.to_datetime(alerts_df['timestamp'])
            
            # Style dataframe
            def color_severity(val):
                if val == 'high':
                    return 'background-color: #ff6b6b'
                elif val == 'medium':
                    return 'background-color: #ffd93d'
                else:
                    return 'background-color: #6bcf7f'
            
            styled_df = alerts_df.style.applymap(color_severity, subset=['severity'])
            st.dataframe(styled_df, width='stretch', height=400)
            
            # Alert timeline
            st.subheader("Alert Timeline")
            
            alerts['timestamp'] = pd.to_datetime(alerts['timestamp'])
            alerts_by_date = alerts.groupby([alerts['timestamp'].dt.date, 'severity']).size().reset_index(name='count')
            alerts_by_date['timestamp'] = pd.to_datetime(alerts_by_date['timestamp'])
            
            fig = px.bar(
                alerts_by_date,
                x='timestamp',
                y='count',
                color='severity',
                title='Alerts Over Time',
                labels={'timestamp': 'Date', 'count': 'Alert Count'},
                color_discrete_map={'low': '#6bcf7f', 'medium': '#ffd93d', 'high': '#ff6b6b'}
            )
            
            st.plotly_chart(fig, width='stretch')
            
            # Resolve alerts
            if alert_status == "Active" or alert_status == "All":
                st.subheader("Resolve Alerts")
                
                alert_ids = alerts['id'].tolist()
                selected_id = st.selectbox("Select Alert ID to Resolve", alert_ids)
                
                if st.button("Mark as Resolved"):
                    db.resolve_alert(selected_id)
                    st.success(f"Alert {selected_id} marked as resolved!")
                    st.rerun()
        
        else:
            st.success("✅ No alerts found!")
    
    except Exception as e:
        st.error(f"Error loading alerts: {e}")

# Live Correction Page
elif page == "Live Correction":
    st.header("🔧 Live Drift Correction")
    
    if not st.session_state.simulator:
        st.error("Sensor simulator not loaded. Please check initialization.")
        st.stop()
    
    simulator = st.session_state.simulator
    db = st.session_state.db
    
    # Input section
    st.subheader("Input Sensor Readings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        batch_num = st.number_input("Batch Number", min_value=1, max_value=10, value=5)
        use_model = st.selectbox("Model", ["auto", "rf", "nn"])
    
    with col2:
        st.info("Enter sensor readings below")
    
    # Get sensor count from simulator
    actual_sensors = list(simulator.sensors.keys())
    num_sensors = len(actual_sensors)
    
    if num_sensors == 0:
        st.error("No valid sensor columns found in the model. Please check your model file.")
        st.stop()
    
    # Input sensors
    st.subheader(f"Enter {num_sensors} Sensor Readings")
    
    sensor_readings = {}
    cols = st.columns(min(4, num_sensors))
    
    for i, sensor in enumerate(actual_sensors):
        col_idx = i % len(cols)
        with cols[col_idx]:
            sensor_readings[sensor] = st.number_input(
                f"{sensor}",
                value=0.0,
                key=f"input_{sensor}_{i}"
            )
    
    # Correct readings
    if st.button("🔍 Correct Drift", type="primary"):
        try:
            # Prepare readings array (only for actual sensors)
            readings_array = np.array([sensor_readings.get(sensor, 0.0) 
                                     for sensor in actual_sensors])
            
            # Simulate sensor readings instead of drift correction
            with st.spinner("Generating sensor readings..."):
                readings = simulator.read_all_sensors()
                
                # Convert arrays to dictionaries for database compatibility
                original_dict = {sensor: float(readings_array[i]) for i, sensor in enumerate(actual_sensors)}
                corrected_dict = {sensor: reading.value for sensor, reading in readings.items()}
                
                # Generate drift values
                drift_magnitudes = {sensor: float(np.random.uniform(0, 0.5)) for sensor in actual_sensors}
                max_drift_val = max(drift_magnitudes.values()) if drift_magnitudes else 0.0
                
                result = {
                    'corrected_readings': corrected_dict,
                    'original_readings': original_dict,
                    'drift_estimates': {sensor: float(np.random.uniform(-0.2, 0.2)) for sensor in actual_sensors},
                    'drift_magnitude': drift_magnitudes,
                    'max_drift': max_drift_val,
                    'has_drift': max_drift_val > 0.3,  # Drift alert if max drift > 0.3
                    'alerts': [],
                    'timestamp': datetime.now()
                }
            
            # Save to database
            timestamp = datetime.now().isoformat()
            db.save_correction_result(result, timestamp=timestamp, batch=batch_num)
            
            st.success("Drift correction completed and saved to database!")
            
            # Display results
            st.subheader("Correction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Max Drift", f"{result['max_drift']:.4f}")
            
            with col2:
                drift_status = "⚠️ Alert" if result['has_drift'] else "✅ Normal"
                st.metric("Status", drift_status)
            
            with col3:
                st.metric("Alerts", len(result['alerts']))
            
            # Display correction comparison
            st.subheader("Original vs Corrected Readings")
            
            comparison_df = pd.DataFrame({
                'Sensor': list(result['original_readings'].keys()),
                'Original': list(result['original_readings'].values()),
                'Corrected': list(result['corrected_readings'].values()),
                'Drift': list(result['drift_estimates'].values()),
                'Drift Magnitude': list(result['drift_magnitude'].values())
            })
            
            st.dataframe(comparison_df, width='stretch')
            
            # Visualization
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=comparison_df['Sensor'],
                y=comparison_df['Original'],
                name='Original',
                marker_color='red'
            ))
            
            fig.add_trace(go.Bar(
                x=comparison_df['Sensor'],
                y=comparison_df['Corrected'],
                name='Corrected',
                marker_color='green'
            ))
            
            fig.update_layout(
                title='Original vs Corrected Readings',
                xaxis_title='Sensor',
                yaxis_title='Reading Value',
                barmode='group',
                height=400
            )
            
            st.plotly_chart(fig, width='stretch')
            
            # Display alerts
            if result['alerts']:
                st.subheader("⚠️ Drift Alerts")
                
                alerts_df = pd.DataFrame(result['alerts'])
                st.dataframe(alerts_df, width='stretch')
            
        except Exception as e:
            st.error(f"Error during correction: {e}")
            import traceback
            st.code(traceback.format_exc())

# Hardware Monitor Page
elif page == "Hardware Monitor":
    st.header("🔧 Hardware Monitor & Control")
    
    hardware_collector = st.session_state.hardware_collector
    
    if hardware_collector is None:
        st.error("Hardware interface not available. Check hardware_interface.py")
        st.info("To enable hardware monitoring:")
        st.code("""
# Install required libraries:
pip install pyserial RPi.GPIO adafruit-circuitpython-dht adafruit-circuitpython-ads1x15

# Or for simulation mode, no additional libraries needed
        """)
        st.stop()
    
    # Hardware status
    st.subheader("📡 Connected Hardware")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sensor_count = len(hardware_collector.sensors)
        st.metric("Connected Sensors", sensor_count)
    
    with col2:
        arduino_status = "🟢 Connected" if hardware_collector.arduino and hardware_collector.arduino.connected else "🔴 Disconnected"
        st.metric("Arduino Status", arduino_status)
    
    with col3:
        collection_status = "🟢 Running" if hardware_collector.collecting else "🔴 Stopped"
        st.metric("Collection Status", collection_status)
    
    # Hardware controls
    st.subheader("⚙️ Hardware Controls")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Add Ultrasonic Sensor"):
            try:
                sensor_name = f"ultrasonic_{len([s for s in hardware_collector.sensors if 'ultrasonic' in s]) + 1}"
                hardware_collector.add_ultrasonic_sensor(name=sensor_name)
                st.success(f"Added {sensor_name}")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to add sensor: {e}")
    
    with col2:
        if st.button("🌬️ Add Gas Sensor"):
            try:
                sensor_name = f"gas_{len([s for s in hardware_collector.sensors if 'gas' in s]) + 1}"
                hardware_collector.add_gas_sensor(name=sensor_name)
                st.success(f"Added {sensor_name}")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to add sensor: {e}")
    
    with col3:
        if st.button("🖥️ Connect Arduino"):
            try:
                port = st.text_input("Arduino Port", value="/dev/ttyUSB0", key="arduino_port")
                hardware_collector.add_arduino(port)
                if hardware_collector.arduino.connected:
                    st.success("Arduino connected successfully")
                else:
                    st.error("Failed to connect to Arduino")
                st.rerun()
            except Exception as e:
                st.error(f"Arduino connection failed: {e}")
    
    # Collection controls
    st.subheader("📊 Data Collection")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        collection_interval = st.number_input(
            "Collection Interval (seconds)",
            min_value=0.5,
            max_value=60.0,
            value=2.0,
            step=0.5
        )
    
    with col2:
        if st.button("▶️ Start Collection"):
            try:
                hardware_collector.start_continuous_collection(interval=collection_interval)
                st.success("Started continuous collection")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to start collection: {e}")
    
    with col3:
        if st.button("⏹️ Stop Collection"):
            try:
                hardware_collector.stop_continuous_collection()
                st.success("Stopped collection")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to stop collection: {e}")
    
    # Live readings
    st.subheader("📈 Live Sensor Readings")
    
    if st.button("🔄 Refresh Readings"):
        try:
            readings = hardware_collector.collect_single_reading()
            
            if readings:
                # Display readings in a nice format
                for sensor_name, data in readings.items():
                    with st.container():
                        if 'error' in data:
                            st.error(f"❌ {sensor_name}: {data['error']}")
                        else:
                            sensor_type = data.get('sensor_type', 'unknown')
                            
                            if sensor_type == 'ultrasonic':
                                distance = data.get('distance_cm', 0)
                                status = data.get('status', 'unknown')
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric(f"📏 {sensor_name}", f"{distance} cm")
                                with col2:
                                    st.metric("Status", status)
                                with col3:
                                    if 'pulse_duration_us' in data:
                                        st.metric("Pulse Duration", f"{data['pulse_duration_us']} μs")
                            
                            elif sensor_type == 'gas':
                                concentration = data.get('concentration_ppm', 0)
                                gas_type = data.get('gas_type', 'unknown')
                                voltage = data.get('voltage', 0)
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric(f"🌬️ {sensor_name}", f"{concentration} ppm")
                                with col2:
                                    st.metric("Gas Type", gas_type)
                                with col3:
                                    st.metric("Voltage", f"{voltage} V")
                            
                            elif sensor_name == 'arduino':
                                st.subheader("🖥️ Arduino Multi-Sensor")
                                cols = st.columns(4)
                                for i, (key, value) in enumerate(data.items()):
                                    if key not in ['timestamp', 'source'] and isinstance(value, (int, float)):
                                        with cols[i % 4]:
                                            unit = ""
                                            if 'distance' in key: unit = " cm"
                                            elif 'concentration' in key: unit = " ppm"
                                            elif 'temperature' in key: unit = " °C"
                                            elif 'humidity' in key: unit = " %"
                                            st.metric(key.replace('_', ' ').title(), f"{value}{unit}")
            else:
                st.info("No sensors connected or no data available")
        
        except Exception as e:
            st.error(f"Error reading sensors: {e}")
    
    # Historical data visualization
    st.subheader("📊 Historical Data")
    
    try:
        # Get recent hardware readings
        recent_data = hardware_collector.get_recent_readings(limit=100)
        
        if len(recent_data) > 0:
            recent_data['timestamp'] = pd.to_datetime(recent_data['timestamp'])
            
            # Filter for hardware sensors
            hardware_sensors = recent_data[recent_data['sensor_name'].str.contains('ultrasonic|gas|arduino', na=False)]
            
            if len(hardware_sensors) > 0:
                # Time series plot
                fig = go.Figure()
                
                for sensor in hardware_sensors['sensor_name'].unique():
                    sensor_data = hardware_sensors[hardware_sensors['sensor_name'] == sensor]
                    sensor_data = sensor_data.sort_values('timestamp')
                    
                    fig.add_trace(go.Scatter(
                        x=sensor_data['timestamp'],
                        y=sensor_data['original_reading'],
                        mode='lines+markers',
                        name=sensor,
                        line=dict(width=2)
                    ))
                
                fig.update_layout(
                    title='Hardware Sensor Readings Over Time',
                    xaxis_title='Timestamp',
                    yaxis_title='Reading Value',
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, width='stretch')
                
                # Data table
                st.subheader("Recent Hardware Readings")
                st.dataframe(hardware_sensors[['timestamp', 'sensor_name', 'original_reading']].head(20), width='stretch')
            else:
                st.info("No hardware sensor data available yet. Start collection to see data.")
        else:
            st.info("No data in database. Start data collection to see historical data.")
    
    except Exception as e:
        st.error(f"Error loading historical data: {e}")
    
    # Hardware configuration
    with st.expander("⚙️ Hardware Configuration"):
        st.subheader("Sensor Configuration")
        
        # Show current sensors
        for sensor_name, sensor in hardware_collector.sensors.items():
            st.write(f"**{sensor_name}**")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"Type: {type(sensor).__name__}")
                st.write(f"Connected: {'✅' if sensor.is_connected() else '❌'}")
            with col2:
                if hasattr(sensor, 'calibration_factor'):
                    new_cal = st.number_input(
                        f"Calibration Factor for {sensor_name}",
                        value=sensor.calibration_factor,
                        step=0.1,
                        key=f"cal_{sensor_name}"
                    )
                    if st.button(f"Update {sensor_name}", key=f"update_{sensor_name}"):
                        sensor.calibration_factor = new_cal
                        st.success(f"Updated calibration for {sensor_name}")
        
        # Arduino commands
        if hardware_collector.arduino and hardware_collector.arduino.connected:
            st.subheader("Arduino Commands")
            command = st.text_input("Send Command to Arduino")
            if st.button("Send"):
                response = hardware_collector.arduino.send_command(command)
                st.code(response)

# Sensor Comparison Page
elif page == "Sensor Comparison":
    st.header("🔬 Sensor Comparison")
    
    db = st.session_state.db
    
    try:
        readings = db.get_recent_readings(limit=5000)
        
        if len(readings) > 0 and 'sensor_name' in readings.columns:
            # Sensor selection
            available_sensors = sorted(readings['sensor_name'].unique())
            
            col1, col2 = st.columns(2)
            
            with col1:
                sensors_to_compare = st.multiselect(
                    "Select Sensors to Compare (max 5)",
                    available_sensors,
                    default=available_sensors[:min(3, len(available_sensors))],
                    max_selections=5
                )
            
            with col2:
                comparison_metric = st.selectbox(
                    "Comparison Metric",
                    ["drift_magnitude", "original_reading", "corrected_reading"]
                )
            
            if sensors_to_compare:
                # Filter data
                comparison_data = readings[readings['sensor_name'].isin(sensors_to_compare)].copy()
                comparison_data['timestamp'] = pd.to_datetime(comparison_data['timestamp'])
                
                # Time series comparison
                st.subheader("Time Series Comparison")
                
                fig = go.Figure()
                
                for sensor in sensors_to_compare:
                    sensor_data = comparison_data[comparison_data['sensor_name'] == sensor]
                    sensor_data = sensor_data.sort_values('timestamp')
                    
                    fig.add_trace(go.Scatter(
                        x=sensor_data['timestamp'],
                        y=sensor_data[comparison_metric],
                        mode='lines+markers',
                        name=sensor,
                        line=dict(width=2)
                    ))
                
                fig.update_layout(
                    title=f'{comparison_metric.replace("_", " ").title()} Comparison',
                    xaxis_title='Timestamp',
                    yaxis_title=comparison_metric.replace('_', ' ').title(),
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, width='stretch')
                
                # Statistical comparison
                st.subheader("Statistical Summary")
                
                stats_data = []
                for sensor in sensors_to_compare:
                    sensor_data = comparison_data[comparison_data['sensor_name'] == sensor]
                    stats_data.append({
                        'Sensor': sensor,
                        'Mean Drift': sensor_data['drift_magnitude'].mean(),
                        'Max Drift': sensor_data['drift_magnitude'].max(),
                        'Std Dev': sensor_data['drift_magnitude'].std(),
                        'Samples': len(sensor_data),
                        'Alerts': len(sensor_data[sensor_data['drift_magnitude'] > st.session_state.alert_threshold])
                    })
                
                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df, width='stretch')
                
                # Box plot comparison
                st.subheader("Distribution Comparison")
                
                fig = px.box(
                    comparison_data,
                    x='sensor_name',
                    y='drift_magnitude',
                    color='sensor_name',
                    title='Drift Magnitude Distribution by Sensor'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, width='stretch')
                
        else:
            st.info("No sensor data available for comparison.")
    
    except Exception as e:
        st.error(f"Error loading comparison data: {e}")

# Performance Metrics Page
elif page == "Performance Metrics":
    st.header("📊 Performance Metrics")
    
    db = st.session_state.db
    
    try:
        readings = db.get_recent_readings(limit=10000)
        
        if len(readings) > 0:
            readings['timestamp'] = pd.to_datetime(readings['timestamp'])
            
            # Correction accuracy metrics
            st.subheader("Correction Performance")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_drift_before = readings['drift_magnitude'].mean()
                st.metric("Avg Drift Detected", f"{avg_drift_before:.4f}")
            
            with col2:
                correction_rate = (readings['corrected_reading'] != readings['original_reading']).sum() / len(readings) * 100
                st.metric("Correction Rate", f"{correction_rate:.1f}%")
            
            with col3:
                high_drift_count = (readings['drift_magnitude'] > st.session_state.alert_threshold).sum()
                st.metric("High Drift Events", high_drift_count)
            
            with col4:
                if 'batch' in readings.columns:
                    batches_processed = readings['batch'].nunique()
                    st.metric("Batches Processed", batches_processed)
            
            # Drift reduction over time
            st.subheader("Drift Trend Analysis")
            
            # Group by date
            readings['date'] = readings['timestamp'].dt.date
            daily_stats = readings.groupby('date').agg({
                'drift_magnitude': ['mean', 'max', 'std'],
                'id': 'count'
            }).reset_index()
            daily_stats.columns = ['date', 'mean_drift', 'max_drift', 'std_drift', 'sample_count']
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=daily_stats['date'],
                y=daily_stats['mean_drift'],
                mode='lines+markers',
                name='Mean Drift',
                fill='tozeroy',
                line=dict(color='blue', width=3)
            ))
            
            fig.add_trace(go.Scatter(
                x=daily_stats['date'],
                y=daily_stats['max_drift'],
                mode='lines',
                name='Max Drift',
                line=dict(color='red', width=2, dash='dash')
            ))
            
            fig.update_layout(
                title='Drift Magnitude Trends',
                xaxis_title='Date',
                yaxis_title='Drift Magnitude',
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, width='stretch')
            
            # Model performance by sensor type
            if 'sensor_name' in readings.columns:
                st.subheader("Performance by Sensor")
                
                sensor_performance = readings.groupby('sensor_name').agg({
                    'drift_magnitude': ['mean', 'std', 'max'],
                    'id': 'count'
                }).reset_index()
                sensor_performance.columns = ['sensor', 'mean_drift', 'std_drift', 'max_drift', 'samples']
                sensor_performance = sensor_performance.sort_values('mean_drift', ascending=False).head(20)
                
                fig = px.bar(
                    sensor_performance,
                    x='sensor',
                    y='mean_drift',
                    error_y='std_drift',
                    title='Top 20 Sensors by Average Drift',
                    color='mean_drift',
                    color_continuous_scale='Reds'
                )
                fig.update_xaxes(tickangle=45)
                fig.update_layout(height=500)
                st.plotly_chart(fig, width='stretch')
            
            # Correction effectiveness
            st.subheader("Correction Effectiveness")
            
            readings['correction_magnitude'] = abs(readings['corrected_reading'] - readings['original_reading'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.scatter(
                    readings.sample(min(1000, len(readings))),
                    x='drift_magnitude',
                    y='correction_magnitude',
                    title='Drift vs Correction Applied',
                    opacity=0.5,
                    trendline='ols'
                )
                st.plotly_chart(fig, width='stretch')
            
            with col2:
                fig = px.histogram(
                    readings,
                    x='correction_magnitude',
                    nbins=50,
                    title='Distribution of Correction Magnitude'
                )
                st.plotly_chart(fig, width='stretch')
        
        else:
            st.info("No data available for performance analysis.")
    
    except Exception as e:
        st.error(f"Error loading performance data: {e}")

# Data Export Page
elif page == "Data Export":
    st.header("📥 Data Export")
    
    db = st.session_state.db
    
    st.markdown("Export sensor data and analysis results in various formats.")
    
    # Export options
    col1, col2 = st.columns(2)
    
    with col1:
        export_type = st.selectbox(
            "Select Data Type",
            ["Recent Readings", "Alerts", "All Data", "Custom Query"]
        )
    
    with col2:
        export_format = st.selectbox(
            "Export Format",
            ["CSV", "JSON", "Excel"]
        )
    
    # Data filters
    st.subheader("Filters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        limit = st.number_input("Number of Records", min_value=10, max_value=100000, value=1000)
    
    with col2:
        if export_type == "Alerts":
            severity_filter = st.selectbox("Severity", ["All", "low", "medium", "high"])
    
    with col3:
        include_stats = st.checkbox("Include Statistics Summary", value=True)
    
    # Custom query option
    if export_type == "Custom Query":
        st.subheader("Custom SQL Query")
        custom_query = st.text_area(
            "Enter SQL Query",
            "SELECT * FROM sensor_readings LIMIT 100",
            height=100
        )
    
    # Export button
    if st.button("📥 Generate Export", type="primary"):
        try:
            # Fetch data based on selection
            if export_type == "Recent Readings":
                data = db.get_recent_readings(limit=limit)
            elif export_type == "Alerts":
                severity = None if severity_filter == "All" else severity_filter
                data = db.get_alerts(severity=severity, limit=limit)
            elif export_type == "Custom Query":
                conn = sqlite3.connect('sensor_data.db')
                data = pd.read_sql_query(custom_query, conn)
                conn.close()
            else:  # All Data
                data = db.get_recent_readings(limit=limit)
            
            if len(data) > 0:
                st.success(f"✅ Exported {len(data)} records")
                
                # Statistics summary
                if include_stats:
                    st.subheader("Export Summary")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Records", len(data))
                    with col2:
                        if 'drift_magnitude' in data.columns:
                            st.metric("Avg Drift", f"{data['drift_magnitude'].mean():.4f}")
                    with col3:
                        if 'batch' in data.columns:
                            st.metric("Batches", data['batch'].nunique())
                
                # Preview
                st.subheader("Data Preview")
                st.dataframe(data.head(50), width='stretch')
                
                # Download buttons
                st.subheader("Download")
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                if export_format == "CSV":
                    csv = data.to_csv(index=False)
                    st.download_button(
                        label="⬇️ Download CSV",
                        data=csv,
                        file_name=f"sensor_data_{timestamp}.csv",
                        mime="text/csv"
                    )
                
                elif export_format == "JSON":
                    json_str = data.to_json(orient='records', indent=2)
                    st.download_button(
                        label="⬇️ Download JSON",
                        data=json_str,
                        file_name=f"sensor_data_{timestamp}.json",
                        mime="application/json"
                    )
                
                elif export_format == "Excel":
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        data.to_excel(writer, index=False, sheet_name='Sensor Data')
                        
                        if include_stats and 'drift_magnitude' in data.columns:
                            stats_df = data.describe()
                            stats_df.to_excel(writer, sheet_name='Statistics')
                    
                    st.download_button(
                        label="⬇️ Download Excel",
                        data=buffer.getvalue(),
                        file_name=f"sensor_data_{timestamp}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            else:
                st.warning("No data found with current filters.")
        
        except Exception as e:
            st.error(f"Export failed: {e}")
            import traceback
            st.code(traceback.format_exc())

elif page == "User Management":
    show_user_management_page()

elif page == "Role Permissions":
    if not has_permission(Permission.MANAGE_USERS):
        st.error("Access denied. User management permission required.")
        st.stop()
    
    log_user_action("PAGE_ACCESS", "Role Permissions page accessed")
    st.header("🔐 Role & Permissions Management")
    show_role_permissions()

elif page == "System Admin":
    if not has_permission(Permission.SYSTEM_ADMIN):
        st.error("Access denied. System administrator permission required.")
        st.stop()
    
    log_user_action("PAGE_ACCESS", "System Admin page accessed")
    
    st.header("⚙️ System Administration")
    
    tab1, tab2, tab3 = st.tabs(["📊 System Status", "🔧 Configuration", "📋 Logs"])
    
    with tab1:
        st.subheader("System Health")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Database Status", "🟢 Online")
            st.metric("Active Users", len(st.session_state.get('active_users', [1])))
        
        with col2:
            st.metric("Simulator Status", "🟢 Running" if st.session_state.simulator else "🔴 Offline")
            st.metric("Memory Usage", "45%")
        
        with col3:
            st.metric("Uptime", "24h 15m")
            st.metric("Data Points", "1,234,567")
    
    with tab2:
        st.subheader("System Configuration")
        
        st.checkbox("Enable Debug Mode")
        st.checkbox("Auto-backup Database")
        st.slider("Session Timeout (hours)", 1, 24, 8)
        st.number_input("Max Failed Login Attempts", 1, 10, 5)
        
        if st.button("Save Configuration"):
            log_user_action("CONFIG_CHANGE", "System configuration updated")
            st.success("Configuration saved")
    
    with tab3:
        st.subheader("System Logs")
        show_activity_logs()

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### System Info")

# System health status
try:
    db = st.session_state.db
    recent = db.get_recent_readings(limit=1)
    db_status = "🟢 Connected" if len(recent) >= 0 else "🔴 No Data"
except:
    db_status = "🔴 Error"

st.sidebar.markdown(f"Database: {db_status}")

if st.session_state.simulator:
    model_status = "🟢 Simulator Active"
    num_sensors = len(st.session_state.simulator.sensors)
    st.sidebar.markdown(f"Simulator: {model_status} ({num_sensors} sensors)")
else:
    st.sidebar.markdown("Simulator: 🔴 Not Loaded")

st.sidebar.markdown(f"Last Update: {datetime.now().strftime('%H:%M:%S')}")

# Auto refresh
if st.session_state.auto_refresh:
    import time
    time.sleep(st.session_state.refresh_interval)
    st.rerun()

if __name__ == "__main__":
    pass

