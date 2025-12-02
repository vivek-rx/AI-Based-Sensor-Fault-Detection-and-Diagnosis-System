"""
Database Module for Sensor Data Storage
Manages SQLite database for storing sensor readings, corrected values, and alerts
"""

import sqlite3
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class SensorDatabase:
    """
    SQLite database manager for sensor data and maintenance alerts
    """
    
    def __init__(self, db_path='sensor_data.db'):
        """
        Initialize database connection
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.conn = None
        self.init_database()
    
    def connect(self):
        """Establish database connection"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row  # Enable column access by name
        return self.conn
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def init_database(self):
        """Initialize database tables"""
        conn = self.connect()
        cursor = conn.cursor()
        
        # Table for sensor readings (original and corrected)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sensor_readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                batch INTEGER,
                sensor_name TEXT NOT NULL,
                original_reading REAL,
                corrected_reading REAL,
                drift_estimate REAL,
                drift_magnitude REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Table for maintenance alerts
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS maintenance_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                sensor_name TEXT NOT NULL,
                drift_magnitude REAL,
                severity TEXT NOT NULL,
                original_reading REAL,
                corrected_reading REAL,
                batch INTEGER,
                resolved BOOLEAN DEFAULT 0,
                resolved_at TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Table for batch statistics
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS batch_statistics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                batch INTEGER,
                sample_count INTEGER,
                avg_drift_magnitude REAL,
                max_drift_magnitude REAL,
                alert_count INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Table for sensor health scoring and predictive maintenance
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sensor_health (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                sensor_name TEXT NOT NULL,
                health_score REAL,
                degradation_rate REAL,
                predicted_failure_date TEXT,
                maintenance_urgency TEXT,
                failure_probability REAL,
                remaining_useful_life INTEGER,
                cost_impact REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Table for cross-sensor anomaly detection
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cross_sensor_anomalies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                anomaly_type TEXT NOT NULL,
                affected_sensors TEXT,
                anomaly_score REAL,
                correlation_matrix TEXT,
                potential_cause TEXT,
                severity TEXT,
                resolved BOOLEAN DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Table for cost impact analysis
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cost_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                sensor_name TEXT NOT NULL,
                downtime_cost REAL,
                maintenance_cost REAL,
                replacement_cost REAL,
                production_loss REAL,
                total_cost_impact REAL,
                cost_category TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Table for advanced performance metrics
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL,
                target_value REAL,
                performance_category TEXT,
                improvement_potential REAL,
                benchmark_comparison REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Table for environmental correlation tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS environmental_factors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                temperature REAL,
                humidity REAL,
                pressure REAL,
                vibration REAL,
                electromagnetic_interference REAL,
                correlation_score REAL,
                drift_impact_factor REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes for faster queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_sensor_readings_timestamp ON sensor_readings(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_sensor_readings_sensor ON sensor_readings(sensor_name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON maintenance_alerts(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_sensor ON maintenance_alerts(sensor_name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_resolved ON maintenance_alerts(resolved)')
        
        # Indexes for new tables
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_sensor_health_sensor ON sensor_health(sensor_name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_sensor_health_score ON sensor_health(health_score)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_anomalies_type ON cross_sensor_anomalies(anomaly_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_cost_sensor ON cost_analysis(sensor_name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_name ON performance_metrics(metric_name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_environmental_timestamp ON environmental_factors(timestamp)')
        
        conn.commit()
        print(f"Database initialized: {self.db_path}")
    
    def insert_sensor_reading(self, timestamp, sensor_name, original_reading, 
                              corrected_reading=None, drift_estimate=None, 
                              drift_magnitude=None, batch=None):
        """
        Insert a sensor reading record
        
        Args:
            timestamp: Timestamp string
            sensor_name: Name of the sensor
            original_reading: Original sensor reading
            corrected_reading: Corrected reading
            drift_estimate: Drift estimate
            drift_magnitude: Magnitude of drift
            batch: Batch number
        """
        conn = self.connect()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO sensor_readings 
            (timestamp, batch, sensor_name, original_reading, corrected_reading, 
             drift_estimate, drift_magnitude)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (timestamp, batch, sensor_name, original_reading, corrected_reading,
              drift_estimate, drift_magnitude))
        
        conn.commit()
        return cursor.lastrowid
    
    def insert_alert(self, timestamp, sensor_name, drift_magnitude, severity,
                     original_reading, corrected_reading, batch=None):
        """
        Insert a maintenance alert
        
        Args:
            timestamp: Timestamp string
            sensor_name: Name of the sensor
            drift_magnitude: Magnitude of drift
            severity: 'low', 'medium', or 'high'
            original_reading: Original reading
            corrected_reading: Corrected reading
            batch: Batch number
        """
        conn = self.connect()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO maintenance_alerts 
            (timestamp, sensor_name, drift_magnitude, severity, 
             original_reading, corrected_reading, batch)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (timestamp, sensor_name, drift_magnitude, severity,
              original_reading, corrected_reading, batch))
        
        conn.commit()
        return cursor.lastrowid
    
    def insert_batch_statistics(self, timestamp, batch, sample_count,
                                avg_drift_magnitude, max_drift_magnitude, alert_count):
        """
        Insert batch statistics
        
        Args:
            timestamp: Timestamp string
            batch: Batch number
            sample_count: Number of samples in batch
            avg_drift_magnitude: Average drift magnitude
            max_drift_magnitude: Maximum drift magnitude
            alert_count: Number of alerts in batch
        """
        conn = self.connect()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO batch_statistics 
            (timestamp, batch, sample_count, avg_drift_magnitude, 
             max_drift_magnitude, alert_count)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (timestamp, batch, sample_count, avg_drift_magnitude,
              max_drift_magnitude, alert_count))
        
        conn.commit()
        return cursor.lastrowid
    
    def save_correction_result(self, result, timestamp=None, batch=None):
        """
        Save a complete correction result to database
        
        Args:
            result: Dictionary from DriftCorrector.correct_readings()
            timestamp: Timestamp (if None, uses current time)
            batch: Batch number
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        # Insert sensor readings
        for sensor_name, original in result['original_readings'].items():
            corrected = result['corrected_readings'][sensor_name]
            drift = result['drift_estimates'][sensor_name]
            drift_mag = result['drift_magnitude'][sensor_name]
            
            self.insert_sensor_reading(
                timestamp=timestamp,
                sensor_name=sensor_name,
                original_reading=original,
                corrected_reading=corrected,
                drift_estimate=drift,
                drift_magnitude=drift_mag,
                batch=batch
            )
        
        # Insert alerts
        for alert in result['alerts']:
            self.insert_alert(
                timestamp=timestamp,
                sensor_name=alert['sensor'],
                drift_magnitude=alert['drift_magnitude'],
                severity=alert['severity'],
                original_reading=alert['original_reading'],
                corrected_reading=alert['corrected_reading'],
                batch=batch
            )
    
    def get_recent_readings(self, sensor_name=None, limit=100):
        """
        Get recent sensor readings
        
        Args:
            sensor_name: Filter by sensor (None for all)
            limit: Maximum number of records
            
        Returns:
            DataFrame with readings
        """
        conn = self.connect()
        
        if sensor_name:
            query = '''
                SELECT * FROM sensor_readings 
                WHERE sensor_name = ?
                ORDER BY timestamp DESC
                LIMIT ?
            '''
            df = pd.read_sql_query(query, conn, params=(sensor_name, limit))
        else:
            query = '''
                SELECT * FROM sensor_readings 
                ORDER BY timestamp DESC
                LIMIT ?
            '''
            df = pd.read_sql_query(query, conn, params=(limit,))
        
        return df
    
    def get_alerts(self, resolved=None, severity=None, limit=100):
        """
        Get maintenance alerts
        
        Args:
            resolved: Filter by resolved status (None for all)
            severity: Filter by severity (None for all)
            limit: Maximum number of records
            
        Returns:
            DataFrame with alerts
        """
        conn = self.connect()
        
        conditions = []
        params = []
        
        if resolved is not None:
            conditions.append("resolved = ?")
            params.append(1 if resolved else 0)
        
        if severity:
            conditions.append("severity = ?")
            params.append(severity)
        
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        params.append(limit)
        
        query = f'''
            SELECT * FROM maintenance_alerts 
            {where_clause}
            ORDER BY timestamp DESC
            LIMIT ?
        '''
        
        df = pd.read_sql_query(query, conn, params=tuple(params))
        return df
    
    def get_sensor_statistics(self, sensor_name):
        """
        Get statistics for a specific sensor
        
        Args:
            sensor_name: Name of the sensor
            
        Returns:
            Dictionary with statistics
        """
        conn = self.connect()
        cursor = conn.cursor()
        
        # Get average drift magnitude
        cursor.execute('''
            SELECT AVG(drift_magnitude) as avg_drift,
                   MAX(drift_magnitude) as max_drift,
                   COUNT(*) as total_readings
            FROM sensor_readings
            WHERE sensor_name = ?
        ''', (sensor_name,))
        
        row = cursor.fetchone()
        stats = {
            'avg_drift_magnitude': row['avg_drift'] or 0,
            'max_drift_magnitude': row['max_drift'] or 0,
            'total_readings': row['total_readings'] or 0
        }
        
        # Get alert count
        cursor.execute('''
            SELECT COUNT(*) as alert_count
            FROM maintenance_alerts
            WHERE sensor_name = ? AND resolved = 0
        ''', (sensor_name,))
        
        row = cursor.fetchone()
        stats['active_alert_count'] = row['alert_count'] or 0
        
        return stats
    
    def resolve_alert(self, alert_id):
        """
        Mark an alert as resolved
        
        Args:
            alert_id: ID of the alert
        """
        conn = self.connect()
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE maintenance_alerts
            SET resolved = 1, resolved_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (alert_id,))
        
        conn.commit()
    
    def insert_sensor_health(self, sensor_name, health_score, degradation_rate, 
                            predicted_failure_date, maintenance_urgency, failure_probability,
                            remaining_useful_life, cost_impact, timestamp=None):
        """
        Insert sensor health assessment data
        
        Args:
            sensor_name: Name of the sensor
            health_score: Overall health score (0-100)
            degradation_rate: Rate of degradation per day
            predicted_failure_date: Estimated failure date
            maintenance_urgency: 'low', 'medium', 'high', 'critical'
            failure_probability: Probability of failure (0-1)
            remaining_useful_life: Days until maintenance needed
            cost_impact: Estimated cost impact of failure
            timestamp: Timestamp (if None, uses current time)
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        conn = self.connect()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO sensor_health 
            (timestamp, sensor_name, health_score, degradation_rate, predicted_failure_date,
             maintenance_urgency, failure_probability, remaining_useful_life, cost_impact)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (timestamp, sensor_name, health_score, degradation_rate, predicted_failure_date,
              maintenance_urgency, failure_probability, remaining_useful_life, cost_impact))
        
        conn.commit()
        return cursor.lastrowid
    
    def insert_cross_sensor_anomaly(self, anomaly_type, affected_sensors, anomaly_score,
                                   correlation_matrix, potential_cause, severity, timestamp=None):
        """
        Insert cross-sensor anomaly detection result
        
        Args:
            anomaly_type: Type of anomaly detected
            affected_sensors: List of affected sensor names
            anomaly_score: Anomaly severity score
            correlation_matrix: JSON string of correlation data
            potential_cause: Identified potential cause
            severity: 'low', 'medium', 'high', 'critical'
            timestamp: Timestamp (if None, uses current time)
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        conn = self.connect()
        cursor = conn.cursor()
        
        affected_sensors_str = ','.join(affected_sensors) if isinstance(affected_sensors, list) else str(affected_sensors)
        
        cursor.execute('''
            INSERT INTO cross_sensor_anomalies 
            (timestamp, anomaly_type, affected_sensors, anomaly_score, correlation_matrix,
             potential_cause, severity)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (timestamp, anomaly_type, affected_sensors_str, anomaly_score,
              correlation_matrix, potential_cause, severity))
        
        conn.commit()
        return cursor.lastrowid
    
    def insert_cost_analysis(self, sensor_name, downtime_cost, maintenance_cost,
                           replacement_cost, production_loss, cost_category, timestamp=None):
        """
        Insert cost impact analysis data
        
        Args:
            sensor_name: Name of the sensor
            downtime_cost: Cost of downtime
            maintenance_cost: Cost of maintenance
            replacement_cost: Cost of replacement
            production_loss: Production loss cost
            cost_category: Category of cost analysis
            timestamp: Timestamp (if None, uses current time)
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        total_cost = downtime_cost + maintenance_cost + replacement_cost + production_loss
        
        conn = self.connect()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO cost_analysis 
            (timestamp, sensor_name, downtime_cost, maintenance_cost, replacement_cost,
             production_loss, total_cost_impact, cost_category)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (timestamp, sensor_name, downtime_cost, maintenance_cost, replacement_cost,
              production_loss, total_cost, cost_category))
        
        conn.commit()
        return cursor.lastrowid
    
    def get_sensor_health_scores(self, limit=100):
        """
        Get recent sensor health scores
        
        Args:
            limit: Maximum number of records
            
        Returns:
            DataFrame with health scores
        """
        conn = self.connect()
        query = '''
            SELECT * FROM sensor_health 
            ORDER BY timestamp DESC
            LIMIT ?
        '''
        return pd.read_sql_query(query, conn, params=(limit,))
    
    def get_predictive_maintenance_alerts(self, urgency_threshold='medium'):
        """
        Get sensors requiring predictive maintenance
        
        Args:
            urgency_threshold: Minimum urgency level
            
        Returns:
            DataFrame with maintenance alerts
        """
        conn = self.connect()
        urgency_order = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
        min_urgency = urgency_order.get(urgency_threshold, 2)
        
        query = '''
            SELECT * FROM sensor_health 
            WHERE CASE maintenance_urgency 
                    WHEN 'low' THEN 1
                    WHEN 'medium' THEN 2  
                    WHEN 'high' THEN 3
                    WHEN 'critical' THEN 4
                    ELSE 0 END >= ?
            ORDER BY failure_probability DESC, health_score ASC
        '''
        return pd.read_sql_query(query, conn, params=(min_urgency,))
    
    def get_cross_sensor_anomalies(self, resolved=False, limit=100):
        """
        Get cross-sensor anomalies
        
        Args:
            resolved: Filter by resolved status
            limit: Maximum number of records
            
        Returns:
            DataFrame with anomalies
        """
        conn = self.connect()
        query = '''
            SELECT * FROM cross_sensor_anomalies 
            WHERE resolved = ?
            ORDER BY anomaly_score DESC, timestamp DESC
            LIMIT ?
        '''
        return pd.read_sql_query(query, conn, params=(1 if resolved else 0, limit))
    
    def get_cost_analysis(self, sensor_name=None, limit=100):
        """
        Get cost analysis data
        
        Args:
            sensor_name: Filter by sensor (None for all)
            limit: Maximum number of records
            
        Returns:
            DataFrame with cost analysis
        """
        conn = self.connect()
        
        if sensor_name:
            query = '''
                SELECT * FROM cost_analysis 
                WHERE sensor_name = ?
                ORDER BY total_cost_impact DESC, timestamp DESC
                LIMIT ?
            '''
            return pd.read_sql_query(query, conn, params=(sensor_name, limit))
        else:
            query = '''
                SELECT * FROM cost_analysis 
                ORDER BY total_cost_impact DESC, timestamp DESC
                LIMIT ?
            '''
            return pd.read_sql_query(query, conn, params=(limit,))
    
    def calculate_roi_metrics(self):
        """
        Calculate Return on Investment metrics for the drift correction system
        
        Returns:
            Dictionary with ROI metrics
        """
        conn = self.connect()
        cursor = conn.cursor()
        
        # Calculate prevented costs through early detection
        cursor.execute('''
            SELECT SUM(total_cost_impact) as total_prevented_cost
            FROM cost_analysis
            WHERE cost_category = 'prevented'
        ''')
        prevented_cost = cursor.fetchone()[0] or 0
        
        # Calculate actual maintenance costs
        cursor.execute('''
            SELECT SUM(maintenance_cost) as total_maintenance_cost
            FROM cost_analysis
        ''')
        maintenance_cost = cursor.fetchone()[0] or 0
        
        # Calculate ROI
        roi = (prevented_cost - maintenance_cost) / maintenance_cost * 100 if maintenance_cost > 0 else 0
        
        return {
            'prevented_cost': prevented_cost,
            'maintenance_cost': maintenance_cost,
            'roi_percentage': roi,
            'net_savings': prevented_cost - maintenance_cost
        }
    
    def clear_old_data(self, days=30):
        """
        Delete data older than specified days
        
        Args:
            days: Number of days to keep
        """
        conn = self.connect()
        cursor = conn.cursor()
        
        cutoff_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        cutoff_date = cutoff_date.replace(day=cutoff_date.day - days)
        cutoff_str = cutoff_date.isoformat()
        
        # Delete old readings
        cursor.execute('DELETE FROM sensor_readings WHERE timestamp < ?', (cutoff_str,))
        readings_deleted = cursor.rowcount
        
        # Delete old alerts (only resolved ones)
        cursor.execute('''
            DELETE FROM maintenance_alerts 
            WHERE timestamp < ? AND resolved = 1
        ''', (cutoff_str,))
        alerts_deleted = cursor.rowcount
        
        # Clean up old health data
        cursor.execute('DELETE FROM sensor_health WHERE timestamp < ?', (cutoff_str,))
        health_deleted = cursor.rowcount
        
        # Clean up old cost analysis (keep longer for trend analysis)
        old_cutoff = cutoff_date.replace(day=cutoff_date.day - 90)
        cursor.execute('DELETE FROM cost_analysis WHERE timestamp < ?', (old_cutoff.isoformat(),))
        cost_deleted = cursor.rowcount
        
        conn.commit()
        
        print(f"Deleted {readings_deleted} old readings, {alerts_deleted} resolved alerts, "
              f"{health_deleted} health records, and {cost_deleted} old cost records")
        return readings_deleted, alerts_deleted, health_deleted, cost_deleted


if __name__ == "__main__":
    # Example usage
    db = SensorDatabase('sensor_data.db')
    
    # Example: insert a sample reading
    timestamp = datetime.now().isoformat()
    db.insert_sensor_reading(
        timestamp=timestamp,
        sensor_name='sensor_1',
        original_reading=100.5,
        corrected_reading=98.2,
        drift_estimate=2.3,
        drift_magnitude=2.3,
        batch=1
    )
    
    # Example: insert an alert
    db.insert_alert(
        timestamp=timestamp,
        sensor_name='sensor_1',
        drift_magnitude=4.5,
        severity='high',
        original_reading=100.5,
        corrected_reading=98.2,
        batch=1
    )
    
    # Example: insert sensor health data
    db.insert_sensor_health(
        sensor_name='sensor_1',
        health_score=85.5,
        degradation_rate=0.1,
        predicted_failure_date='2025-03-15',
        maintenance_urgency='medium',
        failure_probability=0.15,
        remaining_useful_life=45,
        cost_impact=25000.0
    )
    
    # Example: insert cost analysis
    db.insert_cost_analysis(
        sensor_name='sensor_1',
        downtime_cost=5000.0,
        maintenance_cost=2000.0,
        replacement_cost=8000.0,
        production_loss=10000.0,
        cost_category='predictive'
    )
    
    # Query recent readings
    readings = db.get_recent_readings(limit=10)
    print("\nRecent readings:")
    print(readings.head())
    
    # Query alerts
    alerts = db.get_alerts(resolved=False, limit=10)
    print("\nActive alerts:")
    print(alerts.head())
    
    # Query health scores
    health = db.get_sensor_health_scores(limit=10)
    print("\nSensor health scores:")
    print(health.head())
    
    # Calculate ROI
    roi_metrics = db.calculate_roi_metrics()
    print("\nROI Metrics:")
    print(f"ROI: {roi_metrics['roi_percentage']:.2f}%")
    print(f"Net Savings: ${roi_metrics['net_savings']:,.2f}")
    
    db.close()



