#!/usr/bin/env python3
"""
User Authentication and Authorization System
Industrial-grade security for sensor monitoring system
"""

import hashlib
import secrets
import sqlite3
import streamlit as st
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from enum import Enum
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UserRole(Enum):
    """User roles with hierarchical permissions"""
    ADMIN = "admin"
    OPERATOR = "operator" 
    VIEWER = "viewer"
    MAINTENANCE = "maintenance"

class Permission(Enum):
    """System permissions"""
    VIEW_DASHBOARD = "view_dashboard"
    VIEW_SENSORS = "view_sensors"
    MODIFY_SETTINGS = "modify_settings"
    MANAGE_USERS = "manage_users"
    EXPORT_DATA = "export_data"
    HARDWARE_CONTROL = "hardware_control"
    MAINTENANCE_MODE = "maintenance_mode"
    SYSTEM_ADMIN = "system_admin"

class UserManager:
    """Manages user authentication, authorization, and activity logging"""
    
    def __init__(self, db_path: str = 'sensor_data.db'):
        self.db_path = db_path
        self.init_user_tables()
        self.role_permissions = self._define_role_permissions()
        
    def init_user_tables(self):
        """Initialize user management database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                role TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_login TEXT,
                is_active BOOLEAN DEFAULT 1,
                failed_login_attempts INTEGER DEFAULT 0,
                locked_until TEXT,
                session_token TEXT,
                token_expires TEXT
            )
        ''')
        
        # Activity log table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS activity_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                username TEXT,
                action TEXT NOT NULL,
                details TEXT,
                ip_address TEXT,
                timestamp TEXT NOT NULL,
                session_id TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # User sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                session_token TEXT UNIQUE NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                is_active BOOLEAN DEFAULT 1,
                ip_address TEXT,
                user_agent TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        conn.commit()
        conn.close()
        
        # Create default admin user if no users exist
        self._create_default_admin()
        
    def _create_default_admin(self):
        """Create default admin user if no users exist"""
        if not self.user_exists("admin"):
            self.create_user(
                username="admin",
                email="admin@sensor-system.com",
                password="admin123",  # Should be changed on first login
                role=UserRole.ADMIN
            )
            logger.info("Default admin user created")
            
    def _define_role_permissions(self) -> Dict[UserRole, List[Permission]]:
        """Define permissions for each role"""
        return {
            UserRole.ADMIN: [
                Permission.VIEW_DASHBOARD,
                Permission.VIEW_SENSORS,
                Permission.MODIFY_SETTINGS,
                Permission.MANAGE_USERS,
                Permission.EXPORT_DATA,
                Permission.HARDWARE_CONTROL,
                Permission.MAINTENANCE_MODE,
                Permission.SYSTEM_ADMIN
            ],
            UserRole.OPERATOR: [
                Permission.VIEW_DASHBOARD,
                Permission.VIEW_SENSORS,
                Permission.EXPORT_DATA,
                Permission.HARDWARE_CONTROL
            ],
            UserRole.VIEWER: [
                Permission.VIEW_DASHBOARD,
                Permission.VIEW_SENSORS
            ],
            UserRole.MAINTENANCE: [
                Permission.VIEW_DASHBOARD,
                Permission.VIEW_SENSORS,
                Permission.HARDWARE_CONTROL,
                Permission.MAINTENANCE_MODE
            ]
        }
    
    def _hash_password(self, password: str, salt: str = None) -> tuple[str, str]:
        """Hash password with salt"""
        if salt is None:
            salt = secrets.token_hex(32)
        
        password_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000  # iterations
        )
        return password_hash.hex(), salt
    
    def create_user(self, username: str, email: str, password: str, role: UserRole) -> bool:
        """Create new user"""
        try:
            password_hash, salt = self._hash_password(password)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO users (username, email, password_hash, salt, role, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (username, email, password_hash, salt, role.value, datetime.now().isoformat()))
            
            conn.commit()
            conn.close()
            
            self.log_activity(None, username, "USER_CREATED", f"New user created with role {role.value}")
            return True
            
        except sqlite3.IntegrityError as e:
            logger.error(f"User creation failed: {e}")
            return False
    
    def authenticate_user(self, username: str, password: str) -> Optional[Dict]:
        """Authenticate user and return user info"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, username, email, password_hash, salt, role, is_active, 
                   failed_login_attempts, locked_until
            FROM users WHERE username = ?
        ''', (username,))
        
        user = cursor.fetchone()
        conn.close()
        
        if not user:
            self.log_activity(None, username, "LOGIN_FAILED", "User not found")
            return None
            
        user_id, db_username, email, stored_hash, salt, role, is_active, failed_attempts, locked_until = user
        
        # Check if account is locked
        if locked_until:
            lock_time = datetime.fromisoformat(locked_until)
            if datetime.now() < lock_time:
                self.log_activity(user_id, username, "LOGIN_BLOCKED", "Account locked")
                return None
        
        # Check if account is active
        if not is_active:
            self.log_activity(user_id, username, "LOGIN_FAILED", "Account inactive")
            return None
        
        # Verify password
        password_hash, _ = self._hash_password(password, salt)
        
        if password_hash == stored_hash:
            # Successful login
            self._reset_failed_attempts(user_id)
            self._update_last_login(user_id)
            session_token = self._create_session(user_id)
            
            user_info = {
                'id': user_id,
                'username': db_username,
                'email': email,
                'role': UserRole(role),
                'session_token': session_token
            }
            
            self.log_activity(user_id, username, "LOGIN_SUCCESS", "User logged in successfully")
            return user_info
        else:
            # Failed login
            self._increment_failed_attempts(user_id)
            self.log_activity(user_id, username, "LOGIN_FAILED", "Invalid password")
            return None
    
    def _create_session(self, user_id: int) -> str:
        """Create new session for user"""
        session_token = secrets.token_urlsafe(32)
        expires_at = datetime.now() + timedelta(hours=24)  # 24 hour sessions
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO user_sessions (user_id, session_token, created_at, expires_at, ip_address)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, session_token, datetime.now().isoformat(), expires_at.isoformat(), 
              self._get_client_ip()))
        
        conn.commit()
        conn.close()
        
        return session_token
    
    def validate_session(self, session_token: str) -> Optional[Dict]:
        """Validate session token and return user info"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT s.user_id, u.username, u.email, u.role, s.expires_at
            FROM user_sessions s
            JOIN users u ON s.user_id = u.id
            WHERE s.session_token = ? AND s.is_active = 1 AND u.is_active = 1
        ''', (session_token,))
        
        session = cursor.fetchone()
        conn.close()
        
        if not session:
            return None
            
        user_id, username, email, role, expires_at = session
        
        # Check if session expired
        if datetime.now() > datetime.fromisoformat(expires_at):
            self._invalidate_session(session_token)
            return None
        
        return {
            'id': user_id,
            'username': username,
            'email': email,
            'role': UserRole(role),
            'session_token': session_token
        }
    
    def logout_user(self, session_token: str):
        """Logout user by invalidating session"""
        user_info = self.validate_session(session_token)
        if user_info:
            self._invalidate_session(session_token)
            self.log_activity(user_info['id'], user_info['username'], "LOGOUT", "User logged out")
    
    def has_permission(self, user_role: UserRole, permission: Permission) -> bool:
        """Check if user role has specific permission"""
        return permission in self.role_permissions.get(user_role, [])
    
    def user_exists(self, username: str) -> bool:
        """Check if username exists"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM users WHERE username = ?", (username,))
        exists = cursor.fetchone() is not None
        conn.close()
        return exists
    
    def get_all_users(self) -> List[Dict]:
        """Get all users (admin only)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, username, email, role, created_at, last_login, is_active
            FROM users ORDER BY created_at DESC
        ''')
        
        users = []
        for row in cursor.fetchall():
            users.append({
                'id': row[0],
                'username': row[1], 
                'email': row[2],
                'role': row[3],
                'created_at': row[4],
                'last_login': row[5],
                'is_active': bool(row[6])
            })
        
        conn.close()
        return users
    
    def update_user_status(self, user_id: int, is_active: bool, admin_user_id: int):
        """Enable/disable user account"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("UPDATE users SET is_active = ? WHERE id = ?", (is_active, user_id))
        conn.commit()
        conn.close()
        
        action = "USER_ENABLED" if is_active else "USER_DISABLED"
        self.log_activity(admin_user_id, None, action, f"User ID {user_id} status changed")
    
    def log_activity(self, user_id: Optional[int], username: Optional[str], 
                    action: str, details: str = None):
        """Log user activity"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO activity_log (user_id, username, action, details, ip_address, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (user_id, username, action, details, self._get_client_ip(), datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
    
    def get_activity_log(self, limit: int = 100) -> List[Dict]:
        """Get recent activity log"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT username, action, details, ip_address, timestamp
            FROM activity_log 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))
        
        activities = []
        for row in cursor.fetchall():
            activities.append({
                'username': row[0],
                'action': row[1],
                'details': row[2],
                'ip_address': row[3],
                'timestamp': row[4]
            })
        
        conn.close()
        return activities
    
    def _reset_failed_attempts(self, user_id: int):
        """Reset failed login attempts"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE users SET failed_login_attempts = 0, locked_until = NULL 
            WHERE id = ?
        ''', (user_id,))
        conn.commit()
        conn.close()
    
    def _increment_failed_attempts(self, user_id: int):
        """Increment failed login attempts and lock account if needed"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT failed_login_attempts FROM users WHERE id = ?", (user_id,))
        attempts = cursor.fetchone()[0] + 1
        
        # Lock account after 5 failed attempts for 30 minutes
        locked_until = None
        if attempts >= 5:
            locked_until = (datetime.now() + timedelta(minutes=30)).isoformat()
        
        cursor.execute('''
            UPDATE users SET failed_login_attempts = ?, locked_until = ? 
            WHERE id = ?
        ''', (attempts, locked_until, user_id))
        
        conn.commit()
        conn.close()
    
    def _update_last_login(self, user_id: int):
        """Update last login timestamp"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET last_login = ? WHERE id = ?", 
                      (datetime.now().isoformat(), user_id))
        conn.commit()
        conn.close()
    
    def _invalidate_session(self, session_token: str):
        """Invalidate user session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("UPDATE user_sessions SET is_active = 0 WHERE session_token = ?", 
                      (session_token,))
        conn.commit()
        conn.close()
    
    def _get_client_ip(self) -> str:
        """Get client IP address (placeholder - would need proper implementation)"""
        return "127.0.0.1"  # In real implementation, extract from request headers

def require_permission(permission: Permission):
    """Decorator to require specific permission for functions"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            if 'user' not in st.session_state:
                st.error("Authentication required")
                st.stop()
            
            user_manager = UserManager()
            user_role = st.session_state.user['role']
            
            if not user_manager.has_permission(user_role, permission):
                st.error("Insufficient permissions")
                st.stop()
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

def require_login():
    """Decorator to require login for functions"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            if 'user' not in st.session_state:
                st.error("Please log in to access this page")
                st.stop()
            return func(*args, **kwargs)
        return wrapper
    return decorator