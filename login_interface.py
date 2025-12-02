#!/usr/bin/env python3
"""
Login and User Management Interface
Streamlit components for authentication and user management
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from auth_system import UserManager, UserRole, Permission
from typing import Dict, Optional

def show_login_page():
    """Display login page"""
    st.markdown("<h1 style='text-align: center;'>🔐 Industrial Sensor System</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Secure access to industrial sensor monitoring system</p>", unsafe_allow_html=True)
    
    # Create columns for better layout
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### Please log in to continue")
        
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            
            submit_button = st.form_submit_button("Login", use_container_width=True)
            
            if submit_button:
                if username and password:
                    user_manager = UserManager()
                    user_info = user_manager.authenticate_user(username, password)
                    
                    if user_info:
                        st.session_state.user = user_info
                        st.session_state.authenticated = True
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
                else:
                    st.warning("Please enter both username and password")

def show_logout_button():
    """Display logout button in sidebar"""
    if st.sidebar.button("🚪 Logout"):
        if 'user' in st.session_state:
            user_manager = UserManager()
            user_manager.logout_user(st.session_state.user.get('session_token'))
            
        # Clear session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

def show_user_info():
    """Display current user info in sidebar"""
    if 'user' in st.session_state:
        user = st.session_state.user
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 👤 Current User")
        st.sidebar.markdown(f"**{user['username']}**")
        st.sidebar.markdown(f"Role: {user['role'].value.title()}")
        st.sidebar.markdown(f"Email: {user['email']}")

def show_user_management_page():
    """Display user management page (Admin only)"""
    if not has_permission(Permission.MANAGE_USERS):
        st.error("Access denied. Admin privileges required.")
        return
    
    st.header("👥 User Management")
    
    user_manager = UserManager()
    
    # Tabs for different user management functions
    tab1, tab2, tab3 = st.tabs(["👥 All Users", "➕ Add User", "📋 Activity Log"])
    
    with tab1:
        st.subheader("System Users")
        
        users = user_manager.get_all_users()
        
        if users:
            df = pd.DataFrame(users)
            df['created_at'] = pd.to_datetime(df['created_at']).dt.strftime('%Y-%m-%d %H:%M')
            df['last_login'] = pd.to_datetime(df['last_login'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M')
            
            # Display users table
            st.dataframe(
                df[['username', 'email', 'role', 'created_at', 'last_login', 'is_active']],
                use_container_width=True
            )
            
            # User actions
            st.subheader("User Actions")
            selected_user = st.selectbox("Select User", options=[u['username'] for u in users])
            
            if selected_user:
                selected_user_data = next(u for u in users if u['username'] == selected_user)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("Enable User" if not selected_user_data['is_active'] else "Disable User"):
                        new_status = not selected_user_data['is_active']
                        user_manager.update_user_status(
                            selected_user_data['id'], 
                            new_status,
                            st.session_state.user['id']
                        )
                        st.success(f"User {'enabled' if new_status else 'disabled'}")
                        st.rerun()
                
                with col2:
                    if st.button("Reset Password"):
                        st.info("Password reset functionality would be implemented here")
        else:
            st.info("No users found")
    
    with tab2:
        st.subheader("Add New User")
        
        with st.form("add_user_form"):
            new_username = st.text_input("Username")
            new_email = st.text_input("Email")
            new_password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            new_role = st.selectbox("Role", options=[role.value for role in UserRole])
            
            if st.form_submit_button("Create User"):
                if new_password != confirm_password:
                    st.error("Passwords do not match")
                elif len(new_password) < 6:
                    st.error("Password must be at least 6 characters")
                else:
                    success = user_manager.create_user(
                        new_username, new_email, new_password, UserRole(new_role)
                    )
                    if success:
                        st.success("User created successfully!")
                    else:
                        st.error("Failed to create user. Username or email may already exist.")
    
    with tab3:
        st.subheader("Activity Log")
        
        activities = user_manager.get_activity_log(limit=50)
        
        if activities:
            df = pd.DataFrame(activities)
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
            
            st.dataframe(
                df[['timestamp', 'username', 'action', 'details', 'ip_address']],
                use_container_width=True
            )
        else:
            st.info("No activity logged yet")

def show_activity_logs():
    """Display activity logs"""
    user_manager = UserManager()
    
    activities = user_manager.get_activity_log(limit=100)
    
    if activities:
        df = pd.DataFrame(activities)
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
        
        st.dataframe(
            df[['timestamp', 'username', 'action', 'details', 'ip_address']],
            use_container_width=True
        )
    else:
        st.info("No activity logged yet")

def show_role_permissions():
    """Display role permissions information"""
    st.subheader("🔐 Role Permissions")
    
    user_manager = UserManager()
    
    # Create permissions matrix
    roles = list(UserRole)
    permissions = list(Permission)
    
    data = []
    for role in roles:
        row = {'Role': role.value.title()}
        for perm in permissions:
            row[perm.value.replace('_', ' ').title()] = '✅' if user_manager.has_permission(role, perm) else '❌'
        data.append(row)
    
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)

def has_permission(permission: Permission) -> bool:
    """Check if current user has permission"""
    if 'user' not in st.session_state:
        return False
    
    user_manager = UserManager()
    return user_manager.has_permission(st.session_state.user['role'], permission)

def require_authentication():
    """Check if user is authenticated, show login if not"""
    if 'authenticated' not in st.session_state or not st.session_state.authenticated:
        show_login_page()
        st.stop()

def require_permission_check(permission: Permission, error_message: str = None):
    """Check permission and show error if not authorized"""
    if not has_permission(permission):
        st.error(error_message or f"Access denied. Required permission: {permission.value}")
        st.stop()

def get_user_role_display() -> str:
    """Get current user's role for display"""
    if 'user' in st.session_state:
        return st.session_state.user['role'].value.title()
    return "Not Authenticated"

def get_username() -> str:
    """Get current username"""
    if 'user' in st.session_state:
        return st.session_state.user['username']
    return "Guest"

def log_user_action(action: str, details: str = None):
    """Log user action"""
    if 'user' in st.session_state:
        user_manager = UserManager()
        user_manager.log_activity(
            st.session_state.user['id'],
            st.session_state.user['username'],
            action,
            details
        )