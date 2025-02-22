import streamlit as st
import bcrypt
import json
from pathlib import Path

class Auth:
    def __init__(self):
        self.users_file = Path("frontend/data/users.json")
        self.users_file.parent.mkdir(exist_ok=True)
        if not self.users_file.exists():
            self.users_file.write_text(json.dumps({}))
        
    def load_users(self):
        """Load users from JSON file"""
        return json.loads(self.users_file.read_text())
    
    def save_users(self, users):
        """Save users to JSON file"""
        self.users_file.write_text(json.dumps(users, indent=2))
    
    def hash_password(self, password):
        """Hash password using bcrypt"""
        return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    
    def verify_password(self, password, hashed):
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode(), hashed.encode())
    
    def register_user(self, username, password):
        """Register a new user"""
        users = self.load_users()
        if username in users:
            return False, "Username already exists"
        
        users[username] = {
            "password": self.hash_password(password),
            "role": "user"
        }
        self.save_users(users)
        return True, "Registration successful"
    
    def login_user(self, username, password):
        """Login a user"""
        users = self.load_users()
        if username not in users:
            return False, "Invalid username"
        
        if not self.verify_password(password, users[username]["password"]):
            return False, "Invalid password"
        
        return True, users[username]

    def show_login_page(self):
        """Display login/register interface"""
        st.title("JoromiGPT Login")
        
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submit = st.form_submit_button("Login")
                
                if submit:
                    success, result = self.login_user(username, password)
                    if success:
                        st.session_state.user = username
                        st.session_state.role = result["role"]
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error(result)
        
        with tab2:
            with st.form("register_form"):
                new_username = st.text_input("New Username")
                new_password = st.text_input("New Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")
                submit = st.form_submit_button("Register")
                
                if submit:
                    if new_password != confirm_password:
                        st.error("Passwords don't match")
                    else:
                        success, message = self.register_user(new_username, new_password)
                        if success:
                            st.success(message)
                        else:
                            st.error(message) 