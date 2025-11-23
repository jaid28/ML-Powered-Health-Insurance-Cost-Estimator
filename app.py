from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, session
from flask_cors import CORS
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt
import pandas as pd
import joblib
import sqlite3
from datetime import datetime
import random
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
import os

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_ADDRESS = os.getenv('EMAIL_ADDRESS')
EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD')

otp_storage = {}

load_dotenv()

app = Flask(__name__)
# app.secret_key = os.getenv('SECRET_KEY')
app.secret_key = 'jaid28-super-secret-2025'
CORS(app)
bcrypt = Bcrypt(app)

# Flask-Login Setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message_category = "info"

# Load ML models
scaler = joblib.load("models/scaler.pkl")
le_gender = joblib.load("models/label_encoder_gender.pkl")
le_diabetic = joblib.load("models/label_encoder_diabetic.pkl")
le_smoker = joblib.load("models/label_encoder_smoker.pkl")
model = joblib.load("models/best_model.pkl")

# User Model
class User(UserMixin):
    def __init__(self, id, username, email):
        self.id = id
        self.username = username
        self.email = email

@login_manager.user_loader
def load_user(user_id):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT id, username, email FROM users WHERE id = ?", (user_id,))
    row = c.fetchone()
    conn.close()
    if row:
        return User(row[0], row[1], row[2])
    return None

# Database Init
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                 id INTEGER PRIMARY KEY,
                 username TEXT UNIQUE NOT NULL,
                 email TEXT UNIQUE NOT NULL,
                 password TEXT NOT NULL,
                 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                 )''')
    conn.commit()
    conn.close()

init_db()

# Routes
@app.route('/')
def home():
    if current_user.is_authenticated:
        return render_template('index.html')    # already logged in → go straight to predictor
    return redirect(url_for('register'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
        
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        hashed = bcrypt.generate_password_hash(password).decode('utf-8')
        
        try:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                     (username, email, hashed))
            conn.commit()
            flash('Account created! You can now log in.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username or email already exists!', 'danger')
        finally:
            conn.close()
    
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])  # FIXED: was "candid methods"
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
        
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("SELECT id, username, email, password FROM users WHERE username = ?", (username,))
        user = c.fetchone()
        conn.close()
        
        if user and bcrypt.check_password_hash(user[3], password):
            user_obj = User(user[0], user[1], user[2])
            login_user(user_obj)
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password', 'danger')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out. See you soon!', 'info')
    return redirect(url_for('register'))

# 1. Forgot Password Page
@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form['email']
        
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("SELECT username FROM users WHERE email = ?", (email,))
        user = c.fetchone()
        conn.close()
        
        if not user:
            flash('No account found with that email.', 'danger')
            return redirect(url_for('forgot_password'))
        
        # Generate OTP
        otp = random.randint(100000, 999999)
        otp_storage[email] = {"otp": otp, "username": user[0]}
        
        # Send email
        try:
            msg = MIMEMultipart()
            msg['From'] = EMAIL_ADDRESS
            msg['To'] = email
            msg['Subject'] = "Your Password Reset OTP - Insurance Predictor"
            
            body = f"""
            <h2>Password Reset Request</h2>
            <p>Hello {user[0]},</p>
            <p>Your OTP is: <b style="font-size:20px">{otp}</b></p>
            <p>It expires in 10 minutes.</p>
            <p>If you didn't request this, ignore this email.</p>
            """
            msg.attach(MIMEText(body, 'html'))
            
            server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.sendmail(EMAIL_ADDRESS, email, msg.as_string())
            server.quit()
            
            flash('OTP sent to your email!', 'success')
            return redirect(url_for('verify_otp', email=email))
            
        except Exception as e:
            flash('Failed to send email. Try again.', 'danger')
    
    return render_template('forgot_password.html')

# 2. Verify OTP Page
@app.route('/verify-otp/<email>', methods=['GET', 'POST'])
def verify_otp(email):
    if request.method == 'POST':
        entered_otp = request.form['otp']
        stored = otp_storage.get(email)
        
        if stored and str(stored['otp']) == entered_otp:
            session['reset_user'] = stored['username']
            del otp_storage[email]
            return redirect(url_for('reset_password'))
        else:
            flash('Invalid or expired OTP', 'danger')
    
    return render_template('verify_otp.html', email=email)

# 3. Reset Password Page
@app.route('/reset-password', methods=['GET', 'POST'])
def reset_password():
    if 'reset_user' not in session:
        return redirect(url_for('login'))
        
    if request.method == 'POST':
        new_pass = request.form['password']
        username = session['reset_user']
        
        hashed = bcrypt.generate_password_hash(new_pass).decode('utf-8')
        
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("UPDATE users SET password = ? WHERE username = ?", (hashed, username))
        conn.commit()
        conn.close()
        
        session.pop('reset_user', None)
        flash('Password reset successful! You can now login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('reset_password.html')

# existing predict route (now protected)
@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        data = request.get_json(force=True)
        
        input_data = pd.DataFrame([{
            "age": data['age'],
            "gender": data['gender'],
            "bmi": data['bmi'],
            "bloodpressure": data['bloodpressure'],
            "diabetic": data['diabetic'],
            "children": data['children'],
            "smoker": data['smoker']
        }])

        # Encoding & Scaling (same as before)
        input_data["gender"] = le_gender.transform(input_data["gender"])
        input_data["diabetic"] = le_diabetic.transform(input_data["diabetic"])
        input_data["smoker"] = le_smoker.transform(input_data["smoker"])
        num_cols = ["age", "bmi", "bloodpressure", "children"]
        input_data[num_cols] = scaler.transform(input_data[num_cols])

        prediction = model.predict(input_data)[0]

        return jsonify({
            'prediction': round(float(prediction), 2),
            'status': 'success'
        })

    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 400

@app.route('/admin-Jayd')
@login_required
def admin_users():
    
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT id, username, email, created_at FROM users ORDER BY created_at DESC")
    all_users = c.fetchall()
    conn.close()
    
    html = """
    <h1 style="text-align:center; color:#1e40af; font-family:Arial">All Registered Users (Only You Can See This)</h1>
    <table border="1" style="width:90%; margin:30px auto; border-collapse:collapse; font-family:Arial">
        <tr style="background:#3b82f6; color:white">
            <th style="padding:15px">ID</th>
            <th style="padding:15px">Username</th>
            <th style="padding:15px">Email</th>
            <th style="padding:15px">Registered On</th>
        </tr>
    """
    for user in all_users:
        html += f"""
        <tr style="text-align:center">
            <td style="padding:10px">{user[0]}</td>
            <td style="padding:10px">{user[1]}</td>
            <td style="padding:10px">{user[2]}</td>
            <td style="padding:10px">{user[3]}</td>
        </tr>
        """
    html += "</table>"
    html += '<p style="text-align:center"><a href="/">Back to Predictor</a></p>'
    return html

if __name__ == '__main__':
    app.run(debug=True, port=5000)