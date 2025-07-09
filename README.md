Intelligent Data Analysis Dashboard

An interactive web application built with Flask and Dash that empowers users to upload datasets and instantly generate automated summaries, data quality assessments, and smart visualizations without writing any code.
This project allows anyone, regardless of their technical background, to gain meaningful insights from their data through a secure, intuitive, and intelligent user interface.
Live Demo: https://datasense-80w9.onrender.com/

An interactive web application built with Flask and Dash that empowers users to upload datasets and instantly generate automated summaries, data quality assessments, and smart visualizations without writing any code.This project allows anyone, regardless of their technical background, to gain meaningful insights from their data through a secure, intuitive, and intelligent user interface.

✨ Key Features
Secure User Authentication: Full signup, login, and session management using Firebase Authentication.
Versatile File Upload: Supports both CSV and Excel file formats.
Automated Natural Language Summary: Instantly get a human-readable paragraph describing the key characteristics of your dataset.
In-Depth Data Quality Assessment: Automatically detects missing values, duplicate rows, and columns with no variance.
Smart Visualization Suggestions: An intelligent engine analyzes your data's structure and recommends the most relevant plots (Time Series, Box Plots, Correlation Heatmaps, etc.).
Interactive Plotting Controls: Manually create and customize a wide range of visualizations to explore your data.
Persistent Analysis History: Every upload and visualization action is saved to a personal history log using Firestore.

🛠️ Technology Stack
Backend: Flask, Gunicorn
Dashboarding: Dash, Plotly Express
Data Manipulation: Pandas, NumPy
Authentication & Database: Pyrebase (Auth), Firebase Admin SDK (Firestore)
Styling: Dash Bootstrap Components
Deployment: Render


🚀 Getting Started
Follow these instructions to set up and run the project on your local machine.
Prerequisites
Python 3.9+
A Firebase project with Authentication and Firestore enabled.
1. Clone the Repository
git clone https://github.com/VaishnavDevaraj/Intelligent-Automation-for-Data-Analysis.git
cd Intelligent-Automation-for-Data-Analysis


2. Set Up a Virtual Environment
It's highly recommended to use a virtual environment to manage dependencies.
Windows:
python -m venv venv
.\venv\Scripts\activate

macOS / Linux:
python3 -m venv venv
source venv/bin/activate


3. Install Dependencies
pip install -r requirements.txt


4. Configure Environment Variables
This project uses environment variables to keep your secret keys secure.
Create a file named .env in the root of the project.
Copy the content from the example below and paste it into your .env file.
Fill in the values with your own credentials from your Firebase project and a secure secret key.
.env.example
# === Pyrebase Config (Found in your Firebase project settings -> General) ==========
Backend: Flask, GunicornDashboarding: Dash, Plotly ExpressData Manipulation: Pandas, NumPyAuthentication & Database: Pyrebase (Auth), Firebase Admin SDK (Firestore)Styling: Dash Bootstrap ComponentsDeployment: Render

🚀 Getting Started
Follow these instructions to set up and run the project on your local machine.PrerequisitesPython 3.9+A Firebase project with Authentication and Firestore enabled.1. Clone the Repositorygit clone https://github.com/VaishnavDevaraj/Intelligent-Automation-for-Data-Analysis.git
cd Intelligent-Automation-for-Data-Analysis
2. Set Up a Virtual EnvironmentIt's highly recommended to use a virtual environment to manage dependencies.Windows:python -m venv venv
.\venv\Scripts\activate
macOS / Linux:python3 -m venv venv
source venv/bin/activate
3. Install Dependenciespip install -r requirements.txt
4. Configure Environment VariablesThis project uses environment variables to keep your secret keys secure.Create a file named .env in the root of the project.Copy the content from the example below and paste it into your .env file.Fill in the values with your own credentials from your Firebase project and a secure secret key..env.example# === Pyrebase Config (Found in your Firebase project settings -> General) ===
PYREBASE_API_KEY="AIzaxxxxxxxxxxxxxxxxxxxxx"
PYREBASE_AUTH_DOMAIN="your-project-id.firebaseapp.com"
PYREBASE_DATABASE_URL="https://your-project-id-default-rtdb.firebaseio.com"
PYREBASE_PROJECT_ID="your-project-id"
PYREBASE_STORAGE_BUCKET="your-project-id.appspot.com"
PYREBASE_MESSAGING_SENDER_ID="1234567890"
PYREBASE_APP_ID="1:1234567890:web:xxxxxxxxxxxxxxxxx"

# === Flask Secret Key (Create a long, random string) ===
FLASK_SECRET_KEY="a-very-long-and-random-string-for-security"

# === Firebase Admin SDK Credentials (Paste the entire content of your service account JSON file here as a single line) ===
FIREBASE_CREDS_JSON={"type": "service_account", "project_id": "...", "private_key_id": "...", "private_key": "-----BEGIN PRIVATE KEY-----\\n...", "client_email": "...", "client_id": "...", "auth_uri": "...", "token_uri": "...", "auth_provider_x509_cert_url": "...", "client_x509_cert_url": "..."}
<<<<<<< HEAD


IMPORTANT: Add your .env file to .gitignore to ensure you never commit your secrets.
5. Run the Application
python complete-fixed-app.py

Open your browser and navigate to http://127.0.0.1:5001.

☁️ Deployment
This application is configured for deployment on Render. You can deploy your own version by following these steps:
Push your code to a public GitHub repository.
Create a new "Web Service" on Render and connect it to your repository.
Use the following settings:
Runtime: Python 3
Build Command: pip install -r requirements.txt
Start Command: gunicorn complete-fixed-app:flask_app
Add all the variables from your .env file to the Environment tab in your Render service settings.

🤝 Contributing
Contributions, issues, and feature requests are welcome! While this is a portfolio project with a proprietary license, I am open to suggestions for improvement.
Feel free to open an Issue to report a bug or suggest a new feature.
If you'd like to propose a code change, please Fork the repository and create a Pull Request. Please note that all contributions will be reviewed and are subject to the project owner's discretion.

📜 License
This project is the intellectual property of Vaishnav Devaraj and Jishnu Shyam.
Copyright (c) 2025 Vaishnav Devaraj and Jishnu Shyam. All Rights Reserved.
This software is proprietary. Permission is hereby granted to view the source code and fork the repository for the sole purpose of submitting contributions (pull requests) or for personal, non-commercial, educational use.
You are prohibited from:
Modifying, merging, publishing, distributing, or sublicensing the software for commercial purposes.
Selling copies of the software.
Using the software for any commercial application without the express written permission of the copyright holder.

👤 Contact
VaishnavDevaraj - GitHub Profile
Project Link: https://github.com/VaishnavDevaraj/Intelligent-Automation-for-Data-Analysis
