# ❤️ Heart Disease Risk Prediction System

An **IoT + Machine Learning based Heart Disease Risk Prediction System** that collects health-related data using an **ESP32 and MAX30102 sensor**, stores real-time data using **Firebase Realtime Database**, and uses a **Random Forest Machine Learning model** through a **FastAPI backend** to predict heart disease risk.

The project combines:

- 🔌 IoT
- 🤖 Machine Learning
- 🧠 Random Forest
- 📡 ESP32
- ❤️ MAX30102
- 🔥 Firebase Realtime Database
- ⚡ FastAPI
- 🐍 Python

---

## 🚀 Project Overview

The system is designed to collect health-related parameters and use a trained Machine Learning model to identify whether the given input indicates a potential heart disease risk.

The overall system works as:

```text
MAX30102 Sensor
       ↓
     ESP32
       ↓
Firebase Realtime Database
       ↓
FastAPI Backend
       ↓
Random Forest ML Model
       ↓
Heart Disease Prediction
```

The system can be extended to monitor patient health data in real time and provide ML-based risk predictions.

---

# 🏗️ System Architecture

```text
                    ┌─────────────────────┐
                    │    MAX30102 Sensor  │
                    │                     │
                    │ Heart-related Data  │
                    └──────────┬──────────┘
                               │
                               ↓
                    ┌─────────────────────┐
                    │        ESP32        │
                    │                     │
                    │ Sensor Data Reading │
                    └──────────┬──────────┘
                               │
                               ↓
              ┌────────────────────────────────┐
              │ Firebase Realtime Database     │
              │                                │
              │ Real-time Data Storage         │
              └───────────────┬────────────────┘
                              │
                              ↓
                    ┌─────────────────────┐
                    │    FastAPI Server   │
                    │                     │
                    │ REST API            │
                    └──────────┬──────────┘
                               │
                               ↓
                    ┌─────────────────────┐
                    │ Random Forest Model │
                    │                     │
                    │ ML Prediction       │
                    └──────────┬──────────┘
                               │
                               ↓
                    ┌─────────────────────┐
                    │ Prediction Result   │
                    │                     │
                    │ Disease / No Disease│
                    └─────────────────────┘
```

---

# 🔧 Hardware Components

| Component | Purpose |
|----------|---------|
| ESP32 | Microcontroller and Wi-Fi connectivity |
| MAX30102 | Heart-rate / pulse and SpO₂ sensing |
| Jumper Wires | Hardware connections |
| Breadboard | Prototype circuit |
| Power Supply | Powers the ESP32 and sensors |

---

# ❤️ MAX30102 Sensor

The **MAX30102** is used for collecting heart-related physiological readings.

It can be used for:

- Heart Rate
- Pulse measurement
- SpO₂ measurement

The sensor communicates with the ESP32, which processes the readings and can send them to Firebase through Wi-Fi.

```text
MAX30102
   │
   │ I2C
   ↓
 ESP32
   │
   │ Wi-Fi
   ↓
 Firebase
```

---

# 🔥 Firebase Realtime Database

Firebase Realtime Database is used to store and synchronize the device data in real time.

It provides a cloud-based database that can be accessed by the application and backend.

### Example Firebase Data

```json
{
  "patient": {
    "patient_id": "P001",
    "heartRate": 78,
    "spo2": 97,
    "timestamp": "2026-01-01T10:30:00"
  }
}
```

The ESP32 can send sensor readings to Firebase through Wi-Fi.

### Data Flow

```text
MAX30102
    ↓
  ESP32
    ↓
 Wi-Fi
    ↓
Firebase Realtime Database
```

Firebase can be used for:

- Real-time sensor data storage
- Patient data synchronization
- Cloud-based data access
- Storing historical readings
- Connecting IoT devices with software applications

> **Note:** The current FastAPI prediction endpoint accepts the ML features directly as request data. Firebase acts as the real-time data storage layer and can be integrated with the backend to fetch patient/device data before prediction.

---

# 🤖 Machine Learning Model

The project uses a **Random Forest Classifier** for prediction.

The model is trained using a heart disease dataset containing **20,000+ records**.

The trained model is saved using Python's `pickle` module.

```python
heart_model_one.sav
```

The FastAPI application loads this trained model and uses it to make predictions.

---

# 📊 Machine Learning Features

The model uses the following input features:

| Feature | Description |
|--------|-------------|
| `male` | Gender |
| `age` | Age of patient |
| `currentSmoker` | Current smoking status |
| `BPMeds` | Blood pressure medication |
| `prevalentStroke` | Previous stroke |
| `prevalentHyp` | Prevalent hypertension |
| `diabetes` | Diabetes status |
| `totChol` | Total cholesterol |
| `sysBP` | Systolic blood pressure |
| `diaBP` | Diastolic blood pressure |
| `BMI` | Body Mass Index |
| `heartRate` | Heart rate |
| `glucose` | Blood glucose level |

---

# 🧠 Random Forest

Random Forest is an ensemble Machine Learning algorithm that combines multiple decision trees to make a prediction.

Basic working:

```text
Input Patient Data
       ↓
Decision Tree 1
Decision Tree 2
Decision Tree 3
       .
       .
       .
Decision Tree N
       ↓
Majority Voting
       ↓
Final Prediction
```

The trained model is loaded by FastAPI:

```python
heart_APIModel = pickle.load(
    open('heart_model_one.sav', 'rb')
)
```

---

# ⚡ FastAPI Backend

FastAPI is used to create the REST API for the Machine Learning model.

The API receives patient information and sends it to the trained Random Forest model.

### API Endpoint

```text
POST /heart_prede
```

---

# 📥 Request Format

The API expects the following JSON structure:

```json
{
    "male": 1,
    "age": 45,
    "currentSmoker": 0,
    "BPMeds": 0,
    "prevalentStroke": 0,
    "prevalentHyp": 1,
    "diabetes": 0,
    "totChol": 220,
    "sysBP": 130,
    "diaBP": 85,
    "BMI": 24.5,
    "heartRate": 78,
    "glucose": 95
}
```

---

# 📤 Prediction Response

If the model predicts class `0`:

```text
No disease
```

If the model predicts class `1`:

```text
Disease
```

---

# 🐍 FastAPI Code

```python
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import json

app = FastAPI()


class model_input(BaseModel):
    male: int
    age: int
    currentSmoker: int
    BPMeds: int
    prevalentStroke: int
    prevalentHyp: int
    diabetes: int
    totChol: float
    sysBP: float
    diaBP: float
    BMI: float
    heartRate: int
    glucose: int


heart_APIModel = pickle.load(
    open('heart_model_one.sav', 'rb')
)


@app.post('/heart_prede')
def heart_prede(input_parameters: model_input):

    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)

    gend = input_dictionary['male']
    age_of = input_dictionary['age']
    somker = input_dictionary['currentSmoker']
    bpeds = input_dictionary['BPMeds']
    pstorke = input_dictionary['prevalentStroke']
    phyp = input_dictionary['prevalentHyp']
    dia = input_dictionary['diabetes']
    chol = input_dictionary['totChol']
    sBP = input_dictionary['sysBP']
    dBP = input_dictionary['diaBP']
    bmi = input_dictionary['BMI']
    BPM = input_dictionary['heartRate']
    glu = input_dictionary['glucose']

    input_list = [
        gend,
        age_of,
        somker,
        bpeds,
        pstorke,
        phyp,
        dia,
        chol,
        sBP,
        dBP,
        bmi,
        BPM,
        glu
    ]

    prediction = heart_APIModel.predict([input_list])

    if prediction[0] == 0:
        return "No disease"
    else:
        return "Disease"
```

---

# 🛠️ Technologies Used

## Hardware

- ESP32
- MAX30102
- Jumper Wires
- Breadboard

## Backend

- Python
- FastAPI
- Pydantic

## Machine Learning

- Scikit-learn
- Random Forest
- Pandas
- NumPy

## Database

- Firebase Realtime Database

## Development Tools

- VS Code
- Jupyter Notebook
- Arduino IDE
- Git
- GitHub

---

# 📦 Installation

## 1. Clone Repository

```bash
git clone https://github.com/VishalSharmaCode/HeartSync.git
```

Move into the project directory:

```bash
cd HeartSync
```

---

# 🐍 2. Create Virtual Environment

Windows:

```bash
python -m venv venv
```

Activate:

```bash
venv\Scripts\activate
```

Linux / macOS:

```bash
python3 -m venv venv
```

```bash
source venv/bin/activate
```

---

# 📚 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```text
fastapi
uvicorn
pydantic
numpy
pandas
scikit-learn
```

---

# ▶️ 4. Run FastAPI Server

Run:

```bash
uvicorn app:app --reload
```

The server will start at:

```text
http://127.0.0.1:8000
```

---

# 📖 5. Open API Documentation

FastAPI automatically provides Swagger UI.

Open:

```text
http://127.0.0.1:8000/docs
```

You can test the prediction API directly from the Swagger interface.

---

# 🔌 Hardware Integration

The ESP32 is connected to the MAX30102 sensor.

Basic hardware flow:

```text
MAX30102
     ↓
    I2C
     ↓
   ESP32
     ↓
   Wi-Fi
     ↓
  Firebase
```

The ESP32 reads the sensor data and sends the readings to Firebase Realtime Database.

Example conceptual ESP32 data:

```json
{
    "heartRate": 78,
    "spo2": 97
}
```

---

# 🔄 Complete Working Flow

The complete project can be represented as:

```text
              ┌──────────────────┐
              │    MAX30102      │
              │     Sensor       │
              └────────┬─────────┘
                       │
                       ↓
              ┌──────────────────┐
              │      ESP32       │
              │                  │
              │ Read Sensor Data │
              └────────┬─────────┘
                       │
                       ↓
              ┌──────────────────┐
              │     Firebase     │
              │ Realtime Database│
              └────────┬─────────┘
                       │
                       ↓
              ┌──────────────────┐
              │     FastAPI      │
              │      Backend     │
              └────────┬─────────┘
                       │
                       ↓
              ┌──────────────────┐
              │  Random Forest   │
              │   ML Prediction  │
              └────────┬─────────┘
                       │
                       ↓
              ┌──────────────────┐
              │ Prediction Result│
              │                  │
              │ Disease /        │
              │ No Disease       │
              └──────────────────┘
```

---

# ✨ Key Features

- ❤️ Heart-related health monitoring
- 📡 ESP32-based IoT system
- 🔬 MAX30102 sensor integration
- 🔥 Firebase Realtime Database
- 🤖 Random Forest Machine Learning model
- ⚡ FastAPI REST API
- 📊 20,000+ dataset records
- 🐍 Python-based backend
- ☁️ Real-time cloud data storage
- 🔄 IoT + ML integration

---

# 🔮 Future Improvements

Possible future improvements include:

- Real-time dashboard
- Mobile application
- Patient authentication
- Multiple patient profiles
- Firebase-to-FastAPI automated prediction pipeline
- Real-time prediction from sensor data
- Historical health-data visualization
- Patient health alerts
- Improved ML model performance
- Multiple ML model comparison
- Cloud deployment
- Docker support
- Authentication and authorization
- Secure API communication

---

# 📱 Future Application Architecture

The system can later be extended into a complete mobile/web application:

```text
                    ┌───────────────┐
                    │    ESP32      │
                    └───────┬───────┘
                            │
                            ↓
                    ┌───────────────┐
                    │    Firebase   │
                    └───────┬───────┘
                            │
              ┌─────────────┴─────────────┐
              │                           │
              ↓                           ↓
       ┌──────────────┐           ┌──────────────┐
       │   FastAPI    │           │   Dashboard  │
       │   Backend    │           │ / Mobile App │
       └──────┬───────┘           └──────────────┘
              │
              ↓
       ┌──────────────┐
       │ Random Forest│
       │ ML Model     │
       └──────┬───────┘
              │
              ↓
       ┌──────────────┐
       │   Prediction │
       └──────────────┘
```

---

# 🔐 Security Considerations

For a production deployment, the following should be implemented:

- Firebase Authentication
- Firebase Security Rules
- HTTPS
- API authentication
- Input validation
- Secure storage of credentials
- Environment variables for secrets
- Proper access control
- Rate limiting

Never commit Firebase credentials, API keys, passwords, or private configuration files directly to GitHub.

Use environment variables or secure configuration management instead.

---

# ⚠️ Disclaimer

This project is intended for **educational and research purposes only**.

The Machine Learning prediction should **not be considered a medical diagnosis** or a replacement for professional medical advice.

The prediction generated by the model depends on the quality of the input data and the trained dataset/model.

For real-world medical applications, the system would require appropriate clinical validation, regulatory compliance, security, and professional medical oversight.

---

# 👨‍💻 Project Information

**Project:** Heart Disease Risk Prediction System

**Domain:** IoT + Machine Learning + Healthcare

**Hardware:** ESP32 + MAX30102

**Database:** Firebase Realtime Database

**Backend:** FastAPI

**Machine Learning:** Random Forest

**Language:** Python

---

# ⭐ If You Like This Project

If you find this project useful, consider giving the repository a ⭐ on GitHub.

```text
IoT + ESP32 + MAX30102
          ↓
Firebase Realtime Database
          ↓
       FastAPI
          ↓
   Random Forest ML
          ↓
Heart Disease Risk Prediction
```
