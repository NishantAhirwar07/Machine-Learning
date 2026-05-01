import sys
import pandas as pd
import pickle
import os
import math
import random
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QDoubleSpinBox, QPushButton, QFrame
from PyQt5.QtCore import Qt, QTimer, QPoint, pyqtSignal
from PyQt5.QtGui import QPainter, QColor, QPalette, QFont, QBrush, QPen, QLinearGradient
from PyQt5.QtCore import QRect

class RaindropParticle:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.speed = random.uniform(3, 7)
        self.length = random.uniform(10, 20)

class WeatherCanvas(QFrame):
    def __init__(self):
        super().__init__()
        self.is_rainy = False
        self.temperature = 25
        self.raindrops = []
        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(50)
        
    def set_weather(self, is_rainy, temperature):
        self.is_rainy = is_rainy
        self.temperature = temperature
        if is_rainy and not self.raindrops:
            self.create_raindrops()
        elif not is_rainy:
            self.raindrops = []
    
    def create_raindrops(self):
        self.raindrops = [RaindropParticle(random.randint(0, self.width()), random.randint(-50, self.height())) for _ in range(60)]
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw background gradient based on weather
        if self.is_rainy:
            # Rainy day - dark blue with gray
            gradient = QLinearGradient(0, 0, 0, self.height())
            gradient.setColorAt(0, QColor(70, 100, 140))
            gradient.setColorAt(1, QColor(100, 120, 160))
        else:
            # Sunny day - gradient from light blue to yellow-orange
            gradient = QLinearGradient(0, 0, 0, self.height())
            gradient.setColorAt(0, QColor(135, 206, 235))  # Sky blue
            gradient.setColorAt(1, QColor(255, 200, 100))  # Sunset orange
        
        painter.fillRect(self.rect(), QBrush(gradient))
        
        # Draw sun for sunny weather
        if not self.is_rainy:
            sun_x = self.width() - 80
            sun_y = 80
            painter.setBrush(QColor(255, 220, 0))
            painter.drawEllipse(sun_x, sun_y, 60, 60)
            painter.setBrush(QColor(255, 240, 100))
            painter.drawEllipse(sun_x + 5, sun_y + 5, 50, 50)
        
        # Draw raindrops for rainy weather
        if self.is_rainy:
            painter.setPen(QPen(QColor(100, 200, 255), 2))
            for raindrop in self.raindrops:
                painter.drawLine(int(raindrop.x), int(raindrop.y), 
                               int(raindrop.x), int(raindrop.y + raindrop.length))
                raindrop.y += raindrop.speed
                
                if raindrop.y > self.height():
                    raindrop.y = -20
                    raindrop.x = random.randint(0, self.width())

class WeatherForecastApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = None
        self.feature_names = None
        self.load_model()
        self.init_ui()
    
    def load_model(self):
        MODEL_FILE = 'rainfall_prediction_model.pkl'
        if not os.path.exists(MODEL_FILE):
            print(f"Warning: {MODEL_FILE} not found!")
            return
        
        with open(MODEL_FILE, 'rb') as file:
            model_data = pickle.load(file)
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
    
    def init_ui(self):
        self.setWindowTitle('🌦️ Weather Forecasting Model')
        self.setGeometry(100, 100, 900, 700)
        
        # Main widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # Main layout
        main_layout = QHBoxLayout()
        
        # Left side - Canvas
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        
        # Weather display canvas
        self.canvas = WeatherCanvas()
        self.canvas.setMinimumSize(400, 500)
        left_layout.addWidget(self.canvas)
        
        # Temperature display
        temp_label = QLabel()
        temp_label.setFont(QFont('Arial', 24, QFont.Bold))
        temp_label.setAlignment(Qt.AlignCenter)
        temp_label.setText('Temperature: 25°C')
        self.temp_label = temp_label
        left_layout.addWidget(temp_label)
        
        # Prediction result
        result_label = QLabel()
        result_label.setFont(QFont('Arial', 16, QFont.Bold))
        result_label.setAlignment(Qt.AlignCenter)
        result_label.setText('Ready to predict...')
        self.result_label = result_label
        left_layout.addWidget(result_label)
        
        left_widget.setLayout(left_layout)
        main_layout.addWidget(left_widget, 2)
        
        # Right side - Controls
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        
        # Title
        title = QLabel('Weather Inputs')
        title.setFont(QFont('Arial', 16, QFont.Bold))
        right_layout.addWidget(title)
        
        # Input fields
        self.inputs = {}
        defaults = {'Pressure': 1015.9, 'Dew Point': 19.9, 'Humidity': 95.0, 
                   'Cloud': 81.0, 'Sunshine': 0.0, 'Wind Direction': 40.0, 'Wind Speed': 13.7}
        
        for label, default_value in defaults.items():
            label_widget = QLabel(label)
            label_widget.setFont(QFont('Arial', 10))
            right_layout.addWidget(label_widget)
            
            spinbox = QDoubleSpinBox()
            spinbox.setValue(default_value)
            spinbox.setMaximum(10000)
            spinbox.setMinimum(-10000)
            spinbox.setDecimals(1)
            self.inputs[label] = spinbox
            right_layout.addWidget(spinbox)
        
        # Temperature slider (for demo)
        right_layout.addSpacing(20)
        temp_title = QLabel('Temperature (°C)')
        temp_title.setFont(QFont('Arial', 10))
        right_layout.addWidget(temp_title)
        
        temp_spinbox = QDoubleSpinBox()
        temp_spinbox.setValue(25)
        temp_spinbox.setMaximum(50)
        temp_spinbox.setMinimum(-50)
        temp_spinbox.setDecimals(1)
        self.temp_input = temp_spinbox
        right_layout.addWidget(temp_spinbox)
        
        right_layout.addSpacing(20)
        
        # Predict button
        predict_btn = QPushButton('🔮 Predict Rainfall')
        predict_btn.setFont(QFont('Arial', 12, QFont.Bold))
        predict_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        predict_btn.clicked.connect(self.predict_rainfall)
        right_layout.addWidget(predict_btn)
        
        # Reset button
        reset_btn = QPushButton('🔄 Reset')
        reset_btn.setFont(QFont('Arial', 10))
        reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                padding: 8px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
        """)
        reset_btn.clicked.connect(self.reset_values)
        right_layout.addWidget(reset_btn)
        
        right_layout.addStretch()
        right_widget.setLayout(right_layout)
        right_widget.setMaximumWidth(300)
        main_layout.addWidget(right_widget, 1)
        
        main_widget.setLayout(main_layout)
        
        # Apply stylesheet
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QLabel {
                color: #333;
            }
            QDoubleSpinBox {
                padding: 5px;
                border: 1px solid #ddd;
                border-radius: 3px;
            }
        """)
    
    def predict_rainfall(self):
        if not self.model:
            self.result_label.setText('❌ Model not loaded!')
            return
        
        try:
            # Collect input values
            values = [self.inputs[label].value() for label in self.feature_names]
            temperature = self.temp_input.value()
            
            # Predict
            input_data = pd.DataFrame([values], columns=self.feature_names)
            prediction = self.model.predict(input_data)[0]
            
            # Update canvas and display
            is_rainy = prediction == 1
            self.canvas.set_weather(is_rainy, temperature)
            self.temp_label.setText(f'🌡️ Temperature: {temperature}°C')
            
            if is_rainy:
                self.result_label.setText('🌧️ RAINFALL EXPECTED')
                self.result_label.setStyleSheet('color: #0066cc; font-weight: bold;')
            else:
                self.result_label.setText('☀️ NO RAINFALL EXPECTED')
                self.result_label.setStyleSheet('color: #ff9900; font-weight: bold;')
        
        except Exception as e:
            self.result_label.setText(f'❌ Error: {str(e)}')
    
    def reset_values(self):
        defaults = {'Pressure': 1015.9, 'Dew Point': 19.9, 'Humidity': 95.0, 
                   'Cloud': 81.0, 'Sunshine': 0.0, 'Wind Direction': 40.0, 'Wind Speed': 13.7}
        for label, value in defaults.items():
            self.inputs[label].setValue(value)
        self.temp_input.setValue(25)
        self.canvas.set_weather(False, 25)
        self.temp_label.setText('🌡️ Temperature: 25°C')
        self.result_label.setText('Ready to predict...')
        self.result_label.setStyleSheet('color: #333;')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    weather_app = WeatherForecastApp()
    weather_app.show()
    sys.exit(app.exec_())
