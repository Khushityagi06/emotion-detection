import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras import Input
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Define emotion categories
EMOTIONS = {
    'angry': 0,
    'disgust': 1,
    'fear': 2,
    'happy': 3,
    'sad': 4,
    'surprise': 5,
    'neutral': 6
}

class EmotionDetectionSystem:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.model = self.build_emotion_model()
        
    def load_data(self):
        print("Loading dataset from directories...")
        X = []  # Images
        y = []  # Labels
        
        # Process both train and test directories
        for directory in ['train', 'test']:
            base_path = os.path.join('data', directory)
            
            # Loop through each emotion folder
            for emotion_folder in os.listdir(base_path):
                emotion_path = os.path.join(base_path, emotion_folder)
                if not os.path.isdir(emotion_path):
                    continue
                    
                emotion_label = EMOTIONS[emotion_folder]
                print(f"Processing {directory}/{emotion_folder} images...")
                
                # Loop through each image in the emotion folder
                for image_file in os.listdir(emotion_path):
                    image_path = os.path.join(emotion_path, image_file)
                    try:
                        # Read and preprocess image
                        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                        image = cv2.resize(image, (48, 48))
                        X.append(image)
                        y.append(emotion_label)
                    except Exception as e:
                        print(f"Error processing {image_path}: {e}")
        
        X = np.array(X, dtype='float32')
        y = np.array(y, dtype='int32')
        
        print("Dataset loaded successfully!")
        print(f"Total images loaded: {len(X)}")
        return X, y
    
    def preprocess_data(self, X, y):
        print("Preprocessing data...")
        # Reshape and normalize images
        X = X.reshape(-1, 48, 48, 1)
        X = X / 255.0
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print("Data preprocessing completed!")
        return X_train, X_test, y_train, y_test

    def build_emotion_model(self):
        model = Sequential([
            Input(shape=(48, 48, 1)),
            Conv2D(32, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            Flatten(),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(7, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def train_model(self, X_train, X_test, y_train, y_test):
        print("Starting model training...")
        history = self.model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=15,
            validation_data=(X_test, y_test),
            verbose=1
        )
        print("Model training completed!")
        return history

    def detect_and_predict_emotion(self):
        print("Starting real-time emotion detection. Press 'q' to quit.")
        cap = cv2.VideoCapture(0)
        
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            # For each face
            for (x, y, w, h) in faces:
                # Extract face ROI
                roi_gray = gray[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, (48, 48))
                roi_gray = roi_gray.reshape(1, 48, 48, 1)
                roi_gray = roi_gray.astype('float32') / 255.0
                
                # Predict emotion
                prediction = self.model.predict(roi_gray, verbose=0)
                emotion_label = list(EMOTIONS.keys())[np.argmax(prediction)]
                
                # Draw rectangle and label
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, emotion_label, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            
            # Display the frame
            cv2.imshow('Emotion Detection', frame)
            
            # Break loop with 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Initialize the system
    system = EmotionDetectionSystem()
    
    # Load and preprocess data
    X, y = system.load_data()
    X_train, X_test, y_train, y_test = system.preprocess_data(X, y)
    
    # Train the model
    history = system.train_model(X_train, X_test, y_train, y_test)
    
    # Start real-time detection
    system.detect_and_predict_emotion()