import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from facenet_pytorch import MTCNN

# Paths
model_save_path = "efficientnet_b2_emotion_model.pth"

# Emotion categories (Updated for 7 classes)
emotion_classes = ["Anger", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Load the fine-tuned emotion model (RGB version)
def load_model(model_path):
    print("Loading fine-tuned EfficientNet-B2 model...")

    model = models.efficientnet_b2(pretrained=False)

    # Modify the first layer to accept RGB images (3 channels)
    model.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)

    # Modify classifier head to match 7 classes
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),  
        nn.Linear(model.classifier[1].in_features, len(emotion_classes)),
    )

    # Load the trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print("✅ Model Loaded Successfully!")
    return model

# Load the trained model
model = load_model(model_save_path)

# ✅ Initialize MTCNN for face detection
mtcnn = MTCNN(keep_all=True, device=device)

# ✅ Image Preprocessing Function (RGB Mode)
def preprocess_face(face_image):
    transform = transforms.Compose([
        transforms.Resize((260, 260)),  # Updated to match training size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    pil_image = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
    processed_image = transform(pil_image)
    processed_image = processed_image.unsqueeze(0)
    return processed_image.to(device)

# ✅ Emotion Prediction Function
def predict_emotion(face_image):
    processed_image = preprocess_face(face_image)

    # Perform prediction
    with torch.no_grad():
        outputs = model(processed_image)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class_idx = torch.argmax(probabilities, dim=1).item()
        predicted_class = emotion_classes[predicted_class_idx]
        confidence = probabilities[0, predicted_class_idx].item()

    return predicted_class, confidence

# ✅ Quit Button Click Handling
quit_program = False  # Global flag to exit program
quit_button_coords = (10, 10, 100, 50)  # (x1, y1, x2, y2)

def check_quit_click(event, x, y, flags, param):
    global quit_program
    if event == cv2.EVENT_LBUTTONDOWN:
        if quit_button_coords[0] <= x <= quit_button_coords[2] and quit_button_coords[1] <= y <= quit_button_coords[3]:
            quit_program = True  # Set flag to exit program

# ✅ Live Face Detection & Emotion Prediction
def run_live_emotion_detection():
    global quit_program

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Cannot access webcam.")
        return

    print("Press 'q' or click 'Quit' button to exit the application.")

    # Set mouse callback
    cv2.namedWindow("Live Emotion Detection")
    cv2.setMouseCallback("Live Emotion Detection", check_quit_click)

    while not quit_program:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Cannot read frame from webcam.")
            break

        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        boxes, _ = mtcnn.detect(rgb_frame)

        # Process detected faces
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = [int(coord) for coord in box]
                face_image = frame[y1:y2, x1:x2]

                # Skip prediction if the face is too small
                if face_image.shape[0] > 0 and face_image.shape[1] > 0:
                    emotion, confidence = predict_emotion(face_image)
                    print(f"🎭 Emotion Detected: {emotion} (Confidence: {confidence:.2f})")

                    # Draw the rectangle around the face
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Display the emotion label
                    label = f"{emotion} ({confidence:.2f})"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "No face detected", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Draw the quit button
        cv2.rectangle(frame, (quit_button_coords[0], quit_button_coords[1]), 
                      (quit_button_coords[2], quit_button_coords[3]), (0, 0, 255), -1)
        cv2.putText(frame, "Quit", (quit_button_coords[0] + 10, quit_button_coords[1] + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Display the frame
        cv2.imshow("Live Emotion Detection", frame)

        # Check for quit conditions
        if cv2.waitKey(1) & 0xFF == ord('q') or quit_program:
            print("🛑 Quitting Program...")
            break

    cap.release()
    cv2.destroyAllWindows()

# ✅ Run the Emotion Detection
if __name__ == "__main__":
    run_live_emotion_detection()