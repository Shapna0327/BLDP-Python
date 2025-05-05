import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

model = load_model("leaf_diseases_prediction/banana_disease_CNN_MODEL.h5")

class_mapping = {
    0: "Banana Black Sigatoka Disease",
    1: "Banana Bract Mosaic Virus Disease",
    2: "Banana Healthy Leaf",
    3: "Banana Insect Pest Disease",
    4: "Banana Moko Disease",
    5: "Banana Panama Disease",
    6: "Banana Yellow Sigatoka Disease"
}

disease_info = {
    "Banana Black Sigatoka Disease": {
        "description": "A fungal disease causing dark streaks on leaves.",
        "symptoms": "Black streaks, yellowing of leaves, reduced yield.",
        "solution": "Use fungicides, remove infected leaves, ensure proper spacing."
    },
    "Banana Bract Mosaic Virus Disease": {
        "description": "A viral disease that causes mosaic patterns on leaves.",
        "symptoms": "Mosaic leaf pattern, twisted leaves, poor fruit development.",
        "solution": "Use virus-free planting material, control insect vectors."
    },
    "Banana Healthy Leaf": {
        "description": "The leaf is healthy with no signs of disease.",
        "symptoms": "Green and intact leaf structure.",
        "solution": "Maintain good agricultural practices."
    },
    "Banana Insect Pest Disease": {
        "description": "Damage caused by insects such as banana weevils and aphids.",
        "symptoms": "Holes in leaves, damaged fruit, insect presence.",
        "solution": "Use biological control methods and pesticides."
    },
    "Banana Moko Disease": {
        "description": "A bacterial disease that causes wilting and fruit rot.",
        "symptoms": "Wilting of leaves, fruit discoloration, bacterial ooze.",
        "solution": "Destroy infected plants, avoid cross-contamination."
    },
    "Banana Panama Disease": {
        "description": "A soil-borne fungal disease causing plant wilt.",
        "symptoms": "Yellowing of leaves, stunted growth, root rot.",
        "solution": "Use resistant varieties, improve drainage."
    },
    "Banana Yellow Sigatoka Disease": {
        "description": "A fungal disease causing yellow spots on leaves.",
        "symptoms": "Yellow streaks, premature leaf drop.",
        "solution": "Apply fungicides, improve aeration around plants."
    }
}

def preprocess_image(image_path, target_size):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  
    return img_array

def predict_class(image_path):
    input_size = (128, 128)
    img = preprocess_image(image_path, input_size)
    prediction = model.predict(img)
    pred_class = np.argmax(prediction, axis=1)[0]
    return class_mapping.get(pred_class, "Unknown")

def detect_leaf_diseases(image):
    disease_name = predict_class(image)
    disease_details = disease_info.get(disease_name, {
        "description": "Information not available",
        "symptoms": "Information not available",
        "solution": "Information not available"
    })
    result = {
        "disease": disease_name,
        "description": disease_details["description"],
        "symptoms": disease_details["symptoms"],
        "solution": disease_details["solution"]
    }

    return result