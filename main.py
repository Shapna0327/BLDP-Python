import os
from leaf_classification.main import detect_leaf_or_not
from leaf_diseases_prediction.main import detect_leaf_diseases
from flask_cors import CORS
from flask import Flask, request, jsonify

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/predict', methods = ['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    image = request.files['image']
    if image.filename == '':
        return jsonify({"error": "No image selected"}), 400
    
    image_path = os.path.join(UPLOAD_FOLDER, image.filename)
    image.save(image_path)

    frame, classes = detect_leaf_or_not(image_path)
    print(classes)
    
    if not classes:
        print("Check")  
        return jsonify({"error": "Not a banana leaf"}), 400
    else: 
        print("Check1")
        result = detect_leaf_diseases(image_path)
        return jsonify({"result":result}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Dynamically use the port from environment variable
    app.run(debug=False, host="0.0.0.0", port=port)  # Listen on all interfaces
