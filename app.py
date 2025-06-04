from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import numpy as np
import base64
import io
from PIL import Image
import torch
import os
from predict_font import predict_font
from font_list import FONT_LIST

# Import your models
from nets.alexnet import AlexNetClassifier
from nets.henet import HENet
from nets.fontclassifier import FontClassifier
from nets.scae import SCAE

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for models
models = {}
font_classes = []  # You'll need to define your font class names

def load_models():
    """Load all trained models"""
    global models, font_classes
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load font class names
    font_classes = FONT_LIST
    
    num_classes = len(font_classes)
    
    # Load AlexNet model
    try:
        alexnet = AlexNetClassifier(num_classes=200)
        alexnet.load_state_dict(torch.load('checkpoints/alexnet20.pt', weights_only=True, map_location=device))
        models['AlexNet'] = alexnet
        print("AlexNet model loaded successfully")
    except Exception as e:
        print(f"Failed to load AlexNet: {e}")
    
    # Load HENet model
    try:
        henet = HENet(num_classes=200)
        henet.load_state_dict(torch.load('checkpoints/henet_best.pth', map_location=device))
        models['HENet'] = henet
        print("HENet model loaded successfully")
    except Exception as e:
        print(f"Failed to load HENet: {e}")
    
    # Load FontClassifier model
    try:
        pretrained_scae = SCAE()
        pretrained_scae.load_state_dict(torch.load('checkpoints/scae_best.pth', map_location=device))
        fontclassifier = FontClassifier(pretrained_scae, num_classes=200)
        fontclassifier.load_state_dict(torch.load('checkpoints/fontclassifier_best.pth', map_location=device))
        models['FontClassifier'] = fontclassifier
        print("FontClassifier model loaded successfully")
    except Exception as e:
        print(f"Failed to load FontClassifier: {e}")


@app.route('/')
def home():
    return render_template('index.html', models=list(models.keys()))


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Get the cropped image data and selected model
        image_data = data['image']
        model_name = data.get('model', 'AlexNet')
        
        if model_name not in models:
            return jsonify({'error': f'Model {model_name} not available'}), 400
        
        # Decode base64 image
        image_data = image_data.split(',')[1]  # Remove data:image/jpeg;base64, prefix
        image_bytes = base64.b64decode(image_data)
        
        # Convert to numpy array
        image = Image.open(io.BytesIO(image_bytes))
        img_array = np.array(image)
        
        # Convert RGB to BGR for OpenCV compatibility if needed
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Get prediction
        model = models[model_name]
        prediction_probs, predicted_class = predict_font(model, img_array)
        
        # Format results
        results = {
            'predicted_class': int(predicted_class),
            'predicted_font': font_classes[predicted_class] if predicted_class < len(font_classes) else 'Unknown',
            'confidence': float(prediction_probs[predicted_class]),
            'all_probabilities': {
                font_classes[i]: float(prob) for i, prob in enumerate(prediction_probs[:len(font_classes)])
            },
            'model_used': model_name
        }
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    load_models()
    app.run(debug=True, host='0.0.0.0', port=5000)