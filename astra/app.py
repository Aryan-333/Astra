# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import replicate
import requests
from werkzeug.utils import secure_filename
import base64
import time
import uuid

app = Flask(__name__)
CORS(app)

# Configure uploads folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Configure Replicate API
REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN")
if not REPLICATE_API_TOKEN:
    print("Warning: REPLICATE_API_TOKEN environment variable not set")

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Create a unique filename to avoid conflicts
    filename = str(uuid.uuid4()) + secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    # Create a temporary public URL for the image
    # In a production environment, you would upload to S3 or similar service
    # For this demo, we'll encode as base64
    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    
    image_url = f"data:image/jpeg;base64,{encoded_string}"
    
    return jsonify({
        'image_path': file_path,
        'image_url': image_url
    }), 200

@app.route('/edit', methods=['POST'])
def edit_image():
    data = request.json
    
    # Check required fields
    if not data.get('image_url') or not data.get('prompt'):
        return jsonify({'error': 'Missing image URL or prompt'}), 400
    
    try:
        # Prepare input for Replicate API
        input_data = {
            "image": data.get('image_url'),
            "prompt": data.get('prompt'),
        }
        
        # Add optional parameters if provided
        optional_params = [
            'seed', 'scheduler', 'num_outputs', 'guidance_scale',
            'negative_prompt', 'num_inference_steps', 'image_guidance_scale'
        ]
        
        for param in optional_params:
            if param in data and data[param] is not None:
                input_data[param] = data[param]
        
        # Run the model
        output = replicate.run(
            "timothybrooks/instruct-pix2pix:30c1d0b916a6f8efce20493f5d61ee27491ab2a60437c13c588468b9810ec23f",
            input=input_data,
            timeout=1000
        )
        
        # Process output and handle FileOutput objects
        result_images = []
        for item in output:
            if isinstance(item, str):
                result_images.append(item)
            else:
                # Handle FileOutput objects
                try:
                    # For FileOutput objects, we can try to get the URL
                    if hasattr(item, 'url'):
                        result_images.append(item.url)
                    elif hasattr(item, '__str__'):
                        # Try using string representation if it has one
                        result_images.append(str(item))
                    else:
                        print(f"Couldn't process output item: {type(item)}")
                except Exception as e:
                    print(f"Error processing output item: {e}")
        
        # If we still don't have any images, try to access the content differently
        if not result_images and output:
            try:
                # Sometimes the output might be a single object with multiple URLs
                if hasattr(output, 'get'):
                    # If it's dict-like
                    output_data = output.get('output') or output
                    if isinstance(output_data, list):
                        result_images = [url for url in output_data if isinstance(url, str)]
                elif hasattr(output, '__iter__'):
                    # Try to iterate and collect all string values
                    for item in output:
                        if isinstance(item, str) and item.startswith('http'):
                            result_images.append(item)
            except Exception as e:
                print(f"Additional error trying to process output: {e}")
        
        # Log what we're returning for debugging
        print(f"Raw output from Replicate: {output}, Type: {type(output)}")
        print(f"Returning result images: {result_images}")
        
        return jsonify({
            'status': 'success',
            'result_images': result_images
        }), 200
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)