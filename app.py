from flask import Flask, request, jsonify
from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import base64
from flask_cors import CORS
import cv2
import os

app = Flask(__name__)
CORS(app)

# Load YOLO model using Ultralytics
model_path = 'best.pt'
model = YOLO(model_path)

dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
model.predict(source=dummy_image, imgsz=640, conf=0.01, verbose=False)

def compress_image_keep_aspect(image, max_size=1280, quality=70):
    """Compress image while maintaining aspect ratio."""
    original_width, original_height = image.size

    # Determine scaling factor
    scale = min(max_size / original_width, max_size / original_height, 1.0)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    resized_img = image.resize((new_width, new_height), Image.LANCZOS)

    # Save to temp file
    compressed_path = 'compressed_image.jpg'
    resized_img.save(compressed_path, format='JPEG', quality=quality)
    return compressed_path

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    
    try:
        # Save uploaded file temporarily (Ultralytics works better with file paths)
        temp_path = 'temp_image.jpg'
        file.save(temp_path)
        
        # Load original image for dimensions and drawing
        img = Image.open(temp_path).convert('RGB')
        original_width, original_height = img.size

        # Attempt compression
        try:
            img = Image.open(file).convert('RGB')
            temp_path = compress_image_keep_aspect(img)
        except Exception as compression_error:
            print("Compression failed:", compression_error)
            file.stream.seek(0)
            fallback_path = 'original_image.jpg'
            file.save(fallback_path)
            temp_path = fallback_path

        # Load compressed or fallback image
        img = Image.open(temp_path).convert('RGB')
        original_width, original_height = img.size

        
        # Run prediction with Ultralytics
        results = model.predict(source=temp_path, conf=0.55)  # Using same confidence threshold as in Colab
        
        # Prepare to draw on the image
        processed_image = img.copy()
        draw = ImageDraw.Draw(processed_image)
        
        try:
            font = ImageFont.truetype("arial.ttf", 15)
        except IOError:
            font = ImageFont.load_default()
        
        # Initialize results and merchandise count
        result = []
        # Use class names from the model itself, just like in Colab
        merch_counts = {model.names[i]: 0 for i in model.names}
        
        # Process results
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Get confidence
                conf = float(box.conf[0].cpu().numpy())
                
                # Get class ID
                cls_id = int(box.cls[0].cpu().numpy())
                
                # Get label from model.names - just like in Colab
                label = model.names[cls_id]
                
                # Update merchandise count
                merch_counts[label] += 1
                
                # Draw bounding box with different colors per class
                colors = ["red", "green", "blue", "orange", "purple"]
                color = colors[cls_id % len(colors)]
                
                draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=3)
                
                # Draw label with background
                text = f"{label} ({conf:.2f})"
                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                
                draw.rectangle(
                    [(x1, y1 - text_height - 4), (x1 + text_width, y1)],
                    fill=color
                )
                draw.text((x1, y1 - text_height - 2), text, fill="white", font=font)
                
                # Add to results
                result.append({
                    'class_id': cls_id,
                    'label': label,
                    'confidence': conf,
                    'bounding_box': [float(x1), float(y1), float(x2), float(y2)]
                })
        
        # Draw merchandise counts on the image
        # Position the counts on the right side of the image
        y_pos = 20
        for class_name, count in merch_counts.items():
            if count > 0:
                count_text = f"{class_name}: {count}"
                text_bbox = draw.textbbox((0, 0), count_text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                
                # Draw text background
                draw.rectangle(
                    [(original_width - text_width - 10, y_pos - 2), 
                     (original_width - 5, y_pos + text_height + 2)],
                    fill="black"
                )
                
                # Draw text
                draw.text((original_width - text_width - 8, y_pos), count_text, 
                          fill="white", font=font)
                y_pos += text_height + 10
        
        # Convert image to base64
        buffer = io.BytesIO()
        processed_image.save(buffer, format="JPEG")
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        # Clean up
        for path in ['compressed_image.jpg', 'original_image.jpg']:
            try:
                os.remove(path)
            except:
                pass

       
        
        return jsonify({
            'detections': result,
            'merch_counts': merch_counts,
            'image_base64': img_str
        })
        
    except Exception as e:
        print(f"Error processing request: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # app.run(debug=True, host='0.0.0.0', port=5000)
    port = int(os.environ.get("PORT", 5000))  # Default to 5000 if PORT not set
    app.run(debug=False, host='0.0.0.0', port=port)


#flask run --host=0.0.0.0 --port=5000
