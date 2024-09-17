from flask import Flask, request, render_template, redirect, url_for, send_from_directory,flash, jsonify
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw, ImageFont
import os
import cv2
import numpy as np
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.structures import Instances
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import logging

app = Flask(__name__)
app.secret_key = 'kkl'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results'

# Ensure the upload and result folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Load the pre-trained Detectron2 model
cfg = get_cfg()
cfg.merge_from_file("config.yaml")
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7  # Number of classes in your dataset
cfg.MODEL.WEIGHTS = "model_final.pth"  # Path to your trained model
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set threshold for this model
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

predictor = DefaultPredictor(cfg)

metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
metadata.thing_classes = ["စပါး", "စပါးဂုတ်ကျိုးရောဂါ","အညိုကွက်ရောဂါ", "ကျန်းမာသော စပါးရွက်", "စပါးလောင်မီးပိုး", "စပါးပင်စည်ပုပ်ရောဂါ", "တန့်ဂရိုရောဂါ"]


def predict_image(image_path):
    # Read the image from the file path using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image could not be read.")

    # Run the prediction using the model
    outputs = predictor(image)

    # Get instances and their scores
    instances = outputs["instances"].to("cpu")
    scores = instances.scores.tolist()
    pred_classes = instances.pred_classes.tolist()
    pred_boxes = instances.pred_boxes.tensor.numpy()  # Extract bounding boxes as numpy array

    if scores:
        max_score_index = scores.index(max(scores))
        max_score_class = pred_classes[max_score_index]

        # Filter instances to only include the highest scoring class
        mask = np.array(pred_classes) == max_score_class
        high_score_boxes = pred_boxes[mask]
        high_score_scores = np.array(scores)[mask]
        high_score_classes = np.array(pred_classes)[mask]

        # Convert the image to RGB (Pillow uses RGB format)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert the image to a Pillow Image
        pil_image = Image.fromarray(image_rgb)

        # Initialize ImageDraw
        draw = ImageDraw.Draw(pil_image)

        # Load a font that supports Myanmar script for text (fallback to default font)
        try:
            myanmar_font = ImageFont.truetype("Padauk-Regular.ttf", size=12)  # Default size
        except IOError:
            # Use a fallback font if the Myanmar font is not available
            myanmar_font = ImageFont.load_default()

        # Define the color for the bounding boxes and text backgrounds
        box_color = (0, 255, 0)  # Green color for bounding boxes
        text_bg_color = (0, 0, 0, 150)  # Semi-transparent black background for text
        text_color = (255, 255, 255)  # White color for text

        # Define minimum and maximum font sizes
        min_font_size = 14
        max_font_size = 24

        # Draw text and bounding boxes using the loaded fonts
        for i, box in enumerate(high_score_boxes):
            class_name = metadata.thing_classes[high_score_classes[i]]
            score = high_score_scores[i]

            # Define the combined text
            combined_text = f"{class_name} ({round(score * 100, 2)}%)"
            x1, y1, x2, y2 = box  # Bounding box coordinates

            # Calculate bbox width and height
            bbox_width = abs(int(x2 - x1))
            bbox_height = abs(int(y2 - y1))

            # Calculate font size based on bbox width and height
            font_size = int(min(bbox_width, bbox_height) * 0.3)  # Adjust the multiplier as needed

            # Clamp the font size between min_font_size and max_font_size
            font_size = max(min_font_size, min(font_size, max_font_size))

            # Reload the font with the new size
            myanmar_font = ImageFont.truetype("Padauk-Regular.ttf", size=font_size)

            # Draw the bounding box
            draw.rectangle([int(x1), int(y1), int(x2), int(y2)], outline=box_color, width=2)

            # Calculate text size and background using getbbox
            text_size = draw.textbbox((0, 0), combined_text, font=myanmar_font)
            text_bg_rect = [int(x1), int(y1) - (text_size[3] - text_size[1]), int(x1) + (text_size[2] - text_size[0]), int(y1)]

            # Ensure text background box stays within image boundaries
            text_bg_rect[1] = max(text_bg_rect[1], 0)  # y1 cannot be less than 0
            text_bg_rect[3] = min(text_bg_rect[3], pil_image.height)  # y2 cannot exceed image height
            text_bg_rect[0] = max(text_bg_rect[0], 0)  # x1 cannot be less than 0
            text_bg_rect[2] = min(text_bg_rect[2], pil_image.width)  # x2 cannot exceed image width

            # Adjust the position if it goes out of bounds
            text_x = max(int(x1), 0)
            text_y = max(int(y1) - (text_size[3] - text_size[1]), 0)
            text_x = min(text_x, pil_image.width - text_size[2] + text_size[0])
            text_y = min(text_y, pil_image.height - (text_size[3] - text_size[1]))

            # Adjust background box position if text is adjusted
            text_bg_rect[0] = text_x
            text_bg_rect[1] = text_y
            text_bg_rect[2] = min(text_bg_rect[2], pil_image.width)
            text_bg_rect[3] = min(text_bg_rect[3], pil_image.height)

            # Draw the background rectangle for text with semi-transparency
            draw.rectangle(text_bg_rect, fill=text_bg_color)

            # Draw the combined text on the image
            draw.text((text_x, text_y), combined_text, font=myanmar_font, fill=text_color)

        # Convert back to OpenCV format (BGR)
        result_image = np.array(pil_image)
        result_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
    else:
        result_image = image  # If no instances, return the original image

    # Save the result image
    result_filename = f"result_{os.path.basename(image_path)}"
    result_image_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
    cv2.imwrite(result_image_path, result_image)

    # Extract class names and scores for the highest scoring class
    detections = [{"class": metadata.thing_classes[max_score_class], "score": max(scores)}] if scores else []

    return result_filename, detections




@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            try:
                result_filename, detections = predict_image(file_path)
                return render_template('result.html', result_image=result_filename, detections=detections)
            except Exception as e:
                logging.error(f"Error during prediction: {e}")
                return "An error occurred during image processing."

    return render_template('detect.html')
    
@app.route('/contactus', methods=['GET', 'POST'])
def contactus():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']

        # Save the data to a text file
        with open('contact_messages.txt', 'a') as file:
            file.write(f"Name: {name}\nEmail: {email}\nMessage: {message}\n\n")

        # Return a JSON response
        return jsonify(success=True)

    return render_template('contactus.html')
    
@app.route('/results/<filename>')
def serve_result(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)
@app.route('/index')
def index():
    return render_template('Index.html')

@app.route('/about')
def about():
    return render_template('about.html')


    
@app.route('/blast')
def blast():
    return render_template('စပါးဂုတ်ကျိုးရောဂါ.html')
@app.route('/brownspot')
def brownspot():
    return render_template('အညိုကွက်ရောဂါ.html')
@app.route('/hispa')
def hispa():
    return render_template('စပါးလောင်မီးပိုး.html')
@app.route('/sheathblight')
def sheathblight():
    return render_template('စပါးပင်စည်ပုပ်ရောဂါ.html')
@app.route('/tungro')
def tungro():
    return render_template('တန့်ဂရိုရောဂါ.html')
@app.route('/disease/<disease_name>')
def disease_page(disease_name):
    # Render the template based on the disease name
    return render_template(f'{disease_name}.html')
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app.run(debug=True)
