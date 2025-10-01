from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory
from PIL import Image
import io, base64, os, time
from datetime import datetime
import random
import fish_type   # updated fish type prediction module
import fish_fresh  # updated freshness prediction module

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# --- Save uploaded images in folder next to app.py ---
UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Mock count & weight ---
def count_and_estimate_weight(img_path):
    count = random.randint(1, 10)
    avg_weight = random.uniform(0.2, 2.0)
    total_weight = round(count * avg_weight, 2)
    return count, total_weight

# --- Mock geotag ---
def get_geotag():
    lat = round(random.uniform(-90, 90), 5)
    lon = round(random.uniform(-180, 180), 5)
    return lat, lon

# --- Overall quality scoring ---
def overall_quality(freshness, count, total_weight):
    score = 0
    if freshness == 'fresh':
        score += 50
    elif freshness == 'stale':
        score += 25
    score += 20 if count > 5 else 10
    score += 30 if total_weight > 5 else 15

    if score >= 70:
        return "GOOD ✅"
    elif score >= 40:
        return "AVERAGE ⚠️"
    else:
        return "POOR ❌"

# ------------------------------
# Routes
# ------------------------------

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/preview', methods=['POST'])
def preview():
    img_data = request.form['image']
    img_str = img_data.split(',')[1]
    img_bytes = io.BytesIO(base64.b64decode(img_str))

    # Save uploaded image with unique timestamp
    timestamp = int(time.time())
    filename = f"img_{timestamp}.png"
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    img = Image.open(img_bytes).convert("RGB")
    img.save(save_path)

    session['img_file'] = filename
    return render_template('preview.html', filename=filename)


@app.route('/analyze', methods=['GET'])
def analyze():
    img_file = session.get('img_file')
    if not img_file:
        return redirect(url_for('index'))

    img_path = os.path.join(app.config['UPLOAD_FOLDER'], img_file)

    # --- Fish type prediction ---
    predicted_class, classes = fish_type.predict(img_path)
    fish_name = classes[predicted_class]

    # --- Freshness prediction ---
    predicted_class_fresh, classes_fresh = fish_fresh.predict(img_path)
    freshness_class = classes_fresh[predicted_class_fresh]
    freshness = "fresh" if freshness_class=="C1" else "stale" if freshness_class=="C2" else "spoiled"

    # --- Count & weight ---
    count, total_weight = count_and_estimate_weight(img_path)
    lat, lon = get_geotag()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    quality = overall_quality(freshness, count, total_weight)

    return render_template(
        'result.html',
        fish_name=fish_name,
        freshness=freshness,
        count=count,
        total_weight=total_weight,
        lat=lat,
        lon=lon,
        timestamp=timestamp,
        quality=quality,
        img_path=f"uploads/{img_file}"
    )


# Serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True)
