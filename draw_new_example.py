from flask import Flask, render_template, request, jsonify
import numpy as np
from backpropagation import forwardpass
app = Flask(__name__)

last_pixel_array = None

@app.get("/")
def index():
    return render_template("index.html")


@app.post('/upload')
def upload_image():
    if request.is_json:
        data = request.get_json()
        width = data.get('width')
        height = data.get('height')
        pixels = data.get('pixels')
        if not (isinstance(width, int) and isinstance(height, int) and isinstance(pixels, list)):
            return jsonify({'message': 'Ungültiges JSON-Format'}), 400
        if len(pixels) != width * height:
            return jsonify({'message': 'Pixel-Länge stimmt nicht mit Breite/Höhe überein'}), 400
        arr = np.array(pixels, dtype=np.float32).reshape((height, width))
        global last_pixel_array
        last_pixel_array = arr
        out, *_ = forwardpass(last_pixel_array,0)
        return jsonify({'message': f'{max(out)} mit {out[max(out)]*100}% sicherheit'}), 200

    return jsonify({'message': 'Nur JSON-Uploads mit Pixel-Array werden unterstützt'}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)