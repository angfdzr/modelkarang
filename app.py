from flask import Flask, request, jsonify, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
from datetime import datetime

# Mengatur level log TensorFlow untuk mengurangi pesan log
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'  # Folder untuk menyimpan gambar yang diunggah
model = load_model('keras_model.h5')  # Memuat model Keras yang sudah dilatih

# Dictionary untuk kelas prediksi
class_dict = {0: 'Karang Bleaching', 1: 'Karang Sehat'}

# Fungsi untuk memprediksi label gambar
def predict_label(img_path):
    # Memuat dan memproses gambar dengan ukuran yang sesuai
    loaded_img = load_img(img_path, target_size=(224, 224))  # Mengubah ukuran gambar menjadi 224x224
    image = img_to_array(loaded_img) / 255.0  # Skala gambar ke [0, 1]
    prediction_image = np.expand_dims(image, axis=0)  # Tambahkan dimensi batch
    prediction = model.predict(prediction_image)  # Melakukan prediksi dengan model
    predicted_class = np.argmax(prediction, axis=1)[0]  # Mengambil kelas dengan nilai probabilitas tertinggi
    return class_dict[predicted_class]  # Mengembalikan kelas yang diprediksi

# Route untuk menampilkan halaman utama
@app.route('/')
def home():
    return "API untuk prediksi kesehatan karang. Gunakan endpoint /predict untuk melakukan prediksi."

# Endpoint untuk menerima unggahan gambar dan mengembalikan prediksi
@app.route('/predict', methods=['POST'])
def predict():
    # Memeriksa apakah ada file yang diunggah
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    # Memeriksa apakah nama file kosong
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    # Nama file unik dengan timestamp untuk menghindari penimpaan
    filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}"
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # Menyimpan gambar yang diunggah
    try:
        file.save(img_path)
        print(f"File saved to {img_path}")
    except Exception as e:
        print(f"Failed to save file: {e}")
        return jsonify({'error': 'Failed to save file'}), 500

    # Melakukan prediksi label untuk gambar yang diunggah
    prediction = predict_label(img_path)
    
    # Mengembalikan hasil prediksi dan path gambar
    result = {
        'prediction': prediction,
        'image_path': f'/uploads/{filename}'  # Path gambar yang dapat diakses
    }
    
    return jsonify(result)

# Endpoint untuk mengakses gambar yang diunggah
@app.route('/uploads/<filename>')
def send_uploaded_image(filename=''):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    # Membuat folder untuk menyimpan gambar jika folder belum ada
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    # Menjalankan aplikasi Flask pada host 0.0.0.0 (menerima koneksi dari mana saja) dan port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
