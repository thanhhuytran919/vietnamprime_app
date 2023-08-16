from flask import Flask, render_template, request, send_from_directory
from keras.models import load_model
import numpy as np
import librosa.display
import cv2
import matplotlib.pyplot as plt
import time
import soundfile as sf
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

path = "./Spectrograms/"
categories = ['Ca_tru', 'Cheo', 'Hat_chau_van', 'Ho',
              'Nhac_cung_dinh', 'Nhac_tai_tu', 'Quan_ho', 'Xam']

# Load mô hình
model = load_model('./Model/lenet_bs32_10s_vs02.h5')

# Đọc thông tin từ tập tin
music_info = {}
with open('music_info.txt', 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split(':')
        if len(parts) == 2:
            category = parts[0].strip()
            info = parts[1].strip()
            music_info[category] = info

# Đọc liên kết YouTube từ các tập tin trong thư mục youtube_links
youtube_links = {}
for class_name in categories:
    link_file_path = f'./youtube_links/{class_name}.txt'
    with open(link_file_path, 'r', encoding='utf-8') as f:
        links = f.readlines()
        youtube_links[class_name] = [link.strip() for link in links]

# Đọc đường dẫn từ tệp văn bản


def read_info_link(class_name):
    info_path = f'./class_info/{class_name}.txt'
    with open(info_path, 'r') as file:
        link = file.read()
    return link


def read_info_music(class_name):
    info_path = f'./class_info_music/{class_name}.txt'
    with open(info_path, 'r') as file:
        music = file.read()
    return music


def create_spectrogram(audio_file, image_file):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    y, sr = librosa.load(audio_file)
    ms = librosa.feature.melspectrogram(y=y, sr=sr)
    log_ms = librosa.power_to_db(ms, ref=np.max)
    librosa.display.specshow(log_ms, sr=sr)

    fig.savefig(image_file)
    plt.close(fig)


def cut_wav_file(input_file, output_directory, duration):
    # Đọc file âm thanh WAV đầu vào
    data, samplerate = sf.read(input_file)

    # Tính toán số lượng phần cần cắt
    num_parts = int(len(data) / (samplerate * duration)) + 1

    count = 0
    segment_parts = []
    # Cắt file âm thanh thành các phần nhỏ
    for i in range(num_parts):
        start = i * samplerate * duration
        end = min((i + 1) * samplerate * duration, len(data))
        part = data[start:end]

        # Tạo tên tệp đầu ra cho từng phần
        output_file = output_directory + '/test{}.wav'.format(count)
        count += 1

        # Ghi phần âm thanh nhỏ vào tệp đầu ra
        sf.write(output_file, part, samplerate)
        segment_parts.append(output_file)

    return segment_parts


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/get_audio/<int:segment_number>')
def get_audio(segment_number):
    audio_path = f'uploads/audio_cut/test{segment_number}.wav'
    return send_from_directory(app.config['UPLOAD_FOLDER'], audio_path)


@app.route('/upload', methods=['POST'])
def upload():
    if 'audio' not in request.files:
        return "Không tìm thấy tệp âm thanh!"

    audio_file = request.files['audio']
    if audio_file.filename == '':
        return "Chưa chọn tệp âm thanh!"

    # Xóa tất cả các tệp trong thư mục uploads/audio_cut
    cut_directory = 'uploads/audio_cut'
    for file in os.listdir(cut_directory):
        file_path = os.path.join(cut_directory, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Không thể xóa tệp {file_path}: {e}")

    # Xóa tất cả các tệp .png trong thư mục static
    static_directory = 'static'
    for file in os.listdir(static_directory):
        if file.endswith('.png'):
            file_path = os.path.join(static_directory, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Không thể xóa tệp {file_path}: {e}")

    audio_file.save('uploads/audio.wav')
    segment_parts = cut_wav_file('uploads/audio.wav', 'uploads/audio_cut', 10)

    segment_results = []

    for i, segment_part in enumerate(segment_parts):
        create_spectrogram(segment_part, f'static/spectrogram{i}.png')
        img_path = f'static/spectrogram{i}.png'
        img = cv2.imread(img_path)
        img = cv2.resize(img, (128, 128))
        img_array = img / 255.0
        img_batch = np.expand_dims(img_array, axis=0)

        pred = model.predict(img_batch)
        class_name = categories[np.argmax(pred)]
        confidence = pred[0][np.argmax(pred)] * 100

        segment_results.append(
            {'segment_number': i, 'class_name': class_name, 'confidence': confidence, 'img_path': img_path})

    best_segment = max(segment_results, key=lambda x: x['confidence'])
    best_class_name = best_segment['class_name']
    best_confidence = best_segment['confidence']
    best_segment_number = best_segment['segment_number']

    # Đọc thông tin từ tệp văn bản cho class_name
    info_link = read_info_link(best_class_name)
    info_music = read_info_music(best_class_name)
    music_description = music_info.get(best_class_name, "Không có thông tin")

    audio_path = f'uploads/audio.wav?t={int(time.time())}'
    segment_img_paths = f'static/spectrogram{best_segment_number}.png'

    return render_template(
        'index.html',
        category=info_music,
        confidence=best_confidence,
        music_description=music_description,
        audio_path=audio_path,
        info_link=info_link,
        youtube_links=youtube_links.get(best_class_name, []),
        segment_img_paths=segment_img_paths,
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=os.environ.get('PORT', 5000))
