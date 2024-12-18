## Prasyarat
Sebelum menjalankan aplikasi ini, pastikan Anda sudah menginstal:
- **Python 3.11.0** atau versi yang lebih baru
- **Django 5.1.3** atau versi yang lebih baru

## File yang Terkait
- **.env**: File konfigurasi API Key.
  - Lokasi: `chatbot_app/.env`
- **settings.py**: Konfigurasi aplikasi Django.
  - Lokasi: `chatbot_app/chatbot_app/settings.py`
- **.pth file**: File Python Path untuk memuat model. 
  - Lokasi: `chatbot_app/chatbot/model_files/`

## Instalasi
### 1. **Clone Repository**
Pertama, clone repository ini ke komputer Anda:
git clone https://github.com/pavellsan8/chatbot_django.git

### 2. **Setup Aplikasi**
Kedua, buat virtual environment dengan command:
python -m venv venv -> cd chatbot_app

### 3. **Instalasi Requirements**
Ketiga, jalankan perintah berikut untuk menginstall seluruh depedencies:
pip install -r requirements.txt

### 4. **Clone Repository**
Keempat, run aplikasi dengan command:
python manage.py runserver