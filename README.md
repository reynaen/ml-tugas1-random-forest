# Simulasi Random Forest (Flask)

Web simulasi perhitungan manual Random Forest untuk studi kasus prediksi pelanggan e-commerce membeli produk.
Menampilkan bootstrap sampling, entropy, gain, split info, gain ratio per node, pohon keputusan, dan mayoritas vote.

## Menjalankan
```
python -m venv .venv
.venv\Scripts\activate        # Windows (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
python app.py
```
Buka http://127.0.0.1:5000

## Deploy
- PythonAnywhere atau Render: arahkan ke `app:app` (start command: `gunicorn app:app`).

## Struktur
- `app.py` : route Flask
- `rf.py` : algoritma (entropy, gain, split info, gain ratio, pohon, svg)
- `templates/index.html`, `static/style.css` : tampilan
