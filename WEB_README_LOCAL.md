# Web AI (Local + Vercel) - Python 3.12

Dokumen ini untuk jalankan AI versi web modern + model TensorFlow.js secara lokal.

## Fitur saat ini

- Klasifikasi penyakit kulit kucing (TFJS di browser)
- Validasi input gambar (deteksi kucing + cek texture close-up)
- Heatmap area penting (Grad-CAM style)
- AI explanation via endpoint backend aman (`/api/explain`)
- Integrasi Google Maps dokter hewan

## 1) Buat venv Python 3.12

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements_web_312.txt
```

## 2) Konversi model .pth ke TensorFlow.js

```powershell
.\.venv312\Scripts\python.exe convert_to_tfjs.py
```

Output model web:
- `web/model/model.json`
- `web/model/group1-shard*.bin`
- `web/model/meta.json`

## 3) Jalankan web app lokal (tanpa API explanation)

```powershell
.\.venv312\Scripts\python.exe run_web.py
```

Buka:
- `http://127.0.0.1:8080`

Catatan: di mode ini `/api/explain` tidak aktif, jadi aplikasi otomatis fallback ke explanation lokal.

## 4) Jalankan mode Vercel (dengan backend API)

```powershell
npm i -g vercel
vercel dev
```

Set secret di Vercel project:
- `OPENROUTER_API_KEY`

Lalu buka URL yang diberikan `vercel dev`.

## Struktur file penting

- `convert_to_tfjs.py` : pipeline konversi `.pth -> onnx -> saved_model -> tfjs`
- `web/index.html` : UI web
- `web/styles.css` : style modern dan responsif
- `web/app.js` : inferensi + validasi + heatmap + maps + call API explanation
- `api/explain.js` : endpoint server-side untuk OpenRouter
- `vercel.json` : rewrite route frontend + model
- `run_web.py` : server lokal sederhana
