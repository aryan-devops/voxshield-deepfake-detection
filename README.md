# VoxShield Backend

This is the production-ready Python backend for VoxShield, an AI-assisted synthetic/AI-generated voice detection application.

It is built with **FastAPI** and uses **PyTorch** + **Transformers** to run the trained VoxShield model on overlapping sliding windows of audio data, aggregating the predictions without relying on LLMs for detection logic.

Most importantly, this backend implements a strict **Privacy-First Architecture**.

## Privacy Architecture

1. **Audio Upload**: Received via `multipart/form-data`.
2. **Temporary Processing**: Audio is spooled to a secure temporary file.
3. **Audio Decoding & Windows**: Sliced into 3.0s overlapping windows.
4. **VoxShield Inference**: Model evaluates each window.
5. **Result Generation**: Probabilities are aggregated.
6. **Temporary Audio Deleted**: The temp file is securely unlinked. (Guaranteed via `finally` blocks).
7. **JSON Response**: Client receives structured JSON.

There is NO permanent audio storage, and NO audio is retained for training.

## Prerequisites

- Python 3.10+
- The trained VoxShield model weights must be placed in a directory (e.g., `models/voxshield/`).

### Expected Model Files
The backend expects standard Hugging Face exports:
- `pytorch_model.bin` (or `.safetensors`)
- `config.json`
- `preprocessor_config.json`

## Local Setup

1. **Install Dependencies**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Environment Variables**
   Create a `.env` file (you can copy `.env.example`):
   ```
   VOXSHIELD_MODEL_DIR=./models/voxshield
   MAX_AUDIO_DURATION=30
   WINDOW_SIZE=3
   WINDOW_OVERLAP=0.5
   MAX_FILE_SIZE_MB=25
   FRONTEND_URL=http://localhost:3000
   ```

3. **Run Server**
   ```bash
   uvicorn app.main:app --reload
   ```

## Deployment

### Docker
A `Dockerfile` is provided for standard containerized deployment.
```bash
docker build -t voxshield-backend .
docker run -p 8000:8000 -e VOXSHIELD_MODEL_DIR=/app/models/voxshield voxshield-backend
```

### Railway Deployment
1. Connect your repository to Railway.
2. Set the Root Directory to `backend/`.
3. Railway will automatically detect the `requirements.txt` and start the FastAPI app using Uvicorn on the `$PORT`.
4. Define your Environment Variables in the Railway dashboard (make sure to set `FRONTEND_URL`).
5. Ensure your model files are accessible (either committed via Git-LFS or downloaded during a custom build step).

### Render Deployment
1. Connect to Render as a Web Service.
2. Build Command: `pip install -r requirements.txt`
3. Start Command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

### Vercel Integration (Frontend)
The Next.js frontend running on Vercel simply needs to be pointed to the deployed backend URL via an environment variable.
In your Vercel project settings, set:
`NEXT_PUBLIC_API_URL=https://your-deployed-backend-url.com`

## API Endpoints

- `GET /api/health` - Basic health check
- `GET /api/system/status` - Operational constraints
- `GET /api/model/status` - Loaded model info and device
- `POST /api/analyze` - The core inference endpoint.

### Example cURL

```bash
curl -X POST \
  http://localhost:8000/api/analyze \
  -F "file=@test_audio.wav"
```

## Testing

Run the pytest suite to verify logic, overlapping algorithms, limit enforcements, and privacy cleanup:
```bash
pytest
```
