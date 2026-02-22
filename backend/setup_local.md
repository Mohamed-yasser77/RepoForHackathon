# Local Run Guide — Milestone 2

Follow these steps to run the Milestone 2 engine on your local terminal without Docker, using a free cloud Redis database.

## 1. Initial Setup (One-Time)

### A. Create a free Cloud Redis
1. Go to [Upstash.com](https://upstash.com/) and sign up (Free, no credit card).
2. Create a new **Redis Database**.
3. Copy the **Redis URL** (it looks like `rediss://default:PASSWORD@endpoint.upstash.io:port`).

### B. Configure Environment
Update your `e:\Hackathon\.env` file with your Redis URL:
```bash
REDIS_URL=rediss://default:YOUR_PASSWORD@your-db.upstash.io:6379
```

### C. Install Dependencies
Run this in your terminal to ensure all Milestone 2 libraries are ready:
```powershell
pip install arq "redis[hiredis]" httpx pydantic-settings torch transformers sentence-transformers
```

---

## 2. Running the Engine

Since this is an async system, you need to run **two separate processes** in two terminal windows.

### Terminal 1: The API Server
This terminal will handle incoming ticket requests.
```powershell
cd e:\Hackathon
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Terminal 2: The Background Worker
This terminal will process the ML model and send alerts.
```powershell
cd e:\Hackathon
# Note: Use 'python -m arq' if 'arq' command is not in your PATH
python -m arq app.worker.WorkerSettings
```

---

## 3. Testing

Once both are running, use the demo script in a **third terminal**:

```powershell
cd e:\Hackathon
python demo_concurrency.py
```

### Troubleshooting
- **ModuleNotFoundError**: Ensure you are in the same environment where you ran `pip install`.
- **Connection Error**: Double-check that your `REDIS_URL` in `.env` starts with `rediss://` (with two 's' for TLS).
