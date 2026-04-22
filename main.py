from fastapi import FastAPI

app = FastAPI(
    title="Grimm-OSS AI Trading System",
    version="0.1.0",
    description="An open-source AI-powered automated trading platform."
)

@app.get("/")
def root():
    return {"message": "Grimm-OSS AI Trading System is running."}