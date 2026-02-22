import uvicorn
from fastapi import FastAPI
from dishka import make_async_container
from dishka.integrations.fastapi import setup_dishka
from src.infrastructure.api.controllers import router
from src.infrastructure.api.dependencies import AppProvider

app = FastAPI(title="CatBoost Time Series Trainer")
app.include_router(router)

container = make_async_container(AppProvider())
setup_dishka(container, app)

if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host="127.0.0.1",
        port=8080,
        reload=True
    )