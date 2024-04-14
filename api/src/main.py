# основной файл - точка входа в api
from fastapi import FastAPI
from ai.routers import ai_router

app = FastAPI(
    title="Lazy Learn Solution"
)

# добавляем роутер работы с файлами
app.include_router(ai_router,
                   prefix="/ai",
                   tags=["ai models"])
