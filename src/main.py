from fastapi import FastAPI
from src.api.routes import router
from src.retrieval.store import load_index

app = FastAPI(title="queryforge", version="1.0.0")
app.include_router(router)


@app.on_event("startup")
def startup():
    load_index()
