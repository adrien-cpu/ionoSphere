import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Welcome to the Routing Node"}

class RouteRequest(BaseModel):
    destination: str

@app.post("/route")
async def route(request: RouteRequest):
    # Logique de routage ici
    return {"message": f"Routing to {request.destination}"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
