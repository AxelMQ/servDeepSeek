from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import requests
import os
from dotenv import load_dotenv

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

# Crear instancia de FastAPI
app = FastAPI()

# Configurar la API Key de DeepSeek
OPENROUTER_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("--> No se encontro la API Key de OpenRouter en .env")

# Configurar el cliente de OpenAI para usar la API de DeepSeek
client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)


# Definir un modelo Pydantic para la solicitud
class Pregunta(BaseModel):
    text: str  # Campo para la pregunta del usuario

# Ruta para recibir preguntas y enviarlas a DeepSeek
@app.post("/preguntar")
async def preguntar(pregunta: Pregunta):
    try:
        # Depuración: Verifica que la API key se cargó correctamente
        print("--> API Key cargada:", OPENROUTER_API_KEY)
        
        response = client.chat.completions.create(
            model="deepseek/deepseek-r1:free", 
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": pregunta.text},
            ],
            stream=False
        )

        # Devuelve la respuesta de DeepSeek
        return {
            "respuesta": response.choices[0].message.content,
            "model": response.model,
            "tokens_utilizados": response.usage.total_tokens
        }
    
    except requests.exceptions.HTTPError as e:
        # Maneja errores HTTP (por ejemplo, 402 - Saldo insuficiente)
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        # Depuración: Captura y muestra cualquier excepción
        print("--> Error en la solicitud:", str(e))
        raise HTTPException(status_code=500, detail=str(e))
    
# Inicia el servidor
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)