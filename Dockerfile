# Usar imagen base de Python 3.10 slim
FROM python:3.10-slim

# Establecer directorio de trabajo
WORKDIR /app

# Copiar requirements
COPY requirements_api.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir --default-timeout=300 -r requirements_api.txt

# Copiar código fuente
COPY src/ ./src/

# Exponer puerto para uvicorn
EXPOSE 8000

# Comando para ejecutar la aplicación
CMD ["uvicorn", "src.model_deploy:app", "--host", "0.0.0.0", "--port", "8000"]
