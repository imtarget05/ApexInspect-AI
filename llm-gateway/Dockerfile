FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 GATEWAY_HOST=0.0.0.0
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY server.py .
EXPOSE 8787
# Same entrypoint as `python server.py`: PORT (or the deprecated GATEWAY_PORT alias) + GATEWAY_HOST.
CMD ["python", "server.py"]
