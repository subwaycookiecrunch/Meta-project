FROM python:3.12-slim
WORKDIR /app
COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONPATH=/app/..
ENV PYTHONIOENCODING=utf-8
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "code_review_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
