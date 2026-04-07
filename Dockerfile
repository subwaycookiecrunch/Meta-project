FROM python:3.12-slim
WORKDIR /code_review_env
COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONPATH=/
ENV PYTHONIOENCODING=utf-8
ENV ENABLE_WEB_INTERFACE=true
EXPOSE 7860
CMD ["python", "-m", "uvicorn", "code_review_env.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
