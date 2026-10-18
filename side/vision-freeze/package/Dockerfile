# docker build -t eeg-vlm .
# docker run --gpus all -e HF_TOKEN -v $PWD/hf-cache:/root/.cache/huggingface eeg-vlm models/<name> models/<name>/samples/*.png
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY predict.py .
COPY models/ models/
ENTRYPOINT ["python", "predict.py"]
