FROM python:3.11-slim
WORKDIR /app
COPY requirements/base.txt /tmp/base.txt
RUN pip install --no-cache-dir -r /tmp/base.txt
COPY . /app
ENV PYTHONPATH=/app/src
CMD ["python", "scripts/smoke_mujoco.py"]
