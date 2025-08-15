FROM python:3.11-slim as builder

WORKDIR /app

# Set up a virtual environment
ENV VIRTUAL_ENV=/app/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"


COPY requirements.txt . 
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim

WORKDIR /app


COPY --from=builder /app/venv /app/venv


COPY ./app /app/app


ENV PATH="/app/venv/bin:$PATH"


EXPOSE 8000


CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]