FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Allow optional heavy deps via build-arg (default ON to match TZ v3)
ARG INSTALL_OPTIONALS=1
ENV INSTALL_OPTIONALS=${INSTALL_OPTIONALS}

COPY requirements.txt ./
COPY requirements-optional.txt ./
RUN pip install -r requirements.txt \
    && if [ "$INSTALL_OPTIONALS" = "1" ]; then pip install -r requirements-optional.txt; fi

COPY src ./src

ENV PYTHONPATH=/app/src

# NLTK/TextBlob corpora
RUN python -m textblob.download_corpora && python - <<'PY'
import nltk
nltk.download('vader_lexicon')
PY

ENTRYPOINT ["python"]
