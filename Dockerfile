# app/Dockerfile

FROM python:3.10-slim

EXPOSE 8501

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/FNRocha/youtube_comment_scraper .

RUN pip3 install -r requirements.txt

RUN python -m nltk.downloader vader_lexicon

ENTRYPOINT ["streamlit", "run", "stream_app.py", "--server.port=8501", "--server.address=0.0.0.0"]