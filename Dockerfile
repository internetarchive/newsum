#!/usr/bin/env -S docker image build -t newsum . -f

FROM        python:3

ENV         STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
WORKDIR     /app
CMD         ["streamlit", "run", "main.py"]

RUN         pip install \
              langchain \
              llama-index \
              openai \
              scikit-learn \
              srt \
              streamlit \
              wordcloud

COPY        . ./
