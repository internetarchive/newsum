#!/usr/bin/env -S docker image build -t newsum . -f

FROM        python:3

ENV         STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
            STREAMLIT_SERVER_ENABLE_STATIC_SERVING=true

WORKDIR     /app

RUN         pip install \
              streamlit \
              wordcloud
COPY        requirements.txt ./
RUN         pip install -r requirements.txt

COPY        . ./

CMD         ["streamlit", "run", "main.py"]
