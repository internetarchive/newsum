#!/usr/bin/env -S docker image build -t newsum . -f

FROM        python:3

ENV         STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
WORKDIR     /app
ENTRYPOINT  ["streamlit", "run"]
CMD         ["main.py"]

RUN         pip install streamlit gpt-index

RUN         cd /tmp \
              && git clone https://github.com/amueller/word_cloud.git \
              && cd word_cloud \
              && pip install . \
              && cd /app \
              && rm -rf /tmp/word_cloud

COPY        . ./
