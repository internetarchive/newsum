#!/usr/bin/env python3

import json
import re
import requests

import altair as alt
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from datetime import datetime, timedelta
from gpt_index import GPTSimpleVectorIndex, GPTListIndex, Document, LLMPredictor
from langchain.llms import OpenAI
from requests.exceptions import HTTPError
from wordcloud import WordCloud


TITLE = "Daily News Summary"
ICON = "https://archive.org/favicon.ico"
VISEXP = "https://storage.googleapis.com/data.gdeltproject.org/gdeltv3/iatv/visualexplorer"

BGNDT = pd.to_datetime("2022-03-25")
ENDDT = datetime.now() - timedelta(hours=30)

CHANNELS = {
  "": "-- Select --",
  "ESPRESO": "Espreso TV",
  "RUSSIA1": "Russia-1",
  "RUSSIA24": "Russia-24",
  "1TV": "1TV",
  "NTV": "NTV",
  "BELARUSTV": "Belarus TV",
  "IRINN": "Islamic Republic of Iran News Network"
}

st.set_page_config(page_title=TITLE, page_icon=ICON, layout="centered")
st.title(TITLE)

llm_predictor = LLMPredictor(llm=OpenAI(max_tokens=1024, model_name="text-davinci-003"))


@st.cache_resource(show_spinner=False)
def load_transcript(id, lg):
  lang = "" if lg == "Original" else ".en"
  r = requests.get(f"{VISEXP}/{id}.transcript{lang}.txt")
  r.raise_for_status()
  return r.content


@st.cache_resource(show_spinner=False)
def load_index(ch, dt, lg):
  r = requests.get(f"{VISEXP}/{ch}.{dt}.inventory.json")
  r.raise_for_status()
  shows = r.json()["shows"]
  idx = GPTSimpleVectorIndex([], llm_predictor=llm_predictor)
  msg = f"Loading `{dt[:4]}-{dt[4:6]}-{dt[6:8]}` {lg} transcripts for `{CHANNELS.get(ch, 'selected')}` channel..."
  prog = st.progress(0.0, text=msg)
  for i, tr in enumerate(shows, start=1):
    try:
      idx.insert(Document(load_transcript(tr["id"], lg).decode("utf-8")), llm_predictor=llm_predictor)
    except HTTPError as e:
      pass
    prog.progress(i/len(shows), text=msg)
  prog.empty()
  return idx


@st.cache_resource(show_spinner="Extracting top entities...")
def get_top_entities(_idx, ch, dt, lg):
  res = _idx.query("20 most frequent entities with their frequency in these articles as key-value in JSON format")
  kw = json.loads(res.response.strip())
  wc = WordCloud(background_color="white")
  wc.generate_from_frequencies(kw)
  fig, ax = plt.subplots()
  ax.imshow(wc)
  ax.axis("off")
  return fig, pd.DataFrame(kw.items()).rename(columns={0: "Entity", 1: "Frequency"}).sort_values("Frequency", ascending=False)


@st.cache_resource(show_spinner="Constructing news headlines...")
def get_headlines(_idx, ch, dt, lg):
  return _idx.query("Top 10 news headlines with summary in these articles in Markdown format")


cols = st.columns(3)
dt = cols[0].date_input("Date", value=ENDDT, min_value=BGNDT, max_value=ENDDT, key="dt").strftime("%Y%m%d")
ch = cols[1].selectbox("Channel", CHANNELS, format_func=lambda x: CHANNELS.get(x, ""), key="ch")
lg = cols[2].selectbox("Language", ["English (Translation)", "Original"], key="lg", disabled=True) # Disabled due to a bug https://github.com/jerryjliu/gpt_index/issues/294

if not ch:
  st.info(f"Select a channel to summarize for the selected day.")
  st.stop()

try:
  idx = load_index(ch, dt, lg)
except HTTPError as e:
  st.warning(f"Transcripts for `{CHANNELS.get(ch, 'selected')}` channel are not available for `{dt[:4]}-{dt[4:6]}-{dt[6:8]}` yet, try selecting another date!", icon="⚠️")
  st.stop()

tbs = st.tabs(["Top Entities", "Frequencies"])
try:
  fig, d = get_top_entities(idx, ch, dt, lg)
  tbs[0].pyplot(fig)
  tbs[1].dataframe(d, use_container_width=True)
except:
  msg = "Entity frequency data is not in the expected JSON shape!"
  tbs[0].warning(msg)
  tbs[1].warning(msg)


"### Top Headlines"

res = get_headlines(idx, ch, dt, lg)
st.markdown(res.response)
