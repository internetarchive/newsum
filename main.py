#!/usr/bin/env python3

import json
import requests
import os

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from datetime import datetime, timedelta

from requests.exceptions import HTTPError

from newsum.summary import *


SUMMARYLOC = "./summaries"

TITLE = "News Summary"
DESC = """
This experimental service presents summaries of the top news stories of archived TV News Channels from around the world.  Audio from those archives are transcribed and translated using Google Cloud services and then stories are identified and summarized using various AI LLMs (we are currently experimenting with several, including Vicuna and GPT-3.5).

This is a work-in-progress and you should expect to see poorly transcribed, translated and/or summarized text and some "hallucinations".

Questions and feedback are requested and appreciated!  How might this service be more useful to you?  Please share your thoughts with info@archive.org.
"""
ICON = "https://archive.org/favicon.ico"

BGNDT = pd.to_datetime("2022-03-25").date()
ENDDT = (datetime.now() - timedelta(hours=30)).date()

CHANNELS = {
  "": "-- Select --",
  "ESPRESO": "Espreso TV",
  "RUSSIA1": "Russia-1",
  "RUSSIA24": "Russia-24",
  "1TV": "Channel One Russia",
  "NTV": "NTV",
  "BELARUSTV": "Belarus TV",
  "IRINN": "Islamic Republic of Iran News Network"
}


st.set_page_config(page_title=TITLE, page_icon=ICON, layout="centered", initial_sidebar_state="collapsed")
st.title(TITLE)
st.info(DESC)


with st.expander("How It Works?"):
  st.subheader("Archiving TV Stream")
  st.write("TV Tuners running at the Internet Archive to build the [TV News collection](https://archive.org/tv).")

  st.subheader("Transcription and Translation")
  st.write("GDELT project uses Google Cloud APIs to generate [subtitles](https://blog.gdeltproject.org/visual-explorer-quick-workflow-for-downloading-belarusian-russian-ukrainian-transcripts-translations/) in the original language as well as English.")

  st.subheader("Chunking")
  st.image("static/chunking.jpg")

  st.subheader("Vectorization")
  st.image("static/vectorization.jpg")

  st.subheader("Clustering and Sampling")
  st.image("static/clustering.jpg")

  st.subheader("Summarization")
  st.write("JSON summary using the following LLM prompt:")
  st.code("""
```{doc}```

Create the most prominent headline from the text enclosed
in three backticks (```) above, describe it in a paragraph,
assign a category to it, determine whether it is of international interest,
determine whether it is an advertisement, and assign the top three keywords
in the following JSON format:

{
    "title": "<TITLE>",
    "description": "<DESCRIPTION>",
    "category": "<CATEGORY>",
    "international_interest": true|false,
    "advertisement": true|false,
    "keywords": ["<KEYWORD1>", "<KEYWORD2>", "<KEYWORD3>"]
}
""", language="json")


@st.cache_resource(show_spinner="Loading program inventory...")
def load_inventory_df(ch, dt):
  r = requests.get(f"{VISEXP}/{ch}.{dt}.inventory.json")
  r.raise_for_status()
  return pd.json_normalize(r.json(), record_path="shows").sort_values("start_time", ignore_index=True)


def draw_summaries(json_output):
  for smr in json_output:
    try:
      st.subheader(smr.get("title", "[EMPTY]"))
      cols = st.columns([1, 2])
      with cols[0]:
        components.iframe(f'https://archive.org/embed/{smr["id"]}?start={smr["start"]}&end={smr["end"]}')
      with cols[1]:
        st.write(smr.get("description", "[EMPTY]"))
      with st.expander(f'[{id_to_time(smr["id"], smr["start"])}] `{smr.get("category", "[EMPTY]").upper()}`'):
        st.write(smr)
    except json.JSONDecodeError as _:
      pass


qp = st.query_params
ss = st.session_state

if "date" not in ss and qp.get("date"):
  ss["date"] = datetime.strptime(qp.get("date"), "%Y-%m-%d").date()
if "chan" not in ss and qp.get("chan"):
  ss["chan"] = qp.get("chan")
if "lang" not in ss and qp.get("lang"):
  ss["lang"] = qp.get("lang")
if "llm" not in ss and qp.get("llm"):
  ss["llm"] = qp.get("llm")
if "chunk" not in ss:
  ss["chunk"] = int(qp.get("chunk")) if "chunk" in qp else 30
if "count" not in ss:
  ss["count"] = int(qp.get("count")) if "count" in qp else 20
if "exad" not in ss and qp.get("exad"):
  ss["exad"] = qp.get("exad")[:1].lower() in ("t", "1")
if "exlc" not in ss and qp.get("exlc"):
  ss["exlc"] = qp.get("exlc")[:1].lower() in ("t", "1")

with st.sidebar:
  st.title("Configurations")
  lm = st.radio("LLM", ["OpenAI", "Vicuna"], key="llm", disabled=True)
  ck = st.slider("Chunk size (sec)", min_value=3, max_value=120, step=3, key="chunk")
  ct = st.slider("Cluster count", min_value=1, max_value=50, key="count")
  ad = st.toggle("Exclude ads", key="exad")
  lc = st.toggle("Exclude local news", key="exlc")

cols = st.columns([1, 2, 1])
dt = cols[0].date_input("Date", value=ENDDT, min_value=BGNDT, max_value=ENDDT, key="date").strftime("%Y%m%d")
ch = cols[1].selectbox("Channel", CHANNELS, format_func=lambda x: CHANNELS.get(x, ""), key="chan")
lg = cols[2].selectbox("Language", ["English", "Original"], key="lang", disabled=True)

if not ch:
  st.info(f"Select a channel to summarize for the selected day.")
  st.stop()

for k, v in ss.items():
  qp[k] = v

try:
  inventory = load_inventory_df(ch, dt)
except HTTPError as _:
  st.warning(f"Inventory for `{CHANNELS.get(ch, 'selected')}` channel is not available for `{dt[:4]}-{dt[4:6]}-{dt[6:8]}` yet, try selecting another date!", icon="⚠️")
  st.stop()

with st.expander("Program Inventory"):
  inventory

jf = f"{ch}-{dt}-{lm}-{lg}.json"
jfp = f"{SUMMARYLOC}/{jf}"
if jf not in os.listdir(SUMMARYLOC):
  with st.spinner("Loading and summarizing program transcripts..."):
    summaries_json = summarize(ch, dt, lg, lm, ck, ct)
  if not summaries_json:
    st.error("Summarization failed, try again!")
    st.stop()
  with open(jfp, 'w+') as f:
    json.dump(summaries_json, f, indent=2)

try:
  summaries_json = json.load(open(jfp))
  pdj = pd.read_json(jfp)[["title", "category", "international_interest", "advertisement", "keywords"]].rename(columns={"title": "Title", "category": "Category", "international_interest": "Intl", "advertisement": "Advt", "keywords": "Keywords"})
except Exception as e:
  st.exception(e)

with st.expander("Overview"):
  cols = st.columns(3)
  cols[0].metric("Headlines", len(pdj))
  cols[1].metric("Advertisements", len(pdj[pdj.Advt]))
  cols[2].metric("International", len(pdj[pdj.Intl]))
  pdj

if ad:
  summaries_json = [s for s in summaries_json if not s.get("advertisement")]
if lc:
  summaries_json = [s for s in summaries_json if s.get("international_interest")]

draw_summaries(summaries_json)
