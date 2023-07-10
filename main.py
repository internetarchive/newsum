#!/usr/bin/env python3

import json
import logging
import re
import requests
from multiprocessing.pool import ThreadPool

import openai
import srt

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from datetime import datetime, timedelta

from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.schema import Document
from requests.exceptions import HTTPError
from sklearn.cluster import KMeans


TITLE = "NewSum: Daily TV News Summary"
ICON = "https://archive.org/favicon.ico"
VISEXP = "https://storage.googleapis.com/data.gdeltproject.org/gdeltv3/iatv/visualexplorer"
VICUNA = "http://fc6000.sf.archive.org:8000/v1"

LLM_MODELS = {
  "Vicuna": "text-embedding-ada-002",
  "OpenAI": "gpt-4",
}

IDDTRE = re.compile(r"^.+_(\d{8}_\d{6})")

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

THREAD_COUNT = 25

st.set_page_config(page_title=TITLE, page_icon=ICON, layout="centered", initial_sidebar_state="collapsed")
st.title(TITLE)


@st.cache_resource(show_spinner=False)
def load_srt(id, lg):
  lang = "" if lg == "Original" else ".en"
  r = requests.get(f"{VISEXP}/{id}.transcript{lang}.srt")
  r.raise_for_status()
  return r.content


@st.cache_resource(show_spinner=False)
def load_inventory(ch, dt, lg):
  r = requests.get(f"{VISEXP}/{ch}.{dt}.inventory.json")
  r.raise_for_status()
  return pd.json_normalize(r.json(), record_path="shows").sort_values("start_time", ignore_index=True)


def create_doc(txt, id, start, end):
  return Document(page_content=txt, metadata={"id": id, "start": round(start.total_seconds()), "end": round(end.total_seconds())})


def chunk_srt(sr, id, lim=3.0):
  docs = []
  ln = 0
  txt = ""
  start = end = timedelta()
  for s in srt.parse(sr.decode()):
    cl = (s.end - s.start).total_seconds()
    if ln + cl > lim:
      if txt:
        docs.append(create_doc(txt, id, start, end))
      ln = cl
      txt = s.content
      start = s.start
      end = s.end
    else:
      ln += cl
      txt += " " + s.content
      end = s.end
  if txt:
    docs.append(create_doc(txt, id, start, end))
  return docs


def load_chunks(inventory, lg, ck):
  msg = "Loading SRT files..."
  prog = st.progress(0.0, text=msg)
  chks = []
  sz = len(inventory)
  for i, r in inventory.iterrows():
    try:
      sr = load_srt(r.id, lg)
    except HTTPError as _:
      continue
    chks += chunk_srt(sr, r.id, lim=ck)
    prog.progress((i+1)/sz, text=msg)
  prog.empty()
  return chks


def load_vector(d):
  embed = OpenAIEmbeddings()
  result = embed.embed_query(d.page_content)
  return result


@st.cache_resource(show_spinner="Loading Vectors...")
def select_docs(dt, ch, lg, lm, ck, ct):
  docs = load_chunks(inventory, lg, ck)
  docs_list = [(d,) for d in docs]

  with ThreadPool(THREAD_COUNT) as pool:
    vectors = pool.starmap(load_vector, docs_list)
  st.success('Vectors loaded!')

  kmeans = KMeans(n_clusters=ct, random_state=10).fit(vectors)
  cent = sorted([np.argmin(np.linalg.norm(vectors - c, axis=1)) for c in kmeans.cluster_centers_])
  return [docs[i] for i in cent]


def id_to_time(id, start=0):
  dt = IDDTRE.match(id).groups()[0]
  return datetime.strptime(dt, "%Y%m%d_%H%M%S") + timedelta(seconds=start)


def get_summary(txt, llm):
  msg = f"""
  ```{txt}```

  Create the most prominent headline from the text enclosed in three backticks (```) above, describe it in a paragraph, and assign a category to it in the following JSON format:

  {{
    "title": "<TITLE>",
    "description": "<DESCRIPTION>",
    "category": "<CATEGORY>"
  }}
  """
  res = openai.ChatCompletion.create(
    model=LLM_MODELS[llm],
    messages=[{"role": "user", "content": msg}]
  )
  return res.choices[0].message.content.strip()


@st.cache_resource(show_spinner="Summarizing...")
def gather_summaries(dt, ch, lg, lm, ck, ct, _seldocs):
  summary_args = [(d,lm) for d in _seldocs]
  with ThreadPool(THREAD_COUNT) as pool:
    summaries = pool.starmap(get_summary, summary_args)
  return summaries

qp = st.experimental_get_query_params()
if "date" not in st.session_state and qp.get("date"):
    st.session_state["date"] = datetime.strptime(qp.get("date")[0], "%Y-%m-%d").date()
if "chan" not in st.session_state and qp.get("chan"):
    st.session_state["chan"] = qp.get("chan")[0]
if "lang" not in st.session_state and qp.get("lang"):
    st.session_state["lang"] = qp.get("lang")[0]
if "llm" not in st.session_state and qp.get("llm"):
    st.session_state["llm"] = qp.get("llm")[0]
if "chunk" not in st.session_state and qp.get("chunk"):
    st.session_state["chunk"] = int(qp.get("chunk")[0])
if "count" not in st.session_state and qp.get("count"):
    st.session_state["count"] = int(qp.get("count")[0])

with st.expander("Configurations"):
  lm = st.radio("LLM", ["Vicuna", "OpenAI"], key="llm", horizontal=True)
  ck = st.slider("Chunk size (sec)", value=30, min_value=3, max_value=120, step=3, key="chunk")
  ct = st.slider("Cluster count", value=20, min_value=1, max_value=50, key="count")

cols = st.columns([1, 2, 1])
dt = cols[0].date_input("Date", value=ENDDT, min_value=BGNDT, max_value=ENDDT, key="date").strftime("%Y%m%d")
ch = cols[1].selectbox("Channel", CHANNELS, format_func=lambda x: CHANNELS.get(x, ""), key="chan")
lg = cols[2].selectbox("Language", ["English", "Original"], key="lang")

if not ch:
  st.info(f"Select a channel to summarize for the selected day.")
  st.stop()

st.experimental_set_query_params(**st.session_state)

if lm == "Vicuna":
  openai.api_key = "EMPTY"
  openai.api_base = VICUNA

try:
  inventory = load_inventory(ch, dt, lg)
except HTTPError as _:
  st.warning(f"Inventory for `{CHANNELS.get(ch, 'selected')}` channel is not available for `{dt[:4]}-{dt[4:6]}-{dt[6:8]}` yet, try selecting another date!", icon="⚠️")
  st.stop()

with st.expander("Program Inventory"):
  inventory

seldocs = select_docs(dt, ch, lg, lm, ck, ct)
summaries = gather_summaries(dt, ch, lg, lm, ck, ct, seldocs)
st.success('Summaries loaded!')

for i, d in enumerate(seldocs):
  try:
    smr = json.loads(summaries[i])
    md = d.metadata
    st.subheader(smr.get("title", "[EMPTY]"))
    cols = st.columns([1, 2])
    with cols[0]:
      components.iframe(f'https://archive.org/embed/{md["id"]}?start={md["start"]}&end={md["end"]}')
    with cols[1]:
      st.write(smr.get("description", "[EMPTY]"))
    with st.expander(f'[{id_to_time(md["id"], md["start"])}] `{smr.get("category", "[EMPTY]").upper()}`'):
      st.caption(d.page_content)
  except json.JSONDecodeError as _:
    pass
