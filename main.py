#!/usr/bin/env python3

import json
import logging
import re
import requests
from multiprocessing.pool import ThreadPool
import os

import openai
import srt

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from datetime import datetime, timedelta

from gensim.downloader import load as wvmodel
from gensim.utils import tokenize
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.schema import Document
from requests.exceptions import HTTPError
from sklearn.cluster import KMeans


TITLE = "News Summary"
DESC = """
This experimental service presents summaries of the top news stories of archived TV News Channels from around the world.  Audio from those archives are transcribed and translated using Google Cloud services and then stories are identified and summarized using various AI LLMs (we are currently experimenting with several, including Vicuna and GPT-3.5).

This is a work-in-progress and you should expect to see poorly transcribed, translated and/or summarized text and some "hallucinations".

Questions and feedback are requested and appreciated!  How might this service be more useful to you?  Please share your thoughts with info@archive.org.
"""
ICON = "https://archive.org/favicon.ico"
VISEXP = "https://storage.googleapis.com/data.gdeltproject.org/gdeltv3/iatv/visualexplorer"
VICUNA = "http://fc6000.sf.archive.org:8000/v1"

LLM_MODELS = {
  "OpenAI": "gpt-3.5-turbo",
  "Vicuna": "text-embedding-ada-002",
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

THREAD_COUNT = 15

st.set_page_config(page_title=TITLE, page_icon=ICON, layout="centered", initial_sidebar_state="auto")
st.title(TITLE)
st.info(DESC)


with st.expander("How It Works?"):
  st.subheader("Archiving TV Stream")
  st.write("TV Tuners running at the Internet Archive to build the [TV News collection](https://archive.org/tv).")

  st.subheader("Transcription and Translation")
  st.write("GDELT project uses Google Cloud APIs to generate [subtitles](https://blog.gdeltproject.org/visual-explorer-quick-workflow-for-downloading-belarusian-russian-ukrainian-transcripts-translations/) in the original language as well as English.")

  st.subheader("Chunking")
  st.image("https://archive.org/~sawood/tmp/NewSum_Chunking.jpg")

  st.subheader("Vectorization")
  st.image("https://archive.org/~sawood/tmp/NewSum_Vectorization.jpg")

  st.subheader("Clustering and Sampling")
  st.image("https://archive.org/~sawood/tmp/NewSum_Clustering.jpg")

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


@st.cache_resource()
def load_word2vec_model(name):
  return wvmodel(name)


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
#  msg = "Loading SRT files..."
#  prog = st.progress(0.0, text=msg)
  chks = []
  sz = len(inventory)
  for i, r in inventory.iterrows():
    try:
      sr = load_srt(r.id, lg)
    except HTTPError as _:
      continue
    chks += chunk_srt(sr, r.id, lim=ck)
#    prog.progress((i+1)/sz, text=msg)
#  prog.empty()
  return chks


def load_vectors(doc):
  tkns = list(tokenize(doc.page_content, lower=True, deacc=True)) or ["EMPTY"]
  return GSWVMODEL.get_mean_vector(tkns)


@st.cache_resource(show_spinner="Loading and processing transcripts...")
def select_docs(dt, ch, lg, ck, ct):
  docs = load_chunks(inventory, lg, ck)
  docs_list = [(d,) for d in docs]

  with ThreadPool(THREAD_COUNT) as pool:
    vectors = pool.starmap(load_vectors, docs_list)

  kmeans = KMeans(n_clusters=ct, random_state=10).fit(vectors)
  cent = sorted([np.argmin(np.linalg.norm(vectors - c, axis=1)) for c in kmeans.cluster_centers_])
  return [docs[i] for i in cent]


def id_to_time(id, start=0):
  dt = IDDTRE.match(id).groups()[0]
  return datetime.strptime(dt, "%Y%m%d_%H%M%S") + timedelta(seconds=start)


def get_summary(d, llm):
  msg = f"""
  ```{d.page_content.strip()}```

  Create the most prominent headline from the text enclosed in three backticks (```) above, describe it in a paragraph, assign a category to it, determine whether it is of international interest, determine whether it is an advertisement, and assign the top three keywords in the following JSON format:

  {{
    "title": "<TITLE>",
    "description": "<DESCRIPTION>",
    "category": "<CATEGORY>",
    "international_interest": true|false,
    "advertisement": true|false,
    "keywords": ["<KEYWORD1>", "<KEYWORD2>", "<KEYWORD3>"]
  }}
  """
  retries = 5
  while retries:
    try:
      res = openai.ChatCompletion.create(
        model=LLM_MODELS[llm],
        messages=[{"role": "user", "content": msg}]
      )
      result = json.loads(res.choices[0].message.content.strip())
      result = result | d.metadata
      result["transcript"] = d.page_content.strip()
      return result
    except json.JSONDecodeError as _:
      retries -= 1
  return None


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


@st.cache_resource(show_spinner="Summarizing...")
def gather_summaries(dt, ch, lg, lm, ck, ct, _seldocs):
  summary_args = [(d,lm) for d in _seldocs]
  with ThreadPool(THREAD_COUNT) as pool:
    summaries = pool.starmap(get_summary, summary_args)
  return list(filter(None, summaries))


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
if "exad" not in st.session_state and qp.get("exad"):
  st.session_state["exad"] = qp.get("exad")[0][:1].lower() in ("t", "1")
if "exlc" not in st.session_state and qp.get("exlc"):
  st.session_state["exlc"] = qp.get("exlc")[0][:1].lower() in ("t", "1")

with st.sidebar:
  st.title("Configurations")
  lm = st.radio("LLM", ["OpenAI", "Vicuna"], key="llm", disabled=True)
  ck = st.slider("Chunk size (sec)", value=30, min_value=3, max_value=120, step=3, key="chunk")
  ct = st.slider("Cluster count", value=20, min_value=1, max_value=50, key="count")
  ad = st.toggle("Exclude ads", key="exad")
  lc = st.toggle("Exclude local news", key="exlc")

cols = st.columns([1, 2, 1])
dt = cols[0].date_input("Date", value=ENDDT, min_value=BGNDT, max_value=ENDDT, key="date").strftime("%Y%m%d")
ch = cols[1].selectbox("Channel", CHANNELS, format_func=lambda x: CHANNELS.get(x, ""), key="chan")
lg = cols[2].selectbox("Language", ["English", "Original"], key="lang", disabled=True)

if not ch:
  st.info(f"Select a channel to summarize for the selected day.")
  st.stop()

gloc = pd.DataFrame({
  "channel": ["ESPRESO", "RUSSIA1"],
  "lat": [50.450001, 55.751244],
  "lon": [30.523333, 37.618423]
})

# st.map(gloc, size=10000)

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

jf = f"{ch}-{dt}-{lm}-{lg}.json"
if jf not in os.listdir("./summaries"):
  GSWVMODEL = load_word2vec_model("glove-wiki-gigaword-50")
  seldocs = select_docs(dt, ch, lg, ck, ct)
  summaries_json = gather_summaries(dt, ch, lg, lm, ck, ct, seldocs)
  with open(f"summaries/{jf}", 'w+') as f:
    f.write(json.dumps(summaries_json, indent=2))

try:
  summaries_json = json.load(open(f"./summaries/{jf}"))
  pdj = pd.read_json(f"./summaries/{jf}")[["title", "category", "international_interest", "advertisement", "keywords"]].rename(columns={"title": "Title", "category": "Category", "international_interest": "Intl", "advertisement": "Advt", "keywords": "Keywords"})
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
