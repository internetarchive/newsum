#!/usr/bin/env python3

import json
import logging
import re
import requests

import openai
import srt

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
MODEL = "text-embedding-ada-002"

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


def load_srt(id, lg):
  lang = "" if lg == "Original" else ".en"
  r = requests.get(f"{VISEXP}/{id}.transcript{lang}.srt")
  r.raise_for_status()
  return r.content


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
  chks = []
  for i, r in inventory.iterrows():
    try:
      sr = load_srt(r.id, lg)
    except HTTPError as _:
      continue
    chks += chunk_srt(sr, r.id, lim=ck)
  return chks


def load_vectors(docs):
  embed = OpenAIEmbeddings()
  msg = f"Loading vectors..."
  vectors = []
  for i, d in enumerate(docs, start=1):
    vectors.append(embed.embed_query(d.page_content))
  return vectors


def select_docs(dt, ch, lg, lm, ck, ct):
  docs = load_chunks(inventory, lg, ck)
  vectors = load_vectors(docs)
  kmeans = KMeans(n_clusters=ct, random_state=10).fit(vectors)
  cent = sorted([np.argmin(np.linalg.norm(vectors - c, axis=1)) for c in kmeans.cluster_centers_])
  return [docs[i] for i in cent]


def id_to_time(id, start=0):
  dt = IDDTRE.match(id).groups()[0]
  return datetime.strptime(dt, "%Y%m%d_%H%M%S") + timedelta(seconds=start)


def get_summary(txt, lm):
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
    model=MODEL,
    messages=[{"role": "user", "content": msg}]
  )
  return json.loads(res.choices[0].message.content.strip())


lm = "OpenAI" # language model
ck = 30 # chunk size
ct = 20 # cluster count
dt = ENDDT # date
ch = "ESPRESO" # channel
lg = "English" # language


if lm == "Vicuna":
  openai.api_key = "EMPTY"
  openai.api_base = VICUNA

try:
  inventory = load_inventory(ch, dt, lg)
except HTTPError as _:
  print(f"Inventory for `{CHANNELS.get(ch, 'selected')}` channel is not available for `{dt[:4]}-{dt[4:6]}-{dt[6:8]}` yet, try selecting another date!", icon="⚠️")

seldocs = select_docs(dt, ch, lg, lm, ck, ct)

with open('VICUNA-2023-07-06-OpenAI-English.json', 'w+') as file:
  for d in seldocs:
    try:
      smr = get_summary(d.page_content, lm)
      md = d.metadata
      print(smr)
      # with cols[0]:
      #   components.iframe(f'https://archive.org/embed/{md["id"]}?start={md["start"]}&end={md["end"]}')
      # with cols[1]:
      #   st.write(smr.get("description", "[EMPTY]"))
      # with st.expander(f'[{id_to_time(md["id"], md["start"])}] `{smr.get("category", "[EMPTY]").upper()}`'):
      #   st.caption(d.page_content)
    except json.JSONDecodeError as _:
      pass
