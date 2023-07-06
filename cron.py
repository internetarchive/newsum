#!/usr/bin/env python3

import json
import re
import requests
import os

import openai
import srt

import numpy as np
import pandas as pd

from datetime import datetime, timedelta

from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from requests.exceptions import HTTPError
from sklearn.cluster import KMeans

DEV_ENV = False


TITLE = "NewSum: Daily TV News Summary"
ICON = "https://archive.org/favicon.ico"
VISEXP = "https://storage.googleapis.com/data.gdeltproject.org/gdeltv3/iatv/visualexplorer"
VICUNA = "http://fc6000.sf.archive.org:8000/v1"
MODEL = "gpt-4"
OUTPUT_FOLDER_NAME = "summaries"

IDDTRE = re.compile(r"^.+_(\d{8}_\d{6})")

CHANNELS = [
  # "ESPRESO",
  # "RUSSIA1",
  # "RUSSIA24",
  "1TV",
  # "NTV",
  # "BELARUSTV",
  # "IRINN",
]

LM = "OpenAI" # language model
CK = 30 # chunk size
CT = 20 # cluster count
DT = (datetime.now() - timedelta(hours=30)).date().strftime("%Y%m%d") # date
CH = "BELARUSTV" # channel
LG = "English" # language


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
    if DEV_ENV: break;
  return chks


def load_vectors(docs):
  embed = OpenAIEmbeddings()
  vectors = []
  for i, d in enumerate(docs, start=1):
    vectors.append(embed.embed_query(d.page_content))
  return vectors


def select_docs(dt, ch, lg, lm, ck, ct):
  docs = load_chunks(inventory, lg, ck)
  vectors = load_vectors(docs)
  print("number of vectors =", len(vectors))
  kmeans = KMeans(n_clusters=ct, random_state=10, n_init=10).fit(vectors)
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


if LM == "Vicuna":
  openai.api_key = "EMPTY"
  openai.api_base = VICUNA

# attempt to create ./summaries
try:
    os.mkdir(f"./{OUTPUT_FOLDER_NAME}")
except FileExistsError:
        pass

for ch in CHANNELS:
  print(f"---\nstarting {ch}...")
  try:
    print("loading inventory...")
    inventory = load_inventory(ch, DT, LG)
  except HTTPError as _:
    print(f"Inventory for `{ch}` channel is not available for `{DT[:4]}-{DT[4:6]}-{DT[6:8]}` yet, try selecting another date!", icon="⚠️")

  print("loading documents...")
  seldocs = select_docs(DT, ch, LG, LM, CK, CT)

  with open(f"{OUTPUT_FOLDER_NAME}/{ch}-{DT[:4]}-{DT[4:6]}-{DT[6:8]}-{LM}-{LG}.json", 'w+') as file:
    print("summarizing each document...")
    if DEV_ENV: seldocs = seldocs[:1]
    for d in seldocs:
      try:
        smr = get_summary(d.page_content, LM)
        smr["metadata"] = d.metadata
        print("writing results...")
        file.write(json.dumps(smr, indent=2))
      except json.JSONDecodeError as _:
        pass
  print(f"finished {ch}")
