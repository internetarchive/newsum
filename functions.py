#!/usr/bin/env python3

import json
import logging
import re
import requests
from multiprocessing.pool import ThreadPool
import os
import time

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


THREAD_COUNT = 15
VISEXP = "https://storage.googleapis.com/data.gdeltproject.org/gdeltv3/iatv/visualexplorer"
LLM_MODELS = {
  "OpenAI": "gpt-3.5-turbo",
  "Vicuna": "text-embedding-ada-002",
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


def load_vectors(doc, llm):
  embed = OpenAIEmbeddings(model=LLM_MODELS[llm])
  return embed.embed_query(doc.page_content)

def select_docs(dt, ch, lg, lm, ck, ct, inventory):
  print("loading chunks...")
  docs = load_chunks(inventory, lg, ck)
  docs_list = [(d,lm) for d in docs]

  print("loading vectors...")
  with ThreadPool(THREAD_COUNT) as pool:
    vectors = pool.starmap(load_vectors, docs_list)

  print("number of vectors =", len(vectors))
  kmeans = KMeans(n_clusters=ct, random_state=10, n_init=10).fit(vectors)
  cent = sorted([np.argmin(np.linalg.norm(vectors - c, axis=1)) for c in kmeans.cluster_centers_])
  return [docs[i] for i in cent]

def get_summary(d, llm):
  msg = f"""
  ```{d}```

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
  for delay_secs in (2**x for x in range(0,6)):
    try:
      res = openai.ChatCompletion.create(
        model=LLM_MODELS[llm],
        messages=[{"role": "user", "content": msg}]
      )
      result = json.loads(res.choices[0].message.content.strip())
      result = result | d.metadata
      result["transcript"] = d.page_content.strip()
      break
    except openai.error.OpenAIError as e:
      print(f"Error: {e}. Retrying in {round(delay_secs, 2)} seconds.")
      time.sleep(delay_secs)
      continue
  return result
