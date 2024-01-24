#!/usr/bin/env python3

import json
import re
import requests
import sys

from datetime import datetime, timedelta
from multiprocessing.pool import ThreadPool

import srt

import numpy as np

from gensim.downloader import load as wvmodel
from gensim.parsing.preprocessing import remove_stopword_tokens
from gensim.utils import tokenize
from langchain.schema import Document
from openai import OpenAI, BadRequestError
from requests.exceptions import HTTPError
from sklearn.cluster import KMeans, DBSCAN


VISEXP = "https://storage.googleapis.com/data.gdeltproject.org/gdeltv3/iatv/visualexplorer"
IDDTRE = re.compile(r"^.+_(\d{8}_\d{6})")
THREAD_COUNT = 15

GSWVMODEL = "glove-wiki-gigaword-50"
LLM_MODELS = {"OpenAI": "gpt-3.5-turbo"}


def cache_model(func):
  cache = {}
  def wrapper(*a, **kw):
    k = (*a, *kw.items())
    if k not in cache:
      cache[k] = func(*a, **kw)
    return cache[k]
  return wrapper


@cache_model
def load_word2vec_model(name):
  return wvmodel(name)


def load_srt(id, lg):
  lang = "" if lg == "Original" else ".en"
  r = requests.get(f"{VISEXP}/{id}.transcript{lang}.srt")
  r.raise_for_status()
  return r.content


def load_inventory(ch, dt):
  r = requests.get(f"{VISEXP}/{ch}.{dt}.inventory.json")
  r.raise_for_status()
  return r.json()["shows"]


def create_doc(txt, id, start, end):
  return Document(page_content=txt.strip(), metadata={"id": id, "start": round(start.total_seconds()), "end": round(end.total_seconds()), "size":len(txt.strip().split())})


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


def load_chunks(ch, dt, lg, ck):
  chks = []
  for r in load_inventory(ch, dt):
    id = r.get("id", "")
    try:
      sr = load_srt(id, lg)
    except HTTPError as _:
      continue
    chks += chunk_srt(sr, id, lim=ck)
  return chks


def load_vectors(doc, wvmdl):
  tkns = remove_stopword_tokens(tokenize(doc.page_content, lower=True, deacc=True)) or ["EMPTY"]
  return wvmdl.get_mean_vector(tkns)


def select_docs(dt, ch, lg, ck, ct):
  docs = load_chunks(ch, dt, lg, ck)
  wvmdl = load_word2vec_model(GSWVMODEL)
  docs_list = [(d,wvmdl) for d in docs]
  with ThreadPool(THREAD_COUNT) as pool:
    vectors = pool.starmap(load_vectors, docs_list)
  kmeans = KMeans(n_clusters=ct, random_state=10).fit(vectors)
  cent = sorted([np.argmin(np.linalg.norm(vectors - c, axis=1)) for c in kmeans.cluster_centers_])
  return [docs[i] for i in cent]


def id_to_time(id, start=0):
  dt = IDDTRE.match(id).groups()[0]
  return datetime.strptime(dt, "%Y%m%d_%H%M%S") + timedelta(seconds=start)


def get_summary(d, mdl):
  msg = f"""
  ```{d.page_content}```

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
  client = OpenAI()
  retries = 5
  while retries:
    try:
      res = client.chat.completions.create(model=mdl, messages=[{"role": "user", "content": msg}])
      result = json.loads(res.choices[0].message.content.strip())
      result = result | d.metadata
      result["transcript"] = d.page_content
      return result
    except json.JSONDecodeError as _:
      retries -= 1
    except BadRequestError as _:
      msg = msg[1000:] # Hack to reduce context length
      retries -= 1
  return None


def gather_summaries(seldocs, lm):
  summary_args = [(d,LLM_MODELS[lm]) for d in seldocs]
  with ThreadPool(THREAD_COUNT) as pool:
    summaries = pool.starmap(get_summary, summary_args)
  return list(filter(None, summaries))


def summarize(ch, dt, lg="English", lm="OpenAI", ck=30, ct=20):
  seldocs = select_docs(dt, ch, lg, ck, ct)
  return gather_summaries(seldocs, lm)


if __name__ == "__main__":
  if len(sys.argv) < 2:
    sys.exit(f"Usage:\v{sys.argv[0]} <Channel> [<YYYYMMDD> [English|Original [<LLM> [<ChunkSize> [ClusterCT]]]]]")
  ch = sys.argv[1]
  dt = sys.argv[2] if len(sys.argv) > 2 else (datetime.now() - timedelta(hours=30)).strftime("%Y%m%d")
  lg = sys.argv[3] if len(sys.argv) > 3 else "English"
  lm = sys.argv[4] if len(sys.argv) > 4 else "OpenAI"
  ck = int(sys.argv[5]) if len(sys.argv) > 5 else 30
  ct = int(sys.argv[6]) if len(sys.argv) > 6 else 20
  json.dump(summarize(ch, dt, lg, lm, ck, ct), sys.stdout, indent=2)
