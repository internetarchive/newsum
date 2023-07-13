#!/usr/bin/env python3

import json
import os
from multiprocessing.pool import ThreadPool
import openai
from datetime import datetime, timedelta
from requests.exceptions import HTTPError

from functions import load_inventory, select_docs, get_summary
from functions import THREAD_COUNT


VICUNA = "http://fc6000.sf.archive.org:8000/v1"
OUTPUT_FOLDER_NAME = "summaries"

CHANNELS = [
  "ESPRESO",
  "RUSSIA1",
  "RUSSIA24",
  "1TV",
  "NTV",
  "BELARUSTV",
  "IRINN",
]

LM = "OpenAI" # language model
CK = 30 # chunk size
CT = 20 # cluster count
DT = (datetime.now() - timedelta(hours=30)).date().strftime("%Y%m%d") # date
LG = "English" # language


if LM == "Vicuna":
  openai.api_key = "EMPTY"
  openai.api_base = VICUNA

# attempt to create ./OUTPUT_FOLDER_NAME
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
  seldocs = select_docs(DT, ch, LG, LM, CK, CT, inventory)

  print("begin summarizing each document...")

  summary_args = [(d,LM) for d in seldocs]
  with ThreadPool(THREAD_COUNT) as pool:
    summaries = pool.starmap(get_summary, summary_args)

  print("writing results...")
  with open(f"{OUTPUT_FOLDER_NAME}/{DT}-{ch}-{LM}-{LG}.json", 'w+') as f:
    f.write(json.dumps(summaries, indent=2))
  print(f"finished {ch}")
