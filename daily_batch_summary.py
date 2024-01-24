#!/usr/bin/env python3

import json
import os
import sys

from datetime import datetime, timedelta
from pathlib import Path

from requests.exceptions import HTTPError

from newsum.summary import *

DTFMT = "%Y-%m-%d %H:%M:%S"
SUMMARYLOC = "./summaries"
CHANNELS = [
  "ESPRESO",
  "RUSSIA1",
  "RUSSIA24",
  "1TV",
  "NTV",
  "BELARUSTV",
  "IRINN"
]


if __name__ == "__main__":
  Path(SUMMARYLOC).mkdir(parents=True, exist_ok=True)
  dt = sys.argv[1].replace("-", "") if len(sys.argv) > 1 else (datetime.now() - timedelta(hours=30)).strftime("%Y%m%d")
  dtf = f"{dt[:4]}-{dt[4:6]}-{dt[6:8]}"
  lg = sys.argv[2] if len(sys.argv) > 2 else "English"
  lm = sys.argv[3] if len(sys.argv) > 3 else "OpenAI"
  ck = int(sys.argv[4]) if len(sys.argv) > 4 else 30
  ct = int(sys.argv[5]) if len(sys.argv) > 5 else 20

  print(f"Generating summaries of all channels from {dtf} in {lg} using {lm}")
  for ch in CHANNELS:
    print(f"[{datetime.now().strftime(DTFMT)}] Summarizing {ch}")
    try:
      summaries_json = summarize(ch, dt, lg, lm, ck, ct)
    except HTTPError as _:
      pass
    if not summaries_json:
      print("ERROR: Summarization of {ch} from {dtf} failed!")
      continue
    jfp = f"{SUMMARYLOC}/{ch}-{dt}-{lm}-{lg}.json"
    with open(jfp, "w+") as f:
      json.dump(summaries_json, f, indent=2)
  print(f"[{datetime.now().strftime(DTFMT)}] ALL DONE!")
