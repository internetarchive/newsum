#!/usr/bin/env python3

from setuptools import setup
from newsum import __version__

with open("README.md") as f:
  ldesc = f.read()
with open("requirements.txt") as f:
  requs = f.read().splitlines()

setup(
  name="newsum",
  version=__version__,
  url="https://github.com/internetarchive/newsum",
  download_url="https://github.com/internetarchive/newsum",
  author="Sawood Alam",
  author_email="sawood@archive.org",
  description="A tool to summarize daily news from the TV News Archive of the Internet Archive using GPT.",
  packages=setuptools.find_packages(),
  python_requires=">=3.11",
  license="AGPLv3",
  long_description=ldesc,
  long_description_content_type="text/markdown",
  provides=["newsum"],
  install_requires=requs,
  keywords="python tv openapi internet-archive summarization gpt gdelt news-summarization tv-news",
  classifiers=[
    "Development Status :: 4 - Beta",
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
    "Intended Audience :: Developers",
    "Intended Audience :: Information Technology",
    "Intended Audience :: Science/Research",
    "Topic :: System :: Archiving",
    "Topic :: Utilities"
  ]
)
