# Daily TV News Summary

This application summarizes daily news from TV News Archive of the Internet Archive using GPT.

To run it locally (in Docker), clone this repository and build a docker image:

```
$ docker image build -t newsum .
```

Run a container from the freshly built Docker image using an OpenAI API key:

```
$ docker container run --rm -it -p 8501:8501 -e OPENAI_API_KEY="<APIKEY>" newsum
```

Access http://localhost:8501/ in a web browser for interactive insights.
