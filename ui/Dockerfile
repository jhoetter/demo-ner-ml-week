FROM python:3.9-slim

WORKDIR /app
COPY ./app .

RUN apt-get update
RUN pip3 install torch --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip3 install -r requirements.txt
RUN python3 -m spacy download en_core_web_sm
RUN python3 -c "from embedders.extraction.contextual import TransformerTokenEmbedder; embedder = TransformerTokenEmbedder('distilbert-base-uncased', 'en_core_web_sm')"

EXPOSE 80
ENTRYPOINT ["streamlit", "run"] 
CMD ["app.py", "--server.port", "80"]