FROM tensorflow/tensorflow:2.5.1-gpu

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y \
    && apt-get -y install apt-utils gcc libpq-dev libsndfile-dev git build-essential cmake screen wget


# Clear cache
RUN apt clean && apt-get clean

ENV PYTHONPATH "${PYTHONPATH}:/app"

COPY . /app
WORKDIR /app

RUN pip install -r requirements.txt

RUN wget https://www.openslr.org/resources/12/dev-clean.tar.gz
RUN tar -xzvf dev-clean.tar.gz 
RUN python ./scripts/create_librispeech_trans.py -d ./LibriSpeech /data/LibriSpeech/test_transcriptions/test.tsv
