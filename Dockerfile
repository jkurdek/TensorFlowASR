FROM tensorflow/tensorflow:2.5.1-gpu

RUN apt-key del 7fa2af80 \
    && apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/3bf863cc.pub \
    && apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y \
    && apt-get -y install apt-utils gcc libpq-dev libsndfile-dev git build-essential cmake screen wget


# Clear cache
RUN apt clean && apt-get clean

ENV PYTHONPATH "${PYTHONPATH}:/app"

COPY . /app
WORKDIR /app

RUN pip install -r requirements-pre.txt && pip install -r requirements.txt

RUN wget https://www.openslr.org/resources/12/dev-test.tar.gz
RUN tar -xzvf dev-clean.tar.gz 
RUN python ./scripts/create_librispeech_trans.py -d ./LibriSpeech /data/LibriSpeech/test_transcriptions/test.tsv
