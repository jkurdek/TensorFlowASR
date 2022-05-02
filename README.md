# Efficient Minum Word Error Rate Training for Attention-Based models

The repository contains implementation of Minimum Word Error Rate (MWER) Training based on Monotonic RNN-T Loss for audio transduction. The solution is based on the TensorFlowASR Library (https://github.com/TensorSpeech/TensorFlowASR).

## Installation:
***NOTE***: We assume the user has sucesfully installed and configured Python (and Docker in case of accessing the codee via Docker) prior to the solution installation.

### Manual installation:
1. Install requirements
```
pip install -r requirements-pre.txt && pip install -r requirements.txt
```

2. Download LibriSpeech test dataset.
```
wget https://www.openslr.org/resources/12/test-clean.tar.gz

tar -xzvf test-clean.tar.gz 
```

3. (Optional) Download LibriSpeech train and dev datasets. ***NOTE***: Training will fail on majority of consumer tier GPU devices due to lack of memory.
```
wget https://www.openslr.org/resources/12/dev-clean.tar.gz
tar -xzvf dev-clean.tar.gz 

wget https://www.openslr.org/resources/12/train-clean-360.tar.gz
tar -xzvf train-clean-360.tar.gz 
```

4. Create transcriptions (.tsv files) for downloaded datasets.
```
python ./scripts/create_librispeech_trans.py -d <extracted_dataset_dir> <target_dir>
```
```
python ./scripts/create_librispeech_trans.py -d ./LibriSpeech /data/LibriSpeech/test_transcriptions/test.tsv
```

5. Provide paths to the generated transcriptions in the config file ('data_path' in train, eval and test subconfigs)
```
./examples/conformer/config.yml
```

## Docker Installation:

For convenience we provide a way to access the code using Docker. The guide assumes user has installed docker, docker-compose and nvidia-docker (https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#install-guide).

 ***NOTE***: By default docker installation downloads only the test dataset. 

1. Build and run docker container.
```
docker compose run tensorflow_asr
```

## Training:


## Test:
