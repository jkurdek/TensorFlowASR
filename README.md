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

3. Create transcriptions (.tsv files) for downloaded datasets.
```
python ./scripts/create_librispeech_trans.py -d <extracted_dataset_dir> <target_dir>
```
```
python ./scripts/create_librispeech_trans.py -d ./LibriSpeech/test-clean /data/LibriSpeech/test_transcriptions/test.tsv
```

5. Provide paths to the generated transcriptions in the config file ('data_path' in train, eval and test subconfigs)
```
./examples/conformer/config.yml
```

### Docker Installation:

For convenience we provide a way to access the code using Docker. The guide assumes user has installed docker, docker-compose and nvidia-docker (https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#install-guide).

 ***NOTE***: By default docker installation downloads only the test dataset. 

1. Build and run docker container.
```
docker compose run tensorflow_asr
```

## Training:
1. Download LibriSpeech train and dev datasets. 
***NOTE***: Datasets are quite large (around 24GB) so download process might take a while. Also training will fail on majority of consumer tier GPU devices due to lack of memory.
```
wget https://www.openslr.org/resources/12/dev-clean.tar.gz
tar -xzvf dev-clean.tar.gz 

wget https://www.openslr.org/resources/12/train-clean-360.tar.gz
tar -xzvf train-clean-360.tar.gz 
```
2. Create transcriptions (.tsv files) for downloaded datasets.
```
python ./scripts/create_librispeech_trans.py -d ./LibriSpeech/train-clean-360 /data/LibriSpeech/test_transcriptions/train.tsv
python ./scripts/create_librispeech_trans.py -d ./LibriSpeech/dev-clean /data/LibriSpeech/test_transcriptions/dev.tsv
```

3. In order to start training, under the path examples/conformer/train.py, 
there's a script starting the training process. If user wishes to train with MWER training procedure, 
in config.yml under model_config there's a boolean mwer_training which, if set to True, 
starts the MWER training procedure. Otherwise it starts standard training procedure with regular RNN-T loss. 
train.py receives specific arguments for training. Most important are:
 
* --config a path to model config.yml file.
* --sentence_piece a flag whether to use sentence_piece as text tokenizer.
* --bs batch size.
* --devices which GPU devices are supposed to be used.

The rest of arguments are described in train.py file. Example of such command (that works under default setup) would be:

```
python examples/conformer/train.py --config examples/conformer/config.yml --sentence_piece --devices 0
```

## Test:
In order to start testing, under the path examples/conformer/test.py, 
there's a script starting the inference process. test.py receives specific arguments for training. 
Most important are:

* --saved a path to saved model.
* --config a path to model config.yml file.
* --sentence_piece a flag whether to use sentence_piece as text tokenizer.
* --bs batch size.
* --output path to output transcriptions.

Example of such command (that works under default setup) would be:
```
python ./examples/conformer/test.py --config ./examples/conformer/config.yml \
                                    --saved predefined_checkpoints/weights.hdf5 \
                                    --sentence_piece \
                                    --output test_result.tsv \
                                    --bs 1
```
