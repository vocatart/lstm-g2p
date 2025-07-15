# lstm-g2p

reimplementation of [OpenUtau's LSTM G2P](https://github.com/stakira/OpenUtau/tree/master/py). mostly as a neural network, deployment, and pipeline exercise.

# installation

1. download the correct version of pytorch. the specific device i wrote this on used cpu pytorch and python 3.13.5 but cuda acceleration should work fine

2. install the requirements

```plaintext
$ pip install -r requirements.txt
```

# usage

change `g2p/data/config.yaml` to fit your needs and set `dict.tsv` to your desired dictionary. current setup is with cmudict-7.0b. at max epochs, all weights will be exported to `artifacts/run_name` including a simplified onnx graph for deployment.

# to do

1. inference example, pytorch + onnx

# references

[stakira/OpenUtau](https://github.com/stakira/OpenUtau)

[cmusphix/cmudict](https://github.com/cmusphinx/cmudict)
