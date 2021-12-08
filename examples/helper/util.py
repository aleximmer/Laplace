import urllib.request
import os.path


def download_pretrained_model():
    # Download pre-trained model if necessary
    if not os.path.isfile('CIFAR10_plain.pt'):
        if not os.path.exists('./temp'):
            os.makedirs('./temp')

        urllib.request.urlretrieve('https://nc.mlcloud.uni-tuebingen.de/index.php/s/2PBDYDsiotN76mq/download', './temp/CIFAR10_plain.pt')
