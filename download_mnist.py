import os
import numpy
from six.moves import urllib
import gzip
import shutil

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
def maybe_download(filename, work_directory):
    """Download the data from Yann's website, unless it's already here."""
    if not os.path.exists(work_directory):
        os.mkdir(work_directory)
    filepath = os.path.join(work_directory, filename)
    if not os.path.exists(filepath):
        filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    return filepath

TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

def downloadMNIST(dst):
    print("Downloading MNIST data from ", SOURCE_URL, " if the data is not already present")
    filepath = maybe_download(TRAIN_IMAGES, dst)
    if not os.path.exists(filepath[:-3]):
        print("Extracting files from ", filepath)
        with gzip.open(filepath, 'r') as f_in:
            with open(filepath[:-3], 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    filepath = maybe_download(TRAIN_LABELS, dst)
    if not os.path.exists(filepath[:-3]):
        print("Extracting files from ", filepath)
        with gzip.open(filepath, 'r') as f_in:
            with open(filepath[:-3], 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    filepath = maybe_download(TEST_IMAGES, dst)
    if not os.path.exists(filepath[:-3]):
        print("Extracting files from ", filepath)
        with gzip.open(filepath, 'r') as f_in:
            with open(filepath[:-3], 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    filepath = maybe_download(TEST_LABELS, dst)
    if not os.path.exists(filepath[:-3]):
        print("Extracting files from ", filepath)
        with gzip.open(filepath, 'r') as f_in:
            with open(filepath[:-3], 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
