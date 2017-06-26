import os
import sys
import re

import numpy as np
import h5py

import requests

google_drive_prefix = "https://docs.google.com/uc?export=download"
image_url = 'https://drive.google.com/open?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM'
attr_url = 'https://drive.google.com/open?id=0B7EVK8r0v71pblRyaVFSWGxPY0U'

outfile = 'celebA.hdf5'
image_file = 'img_align_celeba.zip'
attr_file = 'list_attr_celeba.txt'

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    PROGBAR_WIDTH = 50

    with open(destination, "wb") as f:
        dl = 0
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                dl += len(chunk)
                f.write(chunk)

                mb = dl / 1.0e6
                sys.stdout.write('\r%.2f MB downloaded...' % mb)
                sys.stdout.flush()

        sys.stdout.write('\nFinish!\n')
        sys.stdout.flush()

def download_from_google_drive(url, dest):
    pat = re.compile('id=([a-zA-Z0-9]+)')
    mat = pat.search(url)
    if mat is None:
        raise Exception('Invalide url:', url)

    idx = mat.group(1)

    session = requests.Session()

    response = session.get(google_drive_prefix, params={'id': idx}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': idx, 'confirm': token}
        response = session.get(google_drive_prefix, params=params, stream=True)

    print('Downloading:', url)
    save_response_content(response, dest)

def main():
    # Download image ZIP
    if os.path.exists(image_file):
        print('Image ZIP file exists. Skip downloading.')
    else:
        download_from_google_drive(image_url, image_file)

    # Download attribute file
    if os.path.exists(attr_file):
        print('Attribute file exists. Skip downloading.')
    else:
        download_from_google_drive(attr_url, attr_file)

    sys.exit(1)

    # Create HDF5 file
    with open(label_file, 'r') as lines:
        lines = [l.strip() for l in lines]
        num_images = int(lines[0])

        label_names = re.split('\s+', lines[1])
        label_names = np.array(label_names, dtype=object)
        num_labels = len(label_names)

        lines = lines[2:]
        labels = np.ndarray((10000, num_labels), np.uint8)
        for i in range(10000):
            label = [int(l) for l in re.split('\s+', lines[i])[1:]]
            label = np.maximum(0, label).astype(np.uint8)
            labels[i] = label


    h5 = h5py.File(outfile, 'a')
    string_dt = h5py.special_dtype(vlen=str)
    dset = h5.create_dataset('label_names', data=label_names, dtype=string_dt)
    dset = h5.create_dataset('labels', data=labels)

    h5.flush()
    h5.close()

if __name__ == '__main__':
    main()
