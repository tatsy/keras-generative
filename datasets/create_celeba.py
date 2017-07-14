import os
import sys
import re
import zipfile

import numpy as np
import h5py

import requests
from PIL import Image

google_drive_prefix = "https://docs.google.com/uc?export=download"
image_url = 'https://drive.google.com/open?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM'
attr_url = 'https://drive.google.com/open?id=0B7EVK8r0v71pblRyaVFSWGxPY0U'

outfile = 'celebB.hdf5'
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

    # Parse labels
    with open(attr_file, 'r') as lines:
        lines = [l.strip() for l in lines]
        num_images = int(lines[0])

        label_names = re.split('\s+', lines[1])
        label_names = np.array(label_names, dtype=object)
        num_labels = len(label_names)

        lines = lines[2:]
        labels = np.ndarray((num_images, num_labels), dtype='uint8')
        for i in range(num_images):
            label = [int(l) for l in re.split('\s+', lines[i])[1:]]
            label = np.maximum(0, label).astype(np.uint8)
            labels[i] = label

    ## Parse images
    with zipfile.ZipFile(image_file, 'r', zipfile.ZIP_DEFLATED) as zf:
        image_files = [f for f in zf.namelist()]
        image_files = sorted(image_files)
        image_files = list(filter(lambda f: f.endswith('.jpg'), image_files))

        num_images = len(image_files)
        print('%d images' % (num_images))

        image_data = np.ndarray((num_images, 64, 64, 3), dtype='uint8')
        for i, f in enumerate(image_files):
            image = Image.open(zf.open(f, 'r')).resize((64, 78), Image.ANTIALIAS).crop((0, 7, 64, 64 + 7))
            image = np.asarray(image, dtype='uint8')
            image_data[i] = image
            print('%d / %d' % (i + 1, num_images), end='\r', flush=True)

    # Create HDF5 file
    h5 = h5py.File(outfile, 'w')
    string_dt = h5py.special_dtype(vlen=str)
    dset = h5.create_dataset('images', data=image_data, dtype='uint8')
    dset = h5.create_dataset('label_names', data=label_names, dtype=string_dt)
    dset = h5.create_dataset('labels', data=labels, dtype='uint8')

    h5.flush()
    h5.close()

if __name__ == '__main__':
    main()
