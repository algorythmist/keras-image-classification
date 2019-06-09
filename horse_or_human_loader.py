import os
import urllib.request
import zipfile

DATASETS_LOCATION = '.keras/datasets'
HOME = os.path.expanduser('~')


def get_path(filename):
    return os.path.join(HOME, DATASETS_LOCATION, filename)


def create_dataset_directory():
    dir = os.path.join(HOME, DATASETS_LOCATION)
    os.makedirs(dir)

def download_file(url, filename):
    create_dataset_directory()
    local_zip = get_path(filename)
    urllib.request.urlretrieve(url, local_zip)
    return local_zip


def unzip(file):
    local_zip = get_path(file)
    dir_name, _ = os.path.splitext(local_zip)
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall(dir_name)
    zip_ref.close()
    return dir_name


def prepare_dataset():
    directory = get_path('horse-or-human')
    if (os.path.exists(directory)):
        print(f'directory {directory} already exists')
        return directory
    zipname = directory+'.zip'
    if (not os.path.exists(zipname)):
        print('Downloading zip file...')
        download_file('https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip',
                      'horse-or-human.zip')
        print('download completed')
    print('unzipping file...')
    unzip('horse-or-human.zip')
    print('unzip completed')
    return directory



