import os

# Dowhnload and unzip the data from https://github.com/mindsdb/lightwood/pull/119

def download_linux_osx():
    os.system('cd raw_data && wget https://github.com/OpenProtein/openprotein/blob/master/preprocessing.py; tar -xvf casp11.tar.gz')
