import os
import numpy
import torch
from torch.utils.data import Dataset, TensorDataset, ConcatDataset
import torchvision.datasets
import torchvision.transforms
from torchvision.transforms import ToTensor
from PIL import Image
import pathlib
import requests
import zipfile
import tarfile
'''
This file contains functions for loading (and if necessary downloading) the
relevant datasets.
'''

# Path to root folder for datasets
from workspace_path import home_path
root_folder = home_path / 'datasets'


def dataset_collector(dataset, split, **kwargs):
    '''
    Wrapper for all collectors, will use the datasets dict to see if a given
    dataset and split is available and return it using its specified collector
    Args:
        dataset (str): Key for the dataset in the datasets dict
        split (str): An available split for the given dataset
        **kwargs (dict): Any additional parameters
    Returns (torch.utils.data.Dataset)
    '''
    if dataset not in datasets:
        available_datasets = ', '.join(datasets.keys())
        raise ValueError(f'Unexpected value of dataset: {dataset}. '
                         f'Available datasets are {available_datasets}')
    dataset_info = datasets[dataset]
    if split not in dataset_info['split']:
        available_splits = ', '.join(dataset_info['split'])
        raise ValueError(
            f'Unexpected value of split: {split}. '
            f'Available splits for this dataset are {available_splits}')
    kwargs['dataset'] = dataset
    kwargs['split'] = split
    return dataset_info['source'](**kwargs)


def bapps_collector(split='train', subsplit='all', **kwargs):
    '''
    Function for collecting and, if necessary, downloading the BAPPS dataset
    Args:
        split (str): Which split to collect ('train', 'val', 'jnd/val')
        subsplit (str): Which subsplit to collect ('all' collects all)
        **kwargs (dict): Any additional parameters for collection
    Returns (torch.utils.data.Dataset)
    '''
    (root_folder/'BAPPS/train').mkdir(parents=True, exist_ok=True)
    (root_folder/'BAPPS/val').mkdir(parents=True, exist_ok=True)
    (root_folder/'BAPPS/jnd/val').mkdir(parents=True, exist_ok=True)
    if (
        len(os.listdir(root_folder/'BAPPS/train')) < 3
        or len(os.listdir(root_folder/'BAPPS/val')) < 6
        or len(os.listdir(root_folder/'BAPPS/jnd/val')) < 2
    ):
        if 'download' in kwargs and not kwargs['download']:
            raise FileNotFoundError(
                'Files are missing. Set \'download\' to True to automatically '
                'download them')
        print('Downloading BAPPS dataset...')
        dataset_link = 'https://people.eecs.berkeley.edu/~rich.zhang/projects/2018_perceptual/dataset'
        dataset_parts = ['twoafc_train', 'twoafc_val', 'jnd']
        for part in dataset_parts:
            filename = root_folder/f'BAPPS/{part}.tar.gz'
            if not os.path.isfile(filename):
                download_raw_url(url=f'{dataset_link}/{part}.tar.gz',
                                 save_path=filename)
            with tarfile.TarFile(filename, 'r') as tar_ref:
                tar_ref.extractall(root_folder/'BAPPS')
    subsplits = os.listdir(root_folder/f'BAPPS/{split}')
    if subsplit == 'all':
        ret = []
        for subsplit in subsplits:
            dirs = os.listdir(root_folder/f'BAPPS/{split}/{subsplit}')
            paths = [root_folder/f'BAPPS/{split}/{subsplit}/{d}' for d in dirs]
            ret.append(MultipleFolderDataset(
                *paths, name=subsplit,
                image_transform=kwargs.get('image_transform'))
            )
        return ConcatDataset(ret)
    elif subsplit in subsplits:
        dirs = os.listdir(root_folder/f'BAPPS/{split}/{subsplit}')
        paths = [root_folder/f'BAPPS/{split}/{subsplit}/{d}' for d in dirs]
        return MultipleFolderDataset(
            *paths, name=subsplit,
            image_transform=kwargs.get('image_transform')
        )
    else:
        raise ValueError(
            f'Unexpected value of subsplit: {subsplit}. '
            f'Expected any of: all, {", ".join(subsplits+["all"])}'
        )


# Dictionary of available datasets and their attributes and parameters
datasets = {
    'BAPPS': {
        'full_name': 'Berkeley Adobe Perceptual Patch Similarity',
        'source': bapps_collector,
        'downloadable': True,
        'split': ['train', 'val', 'jnd/val'],
        'output_format': None  # TODO: annotate the format (eg 'onehot')
    }
}


def download_raw_url(url, save_path, show=False, chunk_size=128, decode=False):
    '''
    Downloads raw data from url. Reworked from:
    https://stackoverflow.com/questions/9419162/download-returned-zip-file-from-url
    Args:
        url (str): Url of raw data
        save_path (str): File name to store data under
        show (bool): Whether to print what is being downloaded
        chunk_size (int): How large chunks of data to collect at a time
    '''
    if show:
        print(f'\rDownloading URL: {url}', end='')
    r = requests.get(url, stream=True)
    if decode:
        r.raw.decode_content = True
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)


class MultipleFolderDataset(Dataset):
    '''
    A dataset for loading data where data is contained in folders where each
    matching group of data has the same name (with possibly different
    file-endings)
    Allowed file types: .png, .tif, .tiff, .jpg, .jpeg, .bmp, .npy 
    Args:
        *args (str): Paths to the folders to extract from
        name (str): A name to be returned together with each datapoint
        image_transform (nn.Module): Transform applied when getting images
    '''
    def __init__(self, *args, name=None, image_transform=None):
        super().__init__()
        if len(args) < 1:
            raise RuntimeError('Must be given at least one path')
        self.name = name
        acceptable_endings = [
            'png', 'tif', 'tiff', 'jpg', 'jpeg', 'bmp', 'npy'
        ]
        folder_files = []
        for folder in args:
            files = os.listdir(folder)
            folder_files.append({
                f[:f.index('.')]: f[f.index('.') + 1:]
                for f in files if f[f.index('.') + 1:] in acceptable_endings
            })
        self.data_paths = []
        for filename, ending in folder_files[0].items():
            paths = [f'{args[0]}/{filename}.{ending}']
            for folder, arg in zip(folder_files[1:], args[1:]):
                if filename in folder:
                    paths.append(f'{arg}/{filename}.{folder[filename]}')
                else:
                    break
            if len(paths) != len(args):
                continue
            self.data_paths.append(paths)
        self.image_transform = image_transform
        if self.image_transform is None:
            self.image_transform = ToTensor()


    def __getitem__(self, index):
        image_endings = ['png', 'tif', 'tiff', 'jpg', 'jpeg', 'bmp']
        npy_endings = ['npy']

        ret = {} #= []
        for path in self.data_paths[index]:
            ending = path[path.index('.') + 1:]
            folder = path.split('/')[-2]
            if ending in image_endings:
                image = Image.open(path).convert(mode='RGB')
                ret[folder] = self.image_transform(image)
            elif ending in npy_endings:
                ret[folder] = torch.from_numpy(numpy.load(path))
            else:
                raise RuntimeError('Loading from unsupported file type')
        if not self.name is None:
            ret['name'] = self.name
        return ret

    def __len__(self):
        return len(self.data_paths)