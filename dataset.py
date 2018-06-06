import os
from collections import OrderedDict

import matplotlib.pyplot as plt
import pandas as pd

from mxnet.gluon.data import Dataset
from mxnet import nd, image

whole_url = ''
demo_url = 'https://drive.google.com/uc?export=download&id=12fbMgiEW0MuK85wXOWpZ4n_36llqz6Zw'

# dir_path = os.path.dirname(os.path.realpath(__file__))
# data_dir_path = os.path.join(dir_path, 'ChestX-ray14/')
# images_dir_path = os.path.join(data_dir_path, 'images')
# train_list_path = os.path.join(dir_path, 'train_val_list.txt')
# test_list_path = os.path.join(dir_path, 'test_list.txt')
# data_entry_path = os.path.join(data_dir_path, 'Data_Entry_2017.csv')

_labels = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
           'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
           'Pleural_Thickening', 'Hernia']
_label2order_table = OrderedDict((label, order) for order, label in enumerate(_labels))
num_labels = len(_label2order_table)


def get_data_entry(root=None) -> pd.DataFrame:
    if root is not None:
        root = os.path.expanduser(root)
    else:
        root = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            'ChestX-ray14/')
    data_entry_path = os.path.join(root, 'Data_Entry_2017.csv')
    df = pd.read_csv(data_entry_path, dtype=str)
    df = df.drop(columns=df.columns[2:].tolist())
    df.columns = ['Index', 'Labels']
    df = df.set_index('Index')
    return df


def label_vector2label_str(label_vector: nd.NDArray) -> str:
    label_str = ''
    for indicator, label in zip(label_vector, _labels):
        if indicator == 1:
            label_str += label + ', '
    return label_str if label_str != '' else 'Not_finding'


class ChestXRay14Dataset(Dataset):

    def __init__(self, data_entry: pd.DataFrame, root=None, train=True,
                 transform=lambda X, y: (X.astype('float32') / 255, y.astype('float32'))):
        super(ChestXRay14Dataset, self).__init__()
        self._transform = transform
        self._data_entry = data_entry
        if root is not None:
            root = os.path.expanduser(root)
        else:
            root = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                'ChestX-ray14/')
        self._root = root

        if not os.path.isdir(self._root):
            os.makedirs(root)
            self.download_and_unzip(self._root)

        self._image_dir_path = os.path.join(self._root, 'images')

        if train:
            data_list_path = os.path.join(self._root, 'train_val_list.txt')
        else:
            data_list_path = os.path.join(self._root, 'test_list.txt')
        with open(data_list_path) as f:
            self._image_names = f.readlines()
            self._image_names = [name.strip() for name in self._image_names]

        if not os.path.isdir(self._root):
            os.makedirs(root)
            self.download_and_unzip(self._root)

    def __getitem__(self, idx):
        def get_label_vector(data_entry: pd.DataFrame, pic_name):
            label_vector = nd.zeros(shape=(num_labels,), dtype='float32')
            mixed_labels = data_entry.loc[pic_name, 'Labels']
            labels = mixed_labels.strip().split('|')
            for label in labels:
                order = _label2order_table.get(label, None)
                if order is None:
                    break
                label_vector[order] = 1

            return label_vector

        image_name = self._image_names[idx]
        raw_image = image.imread(os.path.join(self._image_dir_path,
                                              image_name))
        print('image name:\t', image_name)
        label = get_label_vector(self._data_entry, pic_name=image_name)
        if self._transform is not None:
            return self._transform(raw_image, label)
        return raw_image, label

    def __len__(self):
        return len(self._image_names)

    def download_and_unzip(self, root):
        raise NotImplemented


if __name__ == '__main__':
    data_entry = get_data_entry()
    chestX_ray14_train = ChestXRay14Dataset(data_entry, train=False )
    image, label = chestX_ray14_train[3]
    print(label)
    print(label_vector2label_str(label))
    print(image)
    plt.imshow(image)
    plt.show()
