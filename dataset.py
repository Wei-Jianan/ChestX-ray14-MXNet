import fnmatch
import os
import zipfile
from collections import OrderedDict
import matplotlib.pyplot as plt
import pandas as pd
from mxnet.gluon.data import Dataset, DataLoader
from mxnet import nd, image

# whole_url = ''
# demo_url = 'https://drive.google.com/uc?export=download&id=12fbMgiEW0MuK85wXOWpZ4n_36llqz6Zw'

_labels = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
           'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
           'Pleural_Thickening', 'Hernia']
_label2order_table = OrderedDict((label, order) for order, label in enumerate(_labels))
num_labels = len(_label2order_table)


def upzip_and_delete() -> str:
    """
    unzip the zip file and delete it
    :return: path to the extracted folder
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    zip_files_path = []
    for root_path, dir_names, file_names in os.walk(dir_path):
        for file_name in fnmatch.filter(file_names, '*.zip'):
            zip_files_path.append(os.path.join(root_path, file_name))
    assert len(zip_files_path) == 1, 'there should be one zip file and only one zip inside current director.'
    to_path = dir_path
    with zipfile.ZipFile(zip_files_path[0], "r") as zip_ref:
        assert zip_ref.namelist()[0] == 'ChestX-ray14/', "zip file only contains my directory."
        extracted_dir_name = zip_ref.namelist()[0]
        zip_ref.extractall(to_path)

    # TODO leave the demo mode and delete unfinished.

    return extracted_dir_name


def _get_data_entry(root=None) -> pd.DataFrame:
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
                 transform=lambda X, y: (X.astype('float32'), y.astype('float32'))):
        """
        Dataset that could be randomly accessed storing all labeled ChestX-ray images.

        :param data_entry: pandas.DataFrame read the data_entry.csv file from the Chestx_ray14 file
        :param root:
        :param train: True for training dataset
        :param transform: lazy transforming the data when accessed
        """
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
        # print('image name:\t', image_name)
        label = get_label_vector(self._data_entry, pic_name=image_name)
        if self._transform is not None:
            return self._transform(raw_image, label)
        return raw_image, label

    def __len__(self):
        return len(self._image_names)

    def download_and_unzip(self, root):
        raise NotImplemented


def load_data_ChestX_ray14(batch_size, resize=512, root=None):
    """
    :param batch_size:
    :param resize: what size to changed, the raw size is 1024 * 1024
    :param root:
    :return:   (training data iterator, testing data iterator)
                batch iterators that return (resized images, label vectors) when called __next__().

    """
    def transform_mnist(data, label):
        # Transform an example.
        if resize:
            # n = data.shape[0]
            # new_data = nd.zeros((resize, resize, data.shape[2]))
            new_data = image.imresize(data, resize, resize)
            data = new_data

        # change data from height x width x channel to channel x height x width
        return nd.transpose(data.astype('float32'), (2, 0, 1)) / 255, label.astype('float32')

    data_entry = _get_data_entry()
    chestX_ray_train = ChestXRay14Dataset(data_entry, root=root, train=True, transform=transform_mnist)
    num_train = len(chestX_ray_train)
    chestX_ray_test = ChestXRay14Dataset(data_entry, root=root, train=False, transform=transform_mnist)
    num_test = len(chestX_ray_test)

    sampler = None  # TODO random sampler into the DataLoader when network is fine.
    train_data = DataLoader(chestX_ray_train, batch_size, )
    test_data = DataLoader(chestX_ray_test, batch_size)
    return train_data, test_data


if __name__ == '__main__':
    data_entry = _get_data_entry()
    chestX_ray_train = ChestXRay14Dataset(data_entry, train=True, transform=None)
    chestX_ray_test = ChestXRay14Dataset(data_entry, train=False, transform=None)
    # print(chestX_ray_train[1])


    # train_iter, test_iter = load_data_ChestX_ray14(10)
    # for data, label in train_iter:
    #     plt.imshow(nd.transpose(data[0], (1, 2, 0)).asnumpy())
    #     plt.show()
    #     print(label_vector2label_str(label[1]))

    im, ls = chestX_ray_train[1]
    plt.imshow(im.asnumpy())
    plt.show()
    print(im.dtype)
