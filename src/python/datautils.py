import numpy as np
import pickle
import os
from PIL import Image
import PIL
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Union, Optional
from src.python.tensor import FloatTensor, LNSTensor, batch_concat


def random_flip(img, horizontal=True):
    """Randomly flip the image."""
    if np.random.rand() < 0.5:
        if horizontal:
            img = img[:, ::-1, :]
        else:
            img = img[::-1, :, :]

    return img


def random_crop(img, crop_size, padding=4):
    """Randomly crop the image."""

    # pad the border
    img = np.pad(img, ((padding, padding), (padding, padding), (0, 0)), mode="constant")

    h, w = img.shape[:2]
    if crop_size[0] > h or crop_size[1] > w:
        raise ValueError(
            "Requested crop size {} is bigger than input size {}".format(
                crop_size, (h, w)
            )
        )

    top = np.random.randint(0, h - crop_size[0] + 1)
    left = np.random.randint(0, w - crop_size[1] + 1)

    bottom = top + crop_size[0]
    right = left + crop_size[1]

    img = img[top:bottom, left:right, :]

    return img


def center_crop(img, crop_size):
    """Center crop the image."""

    h, w = img.shape[:2]
    if crop_size[0] > h or crop_size[1] > w:
        raise ValueError(
            "Requested crop size {} is bigger than input size {}".format(
                crop_size, (h, w)
            )
        )

    top = (h - crop_size[0]) // 2
    left = (w - crop_size[1]) // 2

    bottom = top + crop_size[0]
    right = left + crop_size[1]

    img = img[top:bottom, left:right, :]

    return img


class Dataset(ABC):
    @abstractmethod
    def __len__(self):
        """Returns the number of data points."""
        pass

    @abstractmethod
    def __getitem__(self, idx):
        """Given an index, returns the corresponding data point."""
        pass


class ImageNet(Dataset):
    def __init__(self, root_dir: str, kind: str, arithmetic: str) -> None:
        self.root_dir = root_dir
        self.kind = kind
        self.arithmetic = arithmetic
        self.nClasses = 1000
        self.crop_size = 224
        self.padding = 4
        self.raw_img_shape = (256, 256, 3)
        self.img_shape = (self.crop_size, self.crop_size, self.raw_img_shape[2])
        self.data = []  # idx to tuple (image_file, label)

        self.lablname_to_label = {}

        if self.kind == "train":
            idx = 0
            train_dir = os.path.join(self.root_dir, "train")
            for class_dir in sorted(os.listdir(train_dir)):
                for filename in os.listdir(os.path.join(train_dir, class_dir)):
                    img_file = os.path.join(train_dir, class_dir, filename)
                    if class_dir not in self.lablname_to_label:
                        self.lablname_to_label[class_dir] = idx
                        idx += 1
                    self.data.append((img_file, self.lablname_to_label[class_dir]))
        elif self.kind == "val":
            idx = 0
            val_dir = os.path.join(self.root_dir, "val")
            for class_dir in sorted(os.listdir(val_dir)):
                for filename in os.listdir(os.path.join(val_dir, class_dir)):
                    img_file = os.path.join(val_dir, class_dir, filename)
                    if class_dir not in self.lablname_to_label:
                        self.lablname_to_label[class_dir] = idx
                        idx += 1
                    self.data.append((img_file, self.lablname_to_label[class_dir]))

        else:
            raise ValueError('Invalid split "%s" for ImageNet' % self.kind)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        assert idx < len(self), "index out of range"

        img_file, label = self.data[idx]

        img = Image.open(img_file)

        # crop to self.crop_size x self.crop_size
        aspect_ratio = img.width / img.height
        if img.width < img.height:
            new_width = 256
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = 256
            new_width = int(new_height * aspect_ratio)

        img_resized = img.resize((new_width, new_height), Image.ANTIALIAS)
        left = (new_width - 224) / 2
        top = (new_height - 224) / 2
        right = (new_width + 224) / 2
        bottom = (new_height + 224) / 2

        img_cropped = img_resized.crop((left, top, right, bottom))
        img_cropped = np.asarray(img_cropped, dtype=np.float32)

        if img_cropped.ndim == 2 or img_cropped.shape[2] == 1:
            ## grayscale file -> convert to rgb (replicate grayscale channel 3 times)
            if img_cropped.ndim == 3:
                img_cropped = np.squeeze(img_cropped, axis=-1)
            img_cropped = np.stack((img_cropped,) * 3, axis=-1)

        label = OneHotEncode(label, self.nClasses)

        # normalize by mean and std of imagenet training set
        img_cropped = img_cropped / 255.0
        img_cropped = (img_cropped - np.array([0.485, 0.456, 0.406])) / np.array(
            [0.229, 0.224, 0.225]
        )

        if self.kind == "train":
            # random crop
            img_cropped = random_crop(img_cropped, crop_size=(224, 224), padding=4)
            # random flip
            img_cropped = random_flip(img_cropped, horizontal=True)

        img_cropped = FloatTensor(np.asarray(img_cropped, dtype=np.float32))
        label = FloatTensor(label)

        if self.arithmetic == "lns":
            img_cropped = img_cropped.to_lns()
            label = label.to_lns()

        return img_cropped, label


class TinyImageNet(Dataset):
    def __init__(self, root_dir: str, kind: str, arithmetic: str) -> None:
        """
        data source: http://cs231n.stanford.edu/tiny-imagenet-200.zip
        code source: https://github.com/yunjey/cs231n/blob/master/assignment1/cs231n/data_utils.py

        Load TinyImageNet. Each of TinyImageNet-100-A, TinyImageNet-100-B, and
        TinyImageNet-200 have the same directory structure, so this can be used
        to load any of them.

        Inputs:

        Returns: A tuple of
        - class_names: A list where class_names[i] is a list of strings giving the
        WordNet names for class i in the loaded dataset.
        """
        self.root_dir = root_dir
        self.kind = kind
        self.arithmetic = arithmetic
        self.nClasses = 200
        self.crop_size = 56
        self.padding = 4
        self.raw_img_shape = (64, 64, 3)
        self.img_shape = (self.crop_size, self.crop_size, self.raw_img_shape[2])
        self.data = []  # idx to tuple (image_file, label)

        # First load wnids
        with open(os.path.join(self.root_dir, "wnids.txt"), "r") as f:
            wnids = [x.strip() for x in f]
            wnids = sorted(wnids)

        # Map wnids to integer labels
        wnid_to_label = {wnid: i for i, wnid in enumerate(wnids)}

        # Use words.txt to get names for each class
        with open(os.path.join(self.root_dir, "words.txt"), "r") as f:
            wnid_to_words = dict(line.split("\t") for line in f)
        for wnid, words in wnid_to_words.items():
            wnid_to_words[wnid] = [w.strip() for w in words.split(",")]
        self.class_names = [wnid_to_words[wnid] for wnid in wnids]

        if self.kind == "train":
            for i, wnid in enumerate(wnids):
                # To figure out the filenames we need to open the boxes file
                boxes_file = os.path.join(
                    self.root_dir, "train", wnid, "%s_boxes.txt" % wnid
                )
                with open(boxes_file, "r") as f:
                    filenames = [x.split("\t")[0] for x in f]
                for j, img_file in enumerate(filenames):
                    img_file = os.path.join(
                        self.root_dir, "train", wnid, "images", img_file
                    )
                    self.data.append((img_file, wnid_to_label[wnid]))
        elif self.kind == "val":
            with open(
                os.path.join(self.root_dir, "val", "val_annotations.txt"), "r"
            ) as f:
                for line in f:
                    img_file, wnid = line.split("\t")[:2]
                    img_file = os.path.join(self.root_dir, "val", "images", img_file)
                    self.data.append((img_file, wnid_to_label[wnid]))
        else:
            raise ValueError('Invalid split "%s" for TinyImageNet' % self.kind)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        """
        Load TinyImageNet. Each of TinyImageNet-100-A, TinyImageNet-100-B, and
        TinyImageNet-200 have the same directory structure, so this can be used
        to load any of them.

        Inputs:
        - path: String giving path to the directory to load.
        - dtype: numpy datatype used to load the data.

        Returns: A tuple of
        - class_names: A list where class_names[i] is a list of strings giving the
        WordNet names for class i in the loaded dataset.
        - X_train: (N_tr, 3, 64, 64) array of training images
        - y_train: (N_tr,) array of training labels
        - X_val: (N_val, 3, 64, 64) array of validation images
        - y_val: (N_val,) array of validation labels
        """

        assert idx < len(self), "index out of range"

        img_file, label = self.data[idx]

        img = np.asarray(Image.open(img_file), dtype=np.float32)
        if img.ndim == 2 or img.shape[2] == 1:
            ## grayscale file -> convert to rgb (replicate grayscale channel 3 times)
            if img.ndim == 3:
                img = np.squeeze(img, axis=-1)
            img = np.stack((img,) * 3, axis=-1)

        label = OneHotEncode(label, self.nClasses)

        # normalize by mean and std of tinyimagenet training set
        img = img / 255.0
        img = (img - np.array([0.4802, 0.4481, 0.3975])) / np.array(
            [0.2764, 0.2689, 0.2816]
        )

        if self.kind == "train":
            # random flip
            img = random_flip(img, horizontal=True)
            # random crop
            img = random_crop(
                img, crop_size=(self.crop_size, self.crop_size), padding=self.padding
            )

        else:
            # center crop
            img = center_crop(img, crop_size=(self.crop_size, self.crop_size))

        img = FloatTensor(img)
        label = FloatTensor(label)

        if self.arithmetic == "lns":
            img = img.to_lns()
            label = label.to_lns()

        return img, label


class Cifar10(Dataset):
    def __init__(self, root_dir: str, kind: str, arithmetic: str) -> None:
        """
        source: https://www.cs.toronto.edu/~kriz/cifar.html
        """
        self.root_dir = root_dir
        self.kind = kind
        self.arithmetic = arithmetic
        self.nClasses = 10
        self.crop_size = 32
        self.padding = 4
        self.raw_img_shape = (32, 32, 3)
        self.img_shape = (self.crop_size, self.crop_size, self.raw_img_shape[2])
        self.class_names = [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]
        self.data = []  # idx to tuple (image_file, label)
        if self.kind == "train":
            for batch_id in range(1, 6):  # CIFAR-10 has 5 batches
                features, labels = load_cifar10_batch(root_dir, batch_id)
                for i, (image, label) in enumerate(zip(features, labels)):
                    image_id = (batch_id - 1) * 10000 + i
                    label_name = self.class_names[label]
                    image_path = save_image(image, image_id, label_name, root_dir, kind)
                    self.data.append((image_path, label))
        elif self.kind == "test":
            features, labels = load_cifar10_batch(root_dir, 0, test_batch=True)
            for i, (image, label) in enumerate(zip(features, labels)):
                image_id = i
                label_name = self.class_names[label]
                image_path = save_image(image, image_id, label_name, root_dir, kind)
                self.data.append((image_path, label))
        else:
            raise ValueError('Invalid split "%s" for Cifar10' % self.kind)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        """ """
        assert idx < len(self), "index out of range"

        img_file, label = self.data[idx]

        img = np.asarray(Image.open(img_file), dtype=np.float32)
        label = OneHotEncode(label, self.nClasses)

        # normalize by mean and std of cifar10 training set
        img = img / 255.0
        img = (img - np.array([0.4914, 0.4822, 0.4465])) / np.array(
            [0.247, 0.243, 0.261]
        )

        if self.kind == "train":
            # random flip
            img = random_flip(img, horizontal=True)
            # random crop
            img = random_crop(
                img, crop_size=(self.crop_size, self.crop_size), padding=self.padding
            )

        img = FloatTensor(img)
        label = FloatTensor(label)

        if self.arithmetic == "lns":
            img = img.to_lns()
            label = label.to_lns()

        return img, label


class Cifar100(Dataset):
    def __init__(self, root_dir: str, kind: str, arithmetic: str) -> None:
        """
        source: https://www.cs.toronto.edu/~kriz/cifar.html
        """
        self.root_dir = root_dir
        self.kind = kind
        self.arithmetic = arithmetic
        self.nClasses = 100
        self.crop_size = 32
        self.padding = 4
        self.raw_img_shape = (32, 32, 3)
        self.img_shape = (self.crop_size, self.crop_size, self.raw_img_shape[2])
        self.data = []  # idx to tuple (image_file, label)
        if self.kind == "train":
            features, labels = load_cifar100(root_dir, kind)
            for i, (image, label) in enumerate(zip(features, labels)):
                image_id = i
                label_name = str(label)
                image_path = save_image(image, image_id, label_name, root_dir, kind)
                self.data.append((image_path, label))
        elif self.kind == "test":
            features, labels = load_cifar100(root_dir, kind)
            for i, (image, label) in enumerate(zip(features, labels)):
                image_id = i
                label_name = str(label)
                image_path = save_image(image, image_id, label_name, root_dir, kind)
                self.data.append((image_path, label))
        else:
            raise ValueError('Invalid split "%s" for Cifar100' % self.kind)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        """ """
        assert idx < len(self), "index out of range"

        img_file, label = self.data[idx]

        img = np.asarray(Image.open(img_file), dtype=np.float32)
        label = OneHotEncode(label, self.nClasses)

        # normalize by mean and std of cifar100 training set
        img = img / 255.0
        img = (img - np.array([0.5071, 0.4867, 0.4408])) / np.array(
            [0.2675, 0.2565, 0.2761]
        )

        if self.kind == "train":
            # random flip
            img = random_flip(img, horizontal=True)
            # random crop
            img = random_crop(
                img, crop_size=(self.crop_size, self.crop_size), padding=self.padding
            )

        img = FloatTensor(img)
        label = FloatTensor(label)

        if self.arithmetic == "lns":
            img = img.to_lns()
            label = label.to_lns()

        return img, label


class DataLoader:
    def __init__(
        self, dataset: Dataset, batch_size: int = 1, shuffle: bool = True
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Create an index list. This will determine the order in which data is fetched.
        self.indices = list(range(len(dataset)))

    def __iter__(self) -> List[Tuple[np.ndarray, int]]:
        # If shuffle flag is on, shuffle the indices
        if self.shuffle:
            np.random.shuffle(self.indices)  # in-place shuffle

        for i in range(0, len(self.indices), self.batch_size):
            # Fetch data in chunks of batch size
            images = batch_concat(
                [self.dataset[idx][0] for idx in self.indices[i : i + self.batch_size]]
            )
            labels = batch_concat(
                [self.dataset[idx][1] for idx in self.indices[i : i + self.batch_size]]
            )
            yield images, labels

    def __len__(self) -> int:
        return (
            len(self.indices) + self.batch_size - 1
        ) // self.batch_size  # Ceiling division


def OneHotEncode(y, M):
    """
    Perform One Hot encoding of Labels.
    Inputs:
        - y: int
        - M: Number of Classes [integer scalar]
    Outputs:
        - yhot: one hot encoded labels [1D array of shape (M)]
    """
    yhot = np.zeros(M, dtype=np.int32)
    yhot[y] = 1
    return yhot


def load_mnist(path, kind="train"):
    import os
    import gzip

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, "%s-labels-idx1-ubyte.gz" % kind)
    images_path = os.path.join(path, "%s-images-idx3-ubyte.gz" % kind)

    with gzip.open(labels_path, "rb") as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, "rb") as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(
            len(labels), 784
        )

    return images, labels


# Load a single batch from CIFAR-10 dataset
def load_cifar10_batch(cifar10_dataset_folder_path, batch_id, test_batch=False):
    if test_batch:
        with open(
            os.path.join(cifar10_dataset_folder_path, "test_batch"), mode="rb"
        ) as file:
            batch = pickle.load(file, encoding="latin1")
    else:
        with open(
            os.path.join(cifar10_dataset_folder_path, f"data_batch_{batch_id}"),
            mode="rb",
        ) as file:
            batch = pickle.load(file, encoding="latin1")
    features = (
        batch["data"].reshape((len(batch["data"]), 3, 32, 32)).transpose(0, 2, 3, 1)
    )
    labels = batch["labels"]
    return features, labels


def load_cifar100(cifar100_dataset_folder_path, kind="train"):
    if kind == "train":
        with open(
            os.path.join(cifar100_dataset_folder_path, "train_raw"), mode="rb"
        ) as file:
            batch = pickle.load(file, encoding="latin1")
    elif kind == "test":
        with open(
            os.path.join(cifar100_dataset_folder_path, "test_raw"), mode="rb"
        ) as file:
            batch = pickle.load(file, encoding="latin1")
    else:
        raise ValueError('Invalid split "%s" for Cifar100' % kind)
    features = (
        batch["data"].reshape((len(batch["data"]), 3, 32, 32)).transpose(0, 2, 3, 1)
    )
    labels = batch["fine_labels"]
    return features, labels


# Save image to disk
def save_image(image, image_id, label, output_folder, kind):
    image_folder = os.path.join(output_folder, kind, label)
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
    image_path = os.path.join(image_folder, f"{image_id}.png")
    if not os.path.exists(image_path):
        image_pil = PIL.Image.fromarray(image)
        image_pil.save(image_path)
    return image_path


def load_dataset(dataset, arithmetic: str):
    if dataset == "fmnist":
        nClasses = 10
        # x is (60000,28,28), y is (60000,)
        (x_train, y_train) = load_mnist("data/fashion", "train")
        x_train = x_train.reshape(-1, 28, 28)
        (x_test, y_test) = load_mnist("data/fashion", "t10k")
        x_test = x_test.reshape(-1, 28, 28)
        x_train = x_train.transpose((1, 2, 0)).astype("float32")
        x_test = x_test.transpose((1, 2, 0)).astype("float32")
        x_train = np.expand_dims(x_train, axis=2)
        x_test = np.expand_dims(x_test, axis=2)
        x_train = x_train / 255
        x_test = x_test / 255

    elif dataset == "cifar10":
        nClasses = 10
        train_dataset = Cifar10("data/cifar-10-batches-py", "train", arithmetic)
        val_dataset = Cifar10("data/cifar-10-batches-py", "test", arithmetic)
        test_dataset = None

    elif dataset == "cifar100":
        nClasses = 100
        train_dataset = Cifar100("data/cifar-100-python", "train", arithmetic)
        val_dataset = Cifar100("data/cifar-100-python", "test", arithmetic)
        test_dataset = None

    elif dataset == "tinyimagenet":
        nClasses = 200
        train_dataset = TinyImageNet(
            os.path.join("data", "tiny-imagenet-200"), "train", arithmetic
        )
        val_dataset = TinyImageNet(
            os.path.join("data", "tiny-imagenet-200"), "val", arithmetic
        )
        test_dataset = None

    elif dataset == "imagenet":
        nClasses = 1000
        train_dataset = ImageNet(
            os.path.join("/", "m2", "data", "imagenet"), "train", arithmetic
        )
        val_dataset = ImageNet(
            os.path.join("/", "m2", "data", "imagenet"), "val", arithmetic
        )
        test_dataset = None

    return train_dataset, val_dataset, test_dataset, nClasses
