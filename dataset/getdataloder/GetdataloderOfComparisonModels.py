# coding=utf-8
import numpy as np
from dataaug.augmentation import get_train_transform_2D
from scipy.ndimage import zoom
from torch.utils.data import Dataset
import pickle
import random

def load_from_pkl(load_path):
    data_input = open(load_path, 'rb')
    read_data = pickle.load(data_input)
    data_input.close()
    return read_data

def resize2D(img, aimsize, order = 3):
    """
    :param img: 3D array
    :param aimsize: list, one or three elements, like [256], or [256,56,56]
    :return:
    """
    _shape =img.shape
    if len(aimsize)==1:
        aimsize = [aimsize[0] for _ in range(2)]
    return zoom(img, (aimsize[0] / _shape[0], aimsize[1] / _shape[1]), order=order)  # resample for cube_size

class Dataset_train(Dataset):
    def __init__(self,data,augmentation_prob,aug,img_size, use_sammask = False):
        self.data = data
        self.augmentation_prob = augmentation_prob
        print(f'{augmentation_prob}概率进行扩增')
        self.aug = aug
        transforms = get_train_transform_2D((img_size, img_size))  #
        self.transforms = transforms['train']
        self.use_sammask = use_sammask
        self.img_size = img_size
        self.transforms_v = get_train_transform_2D((img_size, img_size))['val']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        p_transform = random.random()
        mask = np.zeros((1))
        i = 0
        #如果mask全0就重新选
        while mask.sum() == 0:
            if i > 0: index = random.randint(0, len(self.data)-1)
            data1 = self.data[index][0]
            data1 = data1.astype(np.float32)
            bbox = np.array(self.data[index][5])
            if self.use_sammask:
                mask = np.array(self.data[index][6])
            else:
                mask = np.zeros_like(data1)
                for ii in range(bbox.shape[0]):
                    bbox_one = bbox[ii]
                    x_min,x_max = np.min(bbox_one[:,0]),np.max(bbox_one[:,0])
                    y_min, y_max = np.min(bbox_one[:, 1]), np.max(bbox_one[:, 1])
                    if x_min == x_max or y_min == y_max:
                       continue
                    else:
                        mask[y_min:y_max+1,x_min:x_max+1] = 1.0
            i = i+1

        if self.aug:
            if self.augmentation_prob > p_transform:
                data_dict = dict(data=data1[np.newaxis, np.newaxis, :, :], seg=mask[np.newaxis, np.newaxis, :, :]) #resize
                augmented = self.transforms(**data_dict)
                data1 = np.array(augmented.get("data")[0, 0, :, :])
                mask = np.array(augmented.get("seg")[0, 0, :, :])
        if self.transforms_v is not None:
            data_dict = dict(data=data1[np.newaxis, np.newaxis, :, :], seg=mask[np.newaxis, np.newaxis, :, :]) #resize
            augmented = self.transforms_v(**data_dict)
            data1 = np.array(augmented.get("data")[0, 0, :, :])
            mask = np.array(augmented.get("seg")[0, 0, :, :])
        label = self.data[index][1]
        _ = self.data[index][2]
        data_num = self.data[index][3]
        main_label = self.data[index][4]

        index_ = np.where(mask > 0)
        minx, maxx = np.min(index_[1]), np.max(index_[1])
        miny, maxy = np.min(index_[0]), np.max(index_[0])
        small_img = data1[np.clip(miny - 5, 0, data1.shape[0] - 1):np.clip(maxy + 5 + 1, 0, data1.shape[0]),
                    np.clip(minx - 5, 0, data1.shape[1] - 1):np.clip(minx + 5 + 1, 0, data1.shape[1])]
        data_dict = dict(data=small_img[np.newaxis, np.newaxis, :, :])  # resize
        augmented = self.transforms_v(**data_dict)
        small_img = np.array(augmented.get("data")[0, 0, :, :])

        return data1,label, _,data_num,main_label,mask,small_img

class Dataset_vail(Dataset):
    def __init__(self, data,img_size,use_sammask = False):
        self.data = data
        self.use_sammask = use_sammask
        self.img_size = img_size
        self.transforms_v = get_train_transform_2D((self.img_size, self.img_size))['val']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data1 = self.data[index][0].astype(np.float32)

        label = self.data[index][1]
        _ = self.data[index][2]
        data_num = self.data[index][3]
        main_label = self.data[index][4]
        bbox = np.array(self.data[index][5])
        if self.use_sammask:
            mask = np.array(self.data[index][6])
        else:
            mask = np.zeros_like(data1)
            for ii in range(bbox.shape[0]):
                bbox_one = bbox[ii]
                x_min, x_max = np.min(bbox_one[:, 0]), np.max(bbox_one[:, 0])
                y_min, y_max = np.min(bbox_one[:, 1]), np.max(bbox_one[:, 1])
                if x_min == x_max or y_min == y_max:
                    continue
                else:
                    mask[y_min:y_max + 1, x_min:x_max + 1] = 1.0
        index_ = np.where(mask > 0)
        minx, maxx = np.min(index_[1]), np.max(index_[1])
        miny, maxy = np.min(index_[0]), np.max(index_[0])
        small_img = data1[np.clip(miny - 5, 0, data1.shape[0] - 1):np.clip(maxy + 5 + 1, 0, data1.shape[0]),
                    np.clip(minx - 5, 0, data1.shape[1] - 1):np.clip(minx + 5 + 1, 0, data1.shape[1])]

        data_dict = dict(data=small_img[np.newaxis, np.newaxis, :, :])  # resize
        augmented = self.transforms_v(**data_dict)
        small_img = np.array(augmented.get("data")[0, 0, :, :])
        return data1, label, _, data_num,main_label,mask,small_img



