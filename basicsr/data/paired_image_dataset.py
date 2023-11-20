from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.matlab_functions import bgr2ycbcr
from basicsr.utils.registry import DATASET_REGISTRY
import cv2

import numpy as np

@DATASET_REGISTRY.register()
class PairedImageDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
                Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.task = opt['task'] if 'task' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.ks = (5, 5)
        self.sigma = 1.5

        self.gt_folder, self.ref_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_ref'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.ref_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'ref', 'gt']
            self.paths = paired_paths_from_lmdb([self.lq_folder, self.ref_folder, self.gt_folder], ['lq', 'ref', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file([self.lq_folder, self.ref_folder, self.gt_folder], ['lq', 'ref', 'gt'],
                                                          self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder([self.lq_folder, self.ref_folder, self.gt_folder], ['lq', 'ref', 'gt'], self.filename_tmpl, self.task)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;

        if self.task == 'CAR':
            # image range: [0, 255], int., H W 1
            gt_path = self.paths[index]['gt_path']
            img_bytes = self.file_client.get(gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, flag='grayscale', float32=False)
            lq_path = self.paths[index]['lq_path']
            img_bytes = self.file_client.get(lq_path, 'lq')
            img_lq = imfrombytes(img_bytes, flag='grayscale', float32=False)
            img_gt = np.expand_dims(img_gt, axis=2).astype(np.float32) / 255.
            img_lq = np.expand_dims(img_lq, axis=2).astype(np.float32) / 255.

        elif self.task == 'Color-DN':
            gt_path = self.paths[index]['gt_path']
            lq_path = gt_path
            img_bytes = self.file_client.get(gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, float32=True)
            if self.opt['phase'] != 'train':
                np.random.seed(seed=0)
            img_lq = img_gt + np.random.normal(0, self.noise/255., img_gt.shape)

        else:
            # image range: [0, 1], float32., H W 3
            gt_path = self.paths[index]['gt_path']
            img_bytes = self.file_client.get(gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, float32=True)

            ref_path = self.paths[index]['ref_path']
            img_bytes = self.file_client.get(ref_path, 'ref')
            img_ref = imfrombytes(img_bytes, float32=True)

            lq_path = self.paths[index]['lq_path']
            img_bytes = self.file_client.get(lq_path, 'lq')
            img_lq = imfrombytes(img_bytes, float32=True)

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            img_gt, img_ref, img_lq = paired_random_crop(img_gt, img_ref, img_lq, gt_size, scale, gt_path)
            # flip, rotation
            img_gt, img_ref, img_lq = augment([img_gt, img_ref, img_lq], self.opt['use_hflip'], self.opt['use_rot'])

        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = bgr2ycbcr(img_gt, y_only=True)[..., None]
            img_ref = bgr2ycbcr(img_ref, y_only=True)[..., None]
            img_lq = bgr2ycbcr(img_lq, y_only=True)[..., None]

        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        # TODO: It is better to update the datasets, rather than force to crop
        if self.opt['phase'] != 'train':
            img_gt = img_gt#[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]
            img_ref = img_ref#[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]

        ### get texture
        # cartoon_lq = cv2.GaussianBlur(img_lq, self.ks, self.sigma)
        # texture_lq = img_lq - cartoon_lq
        #
        # cartoon_gt = cv2.GaussianBlur(img_gt, self.ks, self.sigma)
        # texture_gt = img_gt - cartoon_gt
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_ref, img_lq = img2tensor([img_gt, img_ref, img_lq], bgr2rgb=True, float32=True)
        # cartoon_lq, texture_lq, texture_gt = img2tensor([cartoon_lq, texture_lq, texture_gt], bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_ref, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        # print(img_lq.shape,img_gt.shape,img_lq.min(),img_gt.min(),img_lq.max(),img_gt.max(),lq_path,gt_path)

        return {'lq': img_lq, 'ref': img_ref, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)
