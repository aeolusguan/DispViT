import logging
import os
from pathlib import Path
from glob import glob
import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
from typing import Iterable, Optional, TypeVar, List, Tuple, Union

from . import frame_utils
from .transforms import FlowAugmentor, SparseFlowAugmentor
from .base.easy_dataset import EasyDataset


def iterable_to_str(iterable: Iterable) -> str:
    return "'" + "', '".join([str(item) for item in iterable]) + "'"


T = TypeVar("T", str, bytes)


def verify_str_arg(
    value: T,
    arg: Optional[str] = None,
    valid_values: Optional[Iterable[T]] = None,
    custom_msg: Optional[str] = None,
) -> T:
    if not isinstance(value, str):
        if arg is None:
            msg = "Expected type str, but got type {type}."
        else:
            msg = "Expected type str for argument {arg}, but got type {type}."
        msg = msg.format(type=type(value), arg=arg)
        raise ValueError(msg)
    
    if valid_values is None:
        return value
    
    if value not in valid_values:
        if custom_msg is not None:
            msg = custom_msg
        else:
            msg = "Unknown value '{value}' for argument {arg}. Valid values are {{{valid_values}}}."
            msg = msg.format(value=value, arg=arg, valid_values=iterable_to_str(valid_values))
        raise ValueError(msg)
    
    return value


# read all lines in a file
def read_all_lines(filename):
    with open(filename) as fp:
        lines = [line.rstrip() for line in fp.readlines()]
    return lines


class StereoDataset(EasyDataset):
    def __init__(self, aug_params=None, sparse=False, reader=None):
        self.augmentor = None
        self.sparse = sparse
        self.img_pad = aug_params.pop("img_pad", None) if aug_params is not None else None
        if aug_params is not None and "crop_size" in aug_params:
            if self.sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        if reader is None:
            self.disparity_reader = frame_utils.read_gen
        else:
            self.disparity_reader = reader

        self.is_test = False
        self.disparity_list = []
        self.image_list = []

    @property
    def num_of_resolutions(self):
        return len(self.augmentor.crop_size_arr) if self.augmentor is not None else 1

    def __getitem__(self, index):

        sample = {}
        if self.is_test:
            img1 = np.array(frame_utils.read_gen(self.image_list[index][0])).astype(np.uint8)
            img2 = np.array(frame_utils.read_gen(self.image_list[index][1])).astype(np.uint8)
            # grayscale images
            if len(img1.shape) == 2:
                img1 = np.tile(img1[..., None], (1, 1, 3))
                img2 = np.tile(img2[..., None], (1, 1, 3))
            else:
                img1 = img1[..., :3]
                img2 = img2[..., :3]
            
            sample['img1'] = torch.from_numpy(img1).permute(2, 0, 1).float()
            sample['img2'] = torch.from_numpy(img2).permute(2, 0, 1).float()
            sample['meta'] = self.image_list[index][0]
            return sample
        
        index, res_index = index
        index = index % len(self.image_list)
        disp_path = self.disparity_list[index]
        if isinstance(disp_path, (tuple, list)):
            disp_path = disp_path[0]
        disp = self.disparity_reader(disp_path)
        if isinstance(disp, tuple):
            disp, valid = disp
        else:
            valid = disp < 512

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)
        disp = np.array(disp).astype(np.float32)
        flow = np.stack([disp, np.zeros_like(disp)], axis=-1)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[..., None], (1, 1, 3))
            img2 = np.tile(img2[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid, res_index)
            else:
                img1, img2, flow = self.augmentor(img1, img2, flow, res_index)

        sample['img1'] = torch.from_numpy(img1).permute(2, 0, 1).float()
        sample['img2'] = torch.from_numpy(img2).permute(2, 0, 1).float()
        sample['disp'] = torch.from_numpy(flow).permute(2, 0, 1).float()[0]

        if self.sparse:
            valid = torch.from_numpy(valid)
        else:
            valid = sample['disp'] < 512
        sample['valid'] = valid

        if self.img_pad is not None:
            padH, padW = self.img_pad
            img1 = F.pad(img1, [padW]*2 + [padH]*2)
            img2 = F.pad(img2, [padW]*2 + [padH]*2)
        
        return sample
    
    def __len__(self):
        return len(self.image_list)
    
    def _scan_pairs(
        self,
        paths_left_pattern: str,
        paths_right_pattern: Optional[str] = None,
    ) -> List[Tuple[str, Optional[str]]]:
        
        left_paths = list(sorted(glob(paths_left_pattern)))
        
        right_paths = List[Union[None, str]]
        if paths_right_pattern:
            right_paths = list(sorted(glob(paths_right_pattern)))
        else:
            right_paths = list(None for _ in left_paths)
        
        if not left_paths:
            raise FileNotFoundError(f"Could not find any files matching the patterns: {paths_left_pattern}")
        
        if not right_paths:
            raise FileNotFoundError(f"Could not find any files matching the patterns: {paths_right_pattern}")
        
        if len(left_paths) != len(right_paths):
            raise ValueError(
                f"Found {len(left_paths)} left files but {len(right_paths)} right files using:\n"
                f"left pattern: {paths_left_pattern}\n"
                f"right pattern: {paths_right_pattern}\n"
            )
        
        paths = list((left, right) for left, right in zip(left_paths, right_paths))
        return paths
    

class SceneFlowDatasets(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/SceneFlow', dstype='frames_finalpass', things_test=False):
        super().__init__(aug_params)
        self.root = root
        self.dstype = dstype

        if things_test:
            self._add_things("TEST")
        else:
            self._add_things("TRAIN")
            self._add_monkaa()
            self._add_driving()

    def _add_things(self, split='TRAIN'):
        """ Add Flythings3D data """

        original_length = len(self.disparity_list)
        root = osp.join(self.root, 'FlyingThings3D')
        left_images = sorted( glob(osp.join(root, self.dstype, split, '*/*/left/*.png')) )
        right_images = [ im.replace('left', 'right') for im in left_images ]
        disparity_images = [ im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images ]

        for img1, img2, disp in zip(left_images, right_images, disparity_images):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]
        logging.info(f"Added {len(self.disparity_list) - original_length} from FlyingThings {self.dstype}")

    def _add_monkaa(self):
        original_length = len(self.disparity_list)
        root = osp.join(self.root, 'Monkaa')
        left_images = sorted( glob(osp.join(root, self.dstype, '*/left/*.png')) )
        right_images = [ image_file.replace('left', 'right') for image_file in left_images ]
        disparity_images = [ im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images ]

        for img1, img2, disp in zip(left_images, right_images, disparity_images):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]
        logging.info(f"Added {len(self.disparity_list) - original_length} from Monkaa {self.dstype}")

    def _add_driving(self):
        original_length = len(self.disparity_list)
        root = osp.join(self.root, 'Driving')
        left_images = sorted( glob(osp.join(root, self.dstype, '*/*/*/left/*.png')) )
        right_images = [ image_file.replace('left', 'right') for image_file in left_images ]
        disparity_images = [ im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images ]

        for img1, img2, disp in zip(left_images, right_images, disparity_images):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]
        logging.info(f"Added {len(self.disparity_list) - original_length} from Driving {self.dstype}")


class ETH3D(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/ETH3D', split='training'):
        super(ETH3D, self).__init__(aug_params, sparse=True)

        image1_list = sorted( glob(osp.join(root, f'two_view_{split}/*/im0.png')) )
        image2_list = sorted( glob(osp.join(root, f'two_view_{split}/*/im1.png')) )
        disp_list = sorted( glob(osp.join(root, 'two_view_training_gt/*/disp0GT.pfm')) ) if split == 'training' else [osp.join(root, 'two_view_training_gt/playground_1l/disp0GT.pfm')]*len(image1_list)

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]


class KITTI(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/KITTI', split='training', image_set='kitti_mix'):
        super(KITTI, self).__init__(aug_params, sparse=True, reader=frame_utils.readDispKITTI)
        assert os.path.exists(root)

        if split == 'testing':
            self.is_test = True
            if image_set == 'kitti_2012':
                root = osp.join(root, 'KITTI_2012')
                images1 = sorted(glob(osp.join(root, 'testing', 'colored_0/*_10.png')))
                images2 = sorted(glob(osp.join(root, 'testing', 'colored_1/*_10.png')))
            elif image_set == 'kitti_2015':
                root = osp.join(root, 'KITTI_2015')
                images1 = sorted(glob(osp.join(root, 'testing', 'image_2/*_10.png')))
                images2 = sorted(glob(osp.join(root, 'testing', 'image_3/*_10.png')))
            else:
                raise ValueError("Unknown dataset for test: '{}'".format(image_set))
            for img1, img2 in zip(images1, images2):
                self.image_list += [ [img1, img2] ]

        else:
            this_dir = os.path.dirname(__file__)
            kitti_dict = {
                'kitti_mix_2012_train': f'{this_dir}/filenames/KITTI_mix_2012_train.txt',
                'kitti_mix_2015_train': f'{this_dir}/filenames/KITTI_mix_2015_train.txt',
                'kitti_2012_val': f'{this_dir}/filenames/KITTI_2012_val.txt',
                'kitti_2015_val': f'{this_dir}/filenames/KITTI_2015_val.txt',
                'kitti_mix': f'{this_dir}/filenames/KITTI_mix.txt',
                'kitti_2015_train': f'{this_dir}/filenames/KITTI_2015_train.txt',
                'kitti_2015_trainval': f'{this_dir}/filenames/KITTI_2015_trainval.txt',
                'kitti_2012_train': f'{this_dir}/filenames/KITTI_2012_train.txt',
                'kitti_2012_trainval': f'{this_dir}/filenames/KITTI_2012_trainval.txt',
            }

            assert image_set in kitti_dict.keys()
            data_filename = kitti_dict[image_set]

            self._root_12 = os.path.join(root, 'KITTI_2012')
            self._root_15 = os.path.join(root, 'KITTI_2015')

            self.load_path(data_filename)

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        for line in splits:
            left_name = line[0].split('/')[1]
            if left_name.startswith('image'):
                root = self._root_15
            else:
                root = self._root_12
            img1 = os.path.join(root, line[0])
            img2 = os.path.join(root, line[1])
            self.image_list += [ [img1, img2] ]
            if len(line) > 2:
                disp = os.path.join(root, line[2])
                self.disparity_list += [ disp ]


class Middlebury(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/Middlebury', split='F', image_set='training'):
        super(Middlebury, self).__init__(aug_params, sparse=True, reader=frame_utils.readDispMiddlebury)
        assert os.path.exists(root)
        assert split in ["F", "H", "Q", "2005", "2006", "2014", "2021"]

        if split == "2005":
            scenes = list((Path(root) / "2005").glob("*"))
            for scene in scenes:
                for illum in ["1", "2", "3"]:
                    for exp in ["0", "1", "2"]:       
                        self.image_list += [ [str(scene / f"Illum{illum}/Exp{exp}/view1.png"), str(scene / f"Illum{illum}/Exp{exp}/view5.png")] ]
                        self.disparity_list += [ str(scene / "disp1.png") ]
        elif split == "2006":
            scenes = list((Path(root) / "2006").glob("*"))
            for scene in scenes:
                for illum in ["1", "2", "3"]:
                    for exp in ["0", "1", "2"]:       
                        self.image_list += [ [str(scene / f"Illum{illum}/Exp{exp}/view1.png"), str(scene / f"Illum{illum}/Exp{exp}/view5.png")] ]
                        self.disparity_list += [ str(scene / "disp1.png") ]  
        elif split == "2014": # datasets/Middlebury/2014/Pipes-perfect/im0.png
            scenes = list((Path(root) / "2014").glob("*"))
            for scene in scenes:
                for s in ["E","L",""]:
                    self.image_list += [ [str(scene / "im0.png"), str(scene / f"im1{s}.png")] ]
                    self.disparity_list += [ str(scene / "disp0.pfm") ]
        elif split == "2021":
            scenes = list((Path(root) / "2021/data").glob("*"))
            for scene in scenes:
                self.image_list += [ [str(scene / "im0.png"), str(scene / "im1.png")] ]
                self.disparity_list += [ str(scene / "disp0.pfm") ]
                for s in ["0", "1", "2", "3"]:
                    if os.path.exists(str(scene / f"ambient/L0/im0e{s}.png")):
                        self.image_list += [ [str(scene / f"ambient/L0/im0e{s}.png"), str(scene / f"ambient/L0/im1e{s}.png")] ]
                        self.disparity_list += [ str(scene / "disp0.pfm") ]
        else:
            if image_set == 'training':
                lines = list(map(osp.basename, glob(os.path.join(root, "MiddEval3/trainingF/*"))))
            else:
                lines = list(map(osp.basename, glob(os.path.join(root, "MiddEval3/testF/*"))))

            image1_list = sorted([os.path.join(root, "MiddEval3", f'{image_set}{split}', f'{name}/im0.png') for name in lines])
            image2_list = sorted([os.path.join(root, "MiddEval3", f'{image_set}{split}', f'{name}/im1.png') for name in lines])
            disp_list = sorted([os.path.join(root, "MiddEval3", f'training{split}', f'{name}/disp0GT.pfm') for name in lines]) if image_set == 'training' else [os.path.join(root, "MiddEval3", f'training{split}', 'Adirondack/disp0GT.pfm')]*len(image1_list)
            self.init_disp_list = sorted([os.path.join(root, "MiddEval3_init", f'training{split}', f'{name}/disp0BridgeDepth.pfm') for name in lines]) if image_set == 'training' else sorted([os.path.join(root, "MiddEval3_init", f'test{split}', f'{name}/disp0BridgeDepth.pfm') for name in lines])
            assert len(image1_list) == len(image2_list) == len(disp_list) > 0, [image1_list, split]
            for img1, img2, disp in zip(image1_list, image2_list, disp_list):
                self.image_list += [ [img1, img2] ]
                self.disparity_list += [ disp ]


class SintelStereo(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/SintelStereo'):
        super().__init__(aug_params, sparse=True, reader=frame_utils.readDispSintelStereo)
        
        image1_list = sorted( glob(osp.join(root, 'training/*_left/*/frame_*.png')) )
        image2_list = sorted( glob(osp.join(root, 'training/*_right/*/frame_*.png')) )
        disp_list = sorted( glob(osp.join(root, 'training/disparities/*/frame_*.png')) ) * 2
        
        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            assert img1.split('/')[-2:] == disp.split('/')[-2:]
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]
    
    
class FallingThings(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/FallingThings', variant: str = "both"):
        super().__init__(aug_params, reader=frame_utils.readDispFallingThings)
        assert os.path.exists(root)
        root = Path(root)
        
        verify_str_arg(variant, "variant", valid_values=["single", "mixed", "both"])
        
        variants = {
            "single": ["single"],
            "mixed": ["mixed"],
            "both": ["single", "mixed"],
        }[variant]
        
        split_prefix = {
            "single": Path("*") / "*",
            "mixed": Path("*"),
        }
        
        for s in variants:
            left_img_pattern = str(root / s / split_prefix[s] / "*.left.jpg")
            right_img_pattern = str(root / s / split_prefix[s] / "*.right.jpg")
            self.image_list += self._scan_pairs(left_img_pattern, right_img_pattern)
            
            left_disparity_pattern = str(root / s / split_prefix[s] / "*.left.depth.png")
            self.disparity_list += self._scan_pairs(left_disparity_pattern, None)


class TartanAir(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/TartanAir'):
        super().__init__(aug_params, reader=frame_utils.readDispTartanAir)
        assert os.path.exists(root)
        root = Path(root)
        
        left_img_pattern = str(root / "*/*/*/image_left/*_left.png")
        right_img_pattern = str(root / "*/*/*/image_right/*_right.png")
        self.image_list = self._scan_pairs(left_img_pattern, right_img_pattern)
        
        left_disparity_pattern = str(root / "*/*/*/depth_left/*_left_depth.npy")
        self.disparity_list = self._scan_pairs(left_disparity_pattern, None)


class CREStereo(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/CREStereo'):
        super().__init__(aug_params, reader=frame_utils.readDispCREStereo)
        assert os.path.exists(root)
        root = Path(root)
        
        dirs = ["shapenet", "reflective", "tree", "hole"]
        
        for s in dirs:
            left_img_pattern = str(root / s / "*_left.jpg")
            right_img_pattern = str(root / s / "*_right.jpg")
            self.image_list += self._scan_pairs(left_img_pattern, right_img_pattern)
            
            left_disparity_pattern = str(root / s / "*_left.disp.png")
            self.disparity_list += self._scan_pairs(left_disparity_pattern, None)


class VirtualKitti2(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/VKITTI2'):
        super().__init__(aug_params, reader=frame_utils.readDispVKITTI)
        assert os.path.exists(root)
        root = Path(root)
        
        dirs = ["Scene01", "Scene02", "Scene06", "Scene18", "Scene20"]
        for s in dirs:
            left_img_pattern = str(root / s / "*" / "frames" / "rgb" / "Camera_0" / "rgb_*.jpg")
            right_img_pattern = str(root / s / "*" / "frames" / "rgb" / "Camera_1" / "rgb_*.jpg")
            self.image_list += self._scan_pairs(left_img_pattern, right_img_pattern)
            
            left_disparity_pattern = str(root / s / "*" / "frames" / "depth" / "Camera_0" / "depth_*.png")
            self.disparity_list += self._scan_pairs(left_disparity_pattern, None)


class CarlaHighres(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/carla-highres'):
        super().__init__(aug_params=aug_params)
        assert os.path.exists(root)

        image1_list = sorted(glob(osp.join(root, 'trainingF/*/im0.png')))
        image2_list = [im.replace('im0', 'im1') for im in image1_list]
        disp_list = [im.replace('im0.png', 'disp0GT.pfm') for im in image1_list]

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]


class InStereo2K(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/InStereo2K', variant: str = "both"):
        super().__init__(aug_params, sparse=True, reader=frame_utils.readDispInStereo2K)
        assert os.path.exists(root)
        root = Path(root)
        
        verify_str_arg(variant, "variant", valid_values=["single", "mixed", "both"])
        
        variants = {
            "train": ["train"],
            "test": ["test"],
            "both": ["train", "test"],
        }[variant]
        
        for s in variants:
            left_img_pattern = str(root / s / "*" / "left.png")
            right_img_pattern = str(root / s / "*" / "right.png")
            self.image_list += self._scan_pairs(left_img_pattern, right_img_pattern)
            
            left_disparity_pattern = str(root / s / "*" / "left_disp.png")
            self.disparity_list += self._scan_pairs(left_disparity_pattern, None)


class IRS(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/IRS'):
        super().__init__(aug_params, reader=frame_utils.load_exr)
        image1_list = sorted(glob(osp.join(root, '*/*/l_*.png')))
        image2_list = sorted(glob(osp.join(root, '*/*/r_*.png')))
        disp_list = sorted(glob(osp.join(root, '*/*/d_*.exr')))
        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            assert img1.split('/')[-2] == disp.split('/')[-2]
            assert img1.split('.')[0].split('_')[-1] == disp.split('.')[0].split('_')[-1]
            assert img1.split('.')[0].split('_')[-1] == img2.split('.')[0].split('_')[-1]
            if 'QAOfficeAndSecurityRoom2_Night' in img1: # bad scenes
                continue
            self.image_list += [[img1, img2]]
            self.disparity_list += [[disp]]


class Booster(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/booster', split='train'):
        super().__init__(aug_params, sparse=True, reader=frame_utils.readDispBooster)
        assert os.path.exists(root)

        folder_list = sorted(glob(osp.join(root, split + '/balanced/*')))        
        for folder in folder_list:
            image1_list = sorted(glob(osp.join(folder, 'camera_00/im*.png')))
            image2_list = sorted(glob(osp.join(folder, 'camera_02/im*.png')))
            if split == "train":
                for img1 in image1_list:
                    for img2 in image2_list:
                        self.image_list += [ [img1, img2] ]
                        self.disparity_list += [ osp.join(folder, 'disp_00.npy') ]
            else:
                for img1, img2 in zip(image1_list, image2_list):
                    self.image_list += [ [img1, img2] ]
                    self.disparity_list += [ osp.join(folder, 'disp_00.npy') ]

class FSD(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/FSD'):
        super().__init__(aug_params, reader=frame_utils.readDispFSD)
        assert os.path.exists(root)
        root = Path(root)

        left_img_pattern = str(root / "*/dataset/data/left/rgb/*.jpg")
        right_img_pattern = str(root / "*/dataset/data/right/rgb/*.jpg")
        self.image_list = self._scan_pairs(left_img_pattern, right_img_pattern)

        left_disparity_pattern = str(root / "*/dataset/data/left/disparity/*.png")
        self.disparity_list = self._scan_pairs(left_disparity_pattern, None)


def build_train_dataset(crop_size, spatial_scale, yjitter, datasets, folds, saturation_range=None, img_gamma=None, do_flip=None):
    """ Create the training sets """
    aug_params = {'crop_size': list(crop_size), 'min_scale': spatial_scale[0], 'max_scale': spatial_scale[1], 'do_flip': False, 'yjitter': yjitter}
    if saturation_range is not None:
        aug_params["saturation_range"] = tuple(saturation_range)
    if img_gamma is not None:
        aug_params["gamma"] = img_gamma
    if do_flip is not None:
        aug_params["do_flip"] = do_flip

    train_dataset = None
    logger = logging.getLogger(__name__)
    assert len(datasets) == len(folds)
    for dataset_name, fold in zip(datasets, folds):
        if dataset_name.startswith('middlebury_'):
            new_dataset = Middlebury(aug_params, split=dataset_name.replace('middlebury_', ''))
            logger.info(f"{len(new_dataset)} samples from {dataset_name}")
        elif dataset_name == 'sceneflow':
            new_dataset = SceneFlowDatasets(aug_params, dstype='frames_finalpass')
            logger.info(f"{len(new_dataset)} samples from SceneFlow")
        elif dataset_name == 'eth3d':
            new_dataset = ETH3D(aug_params, split='training')
            logger.info(f"{len(new_dataset)} samples from ETH3D")
        elif 'kitti_' in dataset_name:
            new_dataset = KITTI(aug_params, image_set=dataset_name)
            logger.info(f"{len(new_dataset)} samples from KITTI")
        elif dataset_name == 'sintel_stereo':
            new_dataset = SintelStereo(aug_params)
            logger.info(f"{len(new_dataset)} samples from SintelStereo")
        elif dataset_name == 'falling_things':
            new_dataset = FallingThings(aug_params, variant='both')
            logger.info(f"{len(new_dataset)} samples from FallingThings")
        elif dataset_name == 'tartan_air':
            new_dataset = TartanAir(aug_params)
            logger.info(f"{len(new_dataset)} samples from TartanAir")
        elif dataset_name == 'carla_highres':
            new_dataset = CarlaHighres(aug_params)
            logger.info(f"{len(new_dataset)} samples from Carla Highres")
        elif dataset_name == 'crestereo':
            new_dataset = CREStereo(aug_params)
            logger.info(f"{len(new_dataset)} samples from CREStereo")
        elif dataset_name == 'vkitti2':
            new_dataset = VirtualKitti2(aug_params)
            logger.info(f"{len(new_dataset)} samples from VirtualKitti2")
        elif dataset_name.startswith('booster'):
            new_dataset = Booster(aug_params)
            logger.info(f"{len(new_dataset)} samples from Booster")
        elif dataset_name == 'in_stereo2k':
            new_dataset = InStereo2K(aug_params, variant='both')
            logger.info(f"{len(new_dataset)} samples from InStereo2K")
        elif dataset_name == 'fsd':
            new_dataset = FSD(aug_params)
            logger.info(f"{len(new_dataset)} samples from FSD")
        elif dataset_name == 'irs':
            new_dataset = IRS(aug_params)
            logger.info(f"{len(new_dataset)} samples from IRS")
        else:
            raise ValueError(f"Unrecognized dataset {dataset_name}")
        if fold > 0:
            new_dataset = fold * new_dataset
        train_dataset = new_dataset if train_dataset is None else train_dataset + new_dataset

    return train_dataset


def build_val_dataset(dataset_name):
    logger = logging.getLogger(__name__)
    if dataset_name == 'things':
        val_dataset = SceneFlowDatasets(dstype='frames_finalpass', things_test=True)
        logger.info('Number of validation image pairs: %d' % len(val_dataset))
    elif 'kitti' in dataset_name:
        # perform validation using the KITTI (train) split
        val_dataset = KITTI(image_set=dataset_name)
        logger.info('Number of validation image pairs: %d' % len(val_dataset))
    elif dataset_name == 'eth3d':
        val_dataset = ETH3D(split='training')
        logger.info('Number of validation image pairs: %d' % len(val_dataset))
    elif dataset_name.startswith("middlebury_"):
        val_dataset = Middlebury(split=dataset_name.replace('middlebury_', ''))
        logger.info('Number of validation image pairs: %d' % len(val_dataset))

    return val_dataset