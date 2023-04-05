import os
import cv2
from typing import List
from abc import ABC, abstractmethod
from collections import defaultdict

# TODO: this should be moved to a utility library
def list_full_paths(directory):
    return [os.path.join(directory, file) for file in os.listdir(directory)]


class KinshipDatasetParser(ABC):
    @abstractmethod
    def parse(self, dataset_path: str):
        pass

    
    def _get_feature(self, parent_img_path, child_img_path):
        return {
            'parent_image_path': parent_img_path,
            'child_image_path': child_img_path,
            'label': 1,
            'child_img': cv2.imread(child_img_path),
            'parent_img': cv2.imread(parent_img_path)
        }


class KinFace_V2_Parser(KinshipDatasetParser):
    def parse(self, dataset_path: str):
        features = []

        children_images_path = os.path.join(dataset_path, '01')
        young_parents_images_path = os.path.join(dataset_path, '02')
        old_parents_images_path = os.path.join(dataset_path, '03')

        children_images = sorted(list_full_paths(children_images_path))
        young_parents_images = sorted(list_full_paths(young_parents_images_path))
        old_parents_images = sorted(list_full_paths(old_parents_images_path))

        for child, young_parent, old_parent in zip(children_images, young_parents_images, old_parents_images):
            features.append(self._get_feature(young_parent, child))
            features.append(self._get_feature(old_parent, child))
        
        return features


class KinFaceWParser(KinshipDatasetParser):
    def parse(self, dataset_path: str):
        features = []

        folders_prefixes = {
            'father-dau': 'fd',
            'father-son': 'fs',
            'mother-dau': 'md',
            'mother-son': 'ms'
        }

        for subfolder, prefix in folders_prefixes.items():
            if os.path.exists(os.path.join(dataset_path, 'images', subfolder, 'Thumbs.db')):
                os.remove(os.path.join(dataset_path, 'images', subfolder, 'Thumbs.db'))

            features += self._parse_subfolder(os.path.join(dataset_path, 'images', subfolder), prefix)
        
        return features

    def _parse_subfolder(self, subfolder_path, prefix):
        idx = 1
        features = []

        while True:
            parent_img_path = os.path.join(subfolder_path, f'{prefix}_{str(idx).zfill(3)}_1.jpg')
            child_img_path = os.path.join(subfolder_path, f'{prefix}_{str(idx).zfill(3)}_2.jpg')

            if not (os.path.exists(parent_img_path) or os.path.exists(child_img_path)):
                break

            features.append(self._get_feature(parent_img_path, child_img_path))
            idx += 1
        
        assert(len(features) * 2 == len(os.listdir(subfolder_path)))
        return features


class TSKinFaceParser(KinshipDatasetParser):
    def parse(self, dataset_path: str):
        subfolders = ["FMD", "FMS", "FMSD"]
        features = []

        for folder in subfolders:
            features += self.parse_subfolder(os.path.join(dataset_path, folder))
        
        return features

    
    def _make_parent_child_pairs(self, parents_img_paths: List[str], children_img_paths: List[str]):
        features = []
        for parent_path in parents_img_paths:
            for child_path in children_img_paths:
                features.append(self._get_feature(parent_path, child_path))

        return features

    def _get_parents_and_children_images_in_subfolder_by_idx(self, subfolder_path: str, idx: int):
        subfolder_name = os.path.basename(subfolder_path)
        char_to_type = {
            'D': 'child',
            'F': 'parent',
            'M': 'parent',
            'S': 'child'
        }

        img_paths = defaultdict(list)

        for char in subfolder_name:
            bucket = char_to_type[char]
            img_paths[bucket].append(os.path.join(subfolder_path, f'{subfolder_name}-{idx}-{char}.jpg'))
        
        return img_paths['parent'], img_paths['child']

    def parse_subfolder(self, subfolder_path: str):
        idx = 1
        features = []
        
        while True:
            parents, children = self._get_parents_and_children_images_in_subfolder_by_idx(subfolder_path, idx)
            if not (os.path.exists(parents[0])):
                break

            features += self._make_parent_child_pairs(parents, children)
            idx += 1


        return features