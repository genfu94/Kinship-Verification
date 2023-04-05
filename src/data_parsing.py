import os
import cv2
import pickle

# TODO: this should be moved to a utility library
def list_full_paths(directory):
    return [os.path.join(directory, file) for file in os.listdir(directory)]


# TODO: move this somewhere else! Maybe base class?
def _get_feature(parent_img_path, child_img_path):
    return {
        'parent_image_path': parent_img_path,
        'child_image_path': child_img_path,
        'label': 1,
        'child_img': cv2.imread(child_img_path),
        'parent_img': cv2.imread(parent_img_path)
    }


class KinFace_V2_Parser:
    def parse(self, dataset_path: str, output_path: str):
        features = []

        children_images_path = os.path.join(dataset_path, '01')
        young_parents_images_path = os.path.join(dataset_path, '02')
        old_parents_images_path = os.path.join(dataset_path, '03')

        children_images = sorted(list_full_paths(children_images_path))
        young_parents_images = sorted(list_full_paths(young_parents_images_path))
        old_parents_images = sorted(list_full_paths(old_parents_images_path))

        for child, young_parent, old_parent in zip(children_images, young_parents_images, old_parents_images):
            features.append(_get_feature(young_parent, child))
            features.append(_get_feature(old_parent, child))
        
        return pickle.dump(features, open(output_path, 'wb'))


class KinFaceWParser:
    def parse(self, dataset_path: str, output_path: str):
        features = []

        folders_prefixes = {
            'father-dau': 'fd',
            'father-son': 'fs',
            'mother-dau': 'md',
            'mother-son': 'ms'
        }

        for subfolder, prefix in folders_prefixes.items():
            features += self._parse_subfolder(os.path.join(dataset_path, 'images', subfolder), prefix)
        
        return pickle.dump(features, open(output_path, 'wb'))

    def _parse_subfolder(self, subfolder_path, prefix):
        idx = 1
        features = []

        while True:
            parent_img_path = os.path.join(subfolder_path, f'{prefix}_{str(idx).zfill(3)}_1.jpg')
            child_img_path = os.path.join(subfolder_path, f'{prefix}_{str(idx).zfill(3)}_2.jpg')

            if not (os.path.exists(parent_img_path) or os.path.exists(child_img_path)):
                break

            features.append(_get_feature(parent_img_path, child_img_path))
            idx += 1
        
        return features
