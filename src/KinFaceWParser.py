import os
from Utils import *
from sklearn.utils import shuffle
from sklearn.model_selection import KFold

'''Load all pairs in the <dataset_dir> (must be a a KFW-X dataset) that belongs to a certain
   subfolder <category> (e.g. category can be either "father-dau", "mother-dau", "father-son", "mother-son").
   You also have to provide the <category_label> used in that specific <category> folder (i.e. for
   "father-dau" <category_label> = "fd", for "mother-son" <category_label> = "ms" and so on...)
   NOTE: <dataset_dir> must be the root of the KFW-X folder'''
def parseKFWSubset(dataset_dir, category, category_label):
	# Load the parent-son pairs
	images = os.listdir(os.path.join(dataset_dir, "images", category))
	images.sort(key=natural_keys)
	pairs = []

	# True pairs
	for i in range(0, len(images) - 1, 2):
		parentImgPath = os.path.join(dataset_dir, "images", category, images[i])
		childImgPath = os.path.join(dataset_dir, "images", category, images[i + 1])
		pairs.append([parentImgPath, childImgPath, 1])

	pairs = shuffle(pairs)

	return pairs


'''Parse an whole KFW-X dataset rooted at <dataset_dir>, providing as output 1) an array of parent-child
   pairs (one for each category, e.g. "father-son", "mother-dau" ecc...) containing both correct and
   incorrect pairings (THIS TIME THE INCORRECT PAIRING IS ALWAYS OF TYPE PARENT-CHILDREN) and 2) a validation
   fold array containing the indices of the training and test set to use for each validation round.'''
def parseKFWDataset(dataset_dir):
	fd_pairs = parseKFWSubset(dataset_dir, "father-dau", "fd")
	fs_pairs = parseKFWSubset(dataset_dir, "father-son", "fs")
	md_pairs = parseKFWSubset(dataset_dir, "mother-dau", "md")
	ms_pairs = parseKFWSubset(dataset_dir, "mother-son", "ms")

	return fd_pairs + fs_pairs + md_pairs + ms_pairs
