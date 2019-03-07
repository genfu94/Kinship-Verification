import cv2
import numpy as np
from KinFaceWParser import *
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from Utils import *
import keras
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout
import imgaug as ia
from imgaug import augmenters as iaa

'''
iaa.Crop(percent=(0, 0.1)), # random crops
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(0.5,
        iaa.GaussianBlur(sigma=(0, 0.5))
    ),
    # Strengthen or weaken the contrast in each image.
    iaa.ContrastNormalization((0.75, 1.5)),
    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    iaa.Multiply((0.8, 1.2), per_channel=0.2),
    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-25, 25),
        shear=(-8, 8)
    )
'''

seq = iaa.Sequential([
    iaa.Fliplr(0.2), # horizontal flips
], random_order=True) # apply augmenters in random order


def iterate_mini_batches(dataset, batch_size, augment=False):
	#while True:
	x1_b = np.zeros((batch_size, 64, 64, 3), dtype=np.uint8)
	x2_b = np.zeros((batch_size, 64, 64, 3), dtype=np.uint8)
	y_b = np.zeros((batch_size, 2), dtype='int32')
	for i in range(0, int(len(dataset)/batch_size)):
		i_b = 0
		for j in range(i, i+batch_size):
			child, parent, currTarget = dataset[j]
			x1_b[i_b] = child
			x2_b[i_b] = parent
			y_b[i_b] = [1, 0] if currTarget == 0 else [0, 1]
			i_b = i_b + 1
		if augment:
			yield [seq.augment_images(x1_b), seq.augment_images(x2_b)], y_b
		else:
			yield [x1_b, x2_b], y_b


kfw1 = parseKFWDataset('../Datasets/KinFaceW-I')
kfw2 = parseKFWDataset('../Datasets/KinFaceW-II')
kfw_datasets = kfw1 + kfw2
kfw_datasets = shuffle(kfw_datasets)
kfw_true_pair_split1 = kfw_datasets[:int(len(kfw_datasets)*80/100)]
kfw_true_pair_split2 = kfw_datasets[int(len(kfw_datasets)*80/100):]

trainingSet = []
for i, instance in enumerate(kfw_true_pair_split1):
	parentImage = cv2.imread(instance[0])
	childImage = cv2.imread(instance[1])
	if parentImage is None or childImage is None:
		continue
	trainingSet.append([parentImage, childImage, 1])


trainSize = len(trainingSet)
for i in range(trainSize):
	j, k = getDifferentRandomIntegers(0, trainSize - 1)
	trainingSet.append([trainingSet[j][0], trainingSet[k][1], 0])

developmentSet = []
for i, instance in enumerate(kfw_true_pair_split2):
	parentImage = cv2.imread(instance[0])
	childImage = cv2.imread(instance[1])
	developmentSet.append([parentImage, childImage, 1])

devSize = len(developmentSet)
for i in range(devSize):
	j, k = getDifferentRandomIntegers(0, devSize-1)
	developmentSet.append([developmentSet[j][0], developmentSet[k][1], 0])

trainingSet = shuffle(trainingSet)
developmentSet = shuffle(developmentSet)

input_1 = keras.layers.Input(shape=(64, 64, 3))
input_2 = keras.layers.Input(shape=(64, 64, 3))
input_concat = keras.layers.concatenate([input_1, input_2], axis=-1)
conv1 = Conv2D(32, kernel_size=5, activation='relu', padding="same")(input_concat)
mp1 = MaxPool2D(pool_size=[2, 2], strides=2)(conv1)
conv2 = Conv2D(64, kernel_size=5, activation='relu', padding="same")(mp1)
mp2 = MaxPool2D(pool_size=[2, 2], strides=2)(conv2)
conv3 = Conv2D(128, kernel_size=5, activation='relu', padding="same")(mp2)
mp3 = MaxPool2D(pool_size=[2, 2], strides=2)(conv3)
conv4 = Conv2D(256, kernel_size=5, activation='relu', padding="same")(mp3)
mp4 = MaxPool2D(pool_size=[2, 2], strides=2)(conv4)
conv5 = Conv2D(512, kernel_size=5, activation='relu', padding="same")(mp4)
mp5 = MaxPool2D(pool_size=[2, 2], strides=2)(conv5)
conv6 = Conv2D(1024, kernel_size=5, activation='relu', padding="same")(mp5)
mp6 = MaxPool2D(pool_size=[2, 2], strides=2)(conv6)
out = Flatten()(mp6)
out = Dense(units=1024, activation="relu")(out)
out = Dropout(0.2)(out)
out = Dense(2, activation='softmax')(out)
model = keras.models.Model(inputs=[input_1, input_2], outputs=out)

opt = Adam(lr=0.000001)
model.compile(optimizer=opt, loss='categorical_crossentropy',  metrics=['accuracy'])

model.summary()
BATCH_SIZE = 256
TRAINING_STEPS = len(trainingSet)/BATCH_SIZE
VALIDATION_STEPS = len(developmentSet) / BATCH_SIZE
'''model.fit_generator(generator=iterate_mini_batches(trainingSet, BATCH_SIZE),
					validation_data=iterate_mini_batches(developmentSet, BATCH_SIZE),
					validation_steps = VALIDATION_STEPS,
					steps_per_epoch=TRAINING_STEPS,
					epochs=20)'''
for epoch in range(10):
	for X_b, y_b in iterate_mini_batches(trainingSet, BATCH_SIZE, augment = True):
		x1_b = X_b[0]
		x2_b = X_b[1]
		if x1_b.size == 0 or x2_b.size == 0:
			continue
		model.fit([x1_b, x2_b], y_b, verbose=0)

	y_true = []
	y_pred = []
	for X_b, y_b in iterate_mini_batches(developmentSet, 16):
		x1_b = X_b[0]
		x2_b = X_b[1]
		if x1_b.size == 0 or x2_b.size == 0:
			continue
		preds = model.predict([x1_b, x2_b])
		y_pred += (np.argmax(preds, axis=1)).tolist()
		y_true += np.argmax(y_b, axis=1).tolist()

	print("Finished epoch", epoch, "with an accuracy of", accuracy_score(y_true, y_pred))
'''BATCH_SIZE = 128
TRAINING_STEPS = len(trainingSet)/BATCH_SIZE
VALIDATION_STEPS = len(developmentSet) / BATCH_SIZE
model.fit_generator(generator=iterate_mini_batches(trainingSet, BATCH_SIZE),
					validation_data=iterate_mini_batches(developmentSet, BATCH_SIZE),
					validation_steps = VALIDATION_STEPS,
					steps_per_epoch=TRAINING_STEPS,
					epochs=10)'''
