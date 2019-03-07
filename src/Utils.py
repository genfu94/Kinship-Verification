import re
import random


def atoi(text):
	return int(text) if text.isdigit() else text


def natural_keys(text):
	return [atoi(c) for c in re.split('(\d+)', text)]


'''Helper function, returns two integers m,n s.t a <= m,n <= b and m <> n'''
def getDifferentRandomIntegers(a, b):
	i = random.randint(a, b)
	j = random.randint(a, b)
	while i == j:
		j = random.randint(a, b)

	return i, j


'''Helper function, returns a string representation of num with at least three digits.
   For example: 1 => 001    12 => 012    130 => 130'''
def intToThreeDigitString(num):
	out_string = str(num)
	for i in range(len(str(num)), 3):
		out_string = "0" + out_string
	return out_string


def cropImage(image, rectangle):
	x, y, w, h = rectangle
	return image[y: y + h, x: x + w]
