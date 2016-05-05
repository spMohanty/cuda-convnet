#!/usr/bin/env python
import sys
import os
import cPickle as pickle

from PIL import Image
from PIL import ImageOps
from random import shuffle

import numpy as np
from scipy import misc

import traceback

image_size = 256
SIZE = (image_size, image_size)
IMAGES_PER_BATCH = 500
channels = 3

OUTPUT_PATH = "/root/cuda_runtime/food_101/data/batches"
INPUT_DIR ="/root/cuda_runtime/food_101/data/images"


found_ids = {}
count = 0
filename_id_map = {}
for root, dirs, files in os.walk(INPUT_DIR, topdown=False):
    for name in files:
	if count % 1000 == 0:
		sys.stderr.write('Found %d files\n' % (count + 1))
	count += 1

        filename = os.path.join(root, name)
	label = filename.split("/")[-2]

	if not label in found_ids:
		found_ids[label] = []

	found_ids[label].append(filename)
	filename_id_map[filename] = label

all_ids = found_ids.keys()
sys.stderr.write('Found %d total labels\n' % (len(all_ids)))

shuffle(all_ids)

wanted_ids = all_ids
sys.stderr.write('Looking for %d labels\n' % (len(wanted_ids)))

wanted_files = []
label_indexes = {}

for index, id in enumerate(wanted_ids):
	wanted_files += found_ids[id]
	label_indexes[id] = index

shuffle(wanted_files)

sys.stderr.write('Starting to process %d files\n' % (len(wanted_files)))

total_image = np.zeros((SIZE[0] * SIZE[1] * 3), dtype=np.float64)

images_processed = 0

for i in xrange(0, len(wanted_files), IMAGES_PER_BATCH):
	current_images = wanted_files[i:(i + IMAGES_PER_BATCH)]
	labels = []
	images = []
	for image_path in current_images:
		print "Processing......"+image_path
		id = filename_id_map[image_path]
		try:
      			image = misc.imread(image_path)
    		except IOError, e:
			sys.stderr.write("IOError for '%s' - %s" % (image_path, e))

		shape = image.shape
		if len(shape) < 3:
			sys.stderr.write("Missing channels for '%s', skipping" % image_path)
			continue

		width = shape[1]
		height = shape[0]
		channels = shape[2]

		if channels < 3:
			sys.stderr.write("Too few channels for '%s', skipping" % image_path)
			continue

		if width == SIZE[0] and height == SIZE[1]:
			resized = image
		else:
			if width > height:
				margin = ((width - height) / 2)
				image = image[:, margin: -margin]
			if height > width:
				margin = ((height - width) / 2)
				image = image[margin : -margin, :]

			try:
				resized = misc.imresize(image, SIZE)
			except ValueError, e:
				sys.stderr.write("ValueError when resizing '%s' - %s" % (image_path, e))
				continue


		red_channel = resized[:, :, 0]
		red_channel.shape = (image_size * image_size)
		green_channel = resized[:, :, 1]
		green_channel.shape = (image_size * image_size)
		blue_channel = resized[:, :, 2]
		blue_channel.shape = (image_size * image_size)

		all_channels = np.append(np.append(red_channel, green_channel), blue_channel)
		total_image += all_channels
		images.append(all_channels)
		label_index = label_indexes[id]
		labels.append(label_index)
		images_processed += 1

	output_index = (i / IMAGES_PER_BATCH)
	output_path= '%s/data_batch_%d' % (OUTPUT_PATH, output_index)

	output_file = open(output_path, 'wb')

	# CIFAR 
	#images_data = np.vstack(images).transpose()
	#output_dict = { 'data': images_data, 'labels': labels}
	#pickle.dump(output_dict, output_file)
	
	# RAW
	images_data = np.vstack(images).transpose()
	labels_data = np.vstack(labels).astype(np.float32)
	output_file.write(labels_data.tostring())
	output_file.write(images_data.tostring())			

	output_file.close()
	sys.stderr.write('Wrote %s\n' % (output_path))



mean_image = total_image / images_processed
meta = {'data_mean': mean_image, 'label_names': wanted_ids}
meta_output_path= '%s/batches.meta' % (OUTPUT_PATH)
meta_output = open(meta_output_path, 'w')
pickle.dump(meta, meta_output)
meta_output.close()




