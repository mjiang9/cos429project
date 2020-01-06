import os
import cv2
from glob import glob
import numpy as np
import re
import random
from matplotlib import pyplot as plt

testing_set = []
NUM_OF_TESTING = 300
# Equal number of testing images per race, per age category, per gender
NUM_OF_TESTING_PER_CATEGORY = int(NUM_OF_TESTING / 2 / 5 / 2)

def get_training_sets():
	training_faces_dir = 'UTKFace'
	face_filenames = glob(os.path.join(training_faces_dir, '*.jpg'))

	# Age Categories:
	# 0: 0-14 years
	# 1: 15-24 years
	# 2: 25-54 years
	# 3: 55-64 years
	# 4: 65 years and over

	dark_skinned_faces = []
	dark_skinned_faces_male_0 = []
	dark_skinned_faces_female_0 = []
	dark_skinned_faces_male_1 = []
	dark_skinned_faces_female_1 = []
	dark_skinned_faces_male_2 = []
	dark_skinned_faces_female_2 = []
	dark_skinned_faces_male_3 = []
	dark_skinned_faces_female_3 = []
	dark_skinned_faces_male_4 = []
	dark_skinned_faces_female_4 = []

	light_skinned_faces = []
	light_skinned_faces_male_0 = []
	light_skinned_faces_female_0 = []
	light_skinned_faces_male_1 = []
	light_skinned_faces_female_1 = []
	light_skinned_faces_male_2 = []
	light_skinned_faces_female_2 = []
	light_skinned_faces_male_3 = []
	light_skinned_faces_female_3 = []
	light_skinned_faces_male_4 = []
	light_skinned_faces_female_4 = []

	for filename in face_filenames:
		if (re.match(r'UTKFace/([0-9]|[0-9][0-9]|[0-9][0-9][0-9])_[0-1]_1_(.*)', filename)):
			# All dark skinned faces, for testing purposes
			dark_skinned_faces.append(filename)

		# Dark-skinned males 0-14 yrs old
		if (re.match(r'UTKFace/([0-9]|1[0-4])_0_1_(.*)', filename)):
			dark_skinned_faces_male_0.append(filename)

		# Dark-skinned females 0-14 yrs old
		if (re.match(r'UTKFace/([0-9]|1[0-4])_1_1_(.*)', filename)):
			dark_skinned_faces_female_0.append(filename)

		# Dark-skinned males 15-24 yrs old
		if (re.match(r'UTKFace/(1[5-9]|2[0-4])_0_1_(.*)', filename)):
			dark_skinned_faces_male_1.append(filename)

		# Dark-skinned females 15-24 yrs old
		if (re.match(r'UTKFace/(1[5-9]|2[0-4])_1_1_(.*)', filename)):
			dark_skinned_faces_female_1.append(filename)

		# Dark-skinned males 25-54 yrs old
		if (re.match(r'UTKFace/(2[5-9]|3[0-9]|4[0-9]|5[0-4])_0_1_(.*)', filename)):
			dark_skinned_faces_male_2.append(filename)

		# Dark-skinned females 25-54 yrs old
		if (re.match(r'UTKFace/(2[5-9]|3[0-9]|4[0-9]|5[0-4])_1_1_(.*)', filename)):
			dark_skinned_faces_female_2.append(filename)

		# Dark-skinned males 55-64 yrs old
		if (re.match(r'UTKFace/(5[5-9]|6[0-4])_0_1_(.*)', filename)):
			dark_skinned_faces_male_3.append(filename)

		# Dark-skinned females 55-64 yrs old
		if (re.match(r'UTKFace/(5[5-9]|6[0-4])_1_1_(.*)', filename)):
			dark_skinned_faces_female_3.append(filename)

		# Dark-skinned males 65+ yrs old
		if (re.match(r'UTKFace/(6[5-9]|7[0-9]|8[0-9]|9[0-9]|1[0-9][0-9])_0_1_(.*)', filename)):
			dark_skinned_faces_male_4.append(filename)

		# Dark-skinned females 65+ yrs old
		if (re.match(r'UTKFace/(6[5-9]|7[0-9]|8[0-9]|9[0-9]|1[0-9][0-9])_1_1_(.*)', filename)):
			dark_skinned_faces_female_4.append(filename)

	for filename in face_filenames:
		if (re.match(r'UTKFace/([0-9]|[0-9][0-9]|[0-9][0-9][0-9])_[0-1]_0_(.*)', filename)):
			# All light skinned faces, for testing purposes
			light_skinned_faces.append(filename)

		# light-skinned males 0-14 yrs old
		if (re.match(r'UTKFace/([0-9]|1[0-4])_0_0_(.*)', filename)):
			light_skinned_faces_male_0.append(filename)

		# light-skinned females 0-14 yrs old
		if (re.match(r'UTKFace/([0-9]|1[0-4])_1_0_(.*)', filename)):
			light_skinned_faces_female_0.append(filename)

		# light-skinned males 15-24 yrs old
		if (re.match(r'UTKFace/(1[5-9]|2[0-4])_0_0_(.*)', filename)):
			light_skinned_faces_male_1.append(filename)

		# light-skinned females 15-24 yrs old
		if (re.match(r'UTKFace/(1[5-9]|2[0-4])_1_0_(.*)', filename)):
			light_skinned_faces_female_1.append(filename)

		# light-skinned males 25-54 yrs old
		if (re.match(r'UTKFace/(2[5-9]|3[0-9]|4[0-9]|5[0-4])_0_0_(.*)', filename)):
			light_skinned_faces_male_2.append(filename)

		# light-skinned females 25-54 yrs old
		if (re.match(r'UTKFace/(2[5-9]|3[0-9]|4[0-9]|5[0-4])_1_0_(.*)', filename)):
			light_skinned_faces_female_2.append(filename)

		# light-skinned males 55-64 yrs old
		if (re.match(r'UTKFace/(5[5-9]|6[0-4])_0_0_(.*)', filename)):
			light_skinned_faces_male_3.append(filename)

		# light-skinned females 55-64 yrs old
		if (re.match(r'UTKFace/(5[5-9]|6[0-4])_1_0_(.*)', filename)):
			light_skinned_faces_female_3.append(filename)

		# light-skinned males 65+ yrs old
		if (re.match(r'UTKFace/(6[5-9]|7[0-9]|8[0-9]|9[0-9]|1[0-9][0-9])_0_0_(.*)', filename)):
			light_skinned_faces_male_4.append(filename)

		# light-skinned females 65+ yrs old
		if (re.match(r'UTKFace/(6[5-9]|7[0-9]|8[0-9]|9[0-9]|1[0-9][0-9])_1_0_(.*)', filename)):
			light_skinned_faces_female_4.append(filename)

	# Hold back for testing data
	hold_back_testing_set(dark_skinned_faces_male_0)
	hold_back_testing_set(dark_skinned_faces_female_0)
	hold_back_testing_set(dark_skinned_faces_male_1)
	hold_back_testing_set(dark_skinned_faces_female_1)
	hold_back_testing_set(dark_skinned_faces_male_2)
	hold_back_testing_set(dark_skinned_faces_female_2)
	hold_back_testing_set(dark_skinned_faces_male_3)
	hold_back_testing_set(dark_skinned_faces_female_3)
	hold_back_testing_set(dark_skinned_faces_male_4)
	hold_back_testing_set(dark_skinned_faces_female_4)

	hold_back_testing_set(light_skinned_faces_male_0)
	hold_back_testing_set(light_skinned_faces_female_0)
	hold_back_testing_set(light_skinned_faces_male_1)
	hold_back_testing_set(light_skinned_faces_female_1)
	hold_back_testing_set(light_skinned_faces_male_2)
	hold_back_testing_set(light_skinned_faces_female_2)
	hold_back_testing_set(light_skinned_faces_male_3)
	hold_back_testing_set(light_skinned_faces_female_3)
	hold_back_testing_set(light_skinned_faces_male_4)
	hold_back_testing_set(light_skinned_faces_female_4)

	dm0_100 = dark_skinned_faces_male_0.copy()
	df0_100 = dark_skinned_faces_female_0.copy()
	dm1_100 = dark_skinned_faces_male_1.copy()
	df1_100 = dark_skinned_faces_female_1.copy()
	dm2_100 = dark_skinned_faces_male_2.copy()
	df2_100 = dark_skinned_faces_female_2.copy()
	dm3_100 = dark_skinned_faces_male_3.copy()
	df3_100 = dark_skinned_faces_female_3.copy()
	dm4_100 = dark_skinned_faces_male_4.copy()
	df4_100 = dark_skinned_faces_female_4.copy()

	# Balance gender distribution
	counter = 0
	for i in range(len(df0_100) - len(dm0_100)):
		random_index = random.randint(0, len(df0_100) - 1)
		del df0_100[random_index]
		counter += 1

	for i in range(len(df1_100) - len(dm1_100)):
		random_index = random.randint(0, len(df1_100) - 1)
		del df1_100[random_index]
		counter += 1

	for i in range(len(dm2_100) - len(df2_100)):
		random_index = random.randint(0, len(dm2_100) - 1)
		del dm2_100[random_index]
		counter += 1

	for i in range(len(dm3_100) - len(df3_100)):
		random_index = random.randint(0, len(dm3_100) - 1)
		del dm3_100[random_index]
		counter += 1

	for i in range(len(dm4_100) - len(df4_100)):
		random_index = random.randint(0, len(dm4_100) - 1)
		del dm4_100[random_index]
		counter += 1

	# Create 100% dark skinned training set
	train_100 = []
	train_100.extend(dm0_100)
	train_100.extend(df0_100)
	train_100.extend(dm1_100)
	train_100.extend(df1_100)
	train_100.extend(dm2_100)
	train_100.extend(df2_100)
	train_100.extend(dm3_100)
	train_100.extend(df3_100)
	train_100.extend(dm4_100)
	train_100.extend(df4_100)

	# Find number of images to include in other training sets based on
	# age distribution of 100% dark skinned training set
	total_training_dark_skinned_faces = len(dm0_100) + len(df0_100)+ len(dm1_100) + len(df1_100) + len(dm2_100) + len(df2_100) + len(dm3_100) + len(df3_100) + len(dm4_100) + len(df4_100)
	dist_0 = len(dm0_100) + len(df0_100)
	dist_1 = len(dm1_100) + len(df1_100)
	dist_2 = len(dm2_100) + len(df2_100)
	dist_3 = len(dm3_100) + len(df3_100)
	dist_4 = len(dm4_100) + len(df4_100)

	############################################################################################
	# 25 percent dark-skinned faces
	lm0_25 = light_skinned_faces_male_0.copy()
	lf0_25 = light_skinned_faces_female_0.copy()
	lm1_25 = light_skinned_faces_male_1.copy()
	lf1_25 = light_skinned_faces_female_1.copy()
	lm2_25 = light_skinned_faces_male_2.copy()
	lf2_25 = light_skinned_faces_female_2.copy()
	lm3_25 = light_skinned_faces_male_3.copy()
	lf3_25 = light_skinned_faces_female_3.copy()
	lm4_25 = light_skinned_faces_male_4.copy()
	lf4_25 = light_skinned_faces_female_4.copy()

	dm0_25 = dark_skinned_faces_male_0.copy()
	df0_25 = dark_skinned_faces_female_0.copy()
	dm1_25 = dark_skinned_faces_male_1.copy()
	df1_25 = dark_skinned_faces_female_1.copy()
	dm2_25 = dark_skinned_faces_male_2.copy()
	df2_25 = dark_skinned_faces_female_2.copy()
	dm3_25 = dark_skinned_faces_male_3.copy()
	df3_25 = dark_skinned_faces_female_3.copy()
	dm4_25 = dark_skinned_faces_male_4.copy()
	df4_25 = dark_skinned_faces_female_4.copy()

	train_25 = []

	select_for_training_set_category(lm0_25, train_25, round((dist_0 * 0.75)/ 2))
	select_for_training_set_category(lf0_25, train_25, round((dist_0 * 0.75)/ 2))
	select_for_training_set_category(lm1_25, train_25, round((dist_1 * 0.75)/ 2))
	select_for_training_set_category(lf1_25, train_25, round((dist_1 * 0.75)/ 2))
	select_for_training_set_category(lm2_25, train_25, round((dist_2 * 0.75)/ 2))
	select_for_training_set_category(lf2_25, train_25, round((dist_2 * 0.75)/ 2))
	select_for_training_set_category(lm3_25, train_25, round((dist_3 * 0.75)/ 2))
	select_for_training_set_category(lf3_25, train_25, round((dist_3 * 0.75)/ 2))
	select_for_training_set_category(lm4_25, train_25, round((dist_4 * 0.75)/ 2))
	select_for_training_set_category(lf4_25, train_25, round((dist_4 * 0.75)/ 2))

	select_for_training_set_category(dm0_25, train_25, round((dist_0 * 0.25)/ 2))
	select_for_training_set_category(df0_25, train_25, round((dist_0 * 0.25)/ 2))
	select_for_training_set_category(dm1_25, train_25, round((dist_1 * 0.25)/ 2))
	select_for_training_set_category(df1_25, train_25, round((dist_1 * 0.25)/ 2))
	select_for_training_set_category(dm2_25, train_25, round((dist_2 * 0.25)/ 2))
	select_for_training_set_category(df2_25, train_25, round((dist_2 * 0.25)/ 2))
	select_for_training_set_category(dm3_25, train_25, round((dist_3 * 0.25)/ 2))
	select_for_training_set_category(df3_25, train_25, round((dist_3 * 0.25)/ 2))
	select_for_training_set_category(dm4_25, train_25, round((dist_4 * 0.25)/ 2))
	select_for_training_set_category(df4_25, train_25, round((dist_4 * 0.25)/ 2))

	############################################################################################
	# 50 percent dark-skinned faces

	lm0_50 = light_skinned_faces_male_0.copy()
	lf0_50 = light_skinned_faces_female_0.copy()
	lm1_50 = light_skinned_faces_male_1.copy()
	lf1_50 = light_skinned_faces_female_1.copy()
	lm2_50 = light_skinned_faces_male_2.copy()
	lf2_50 = light_skinned_faces_female_2.copy()
	lm3_50 = light_skinned_faces_male_3.copy()
	lf3_50 = light_skinned_faces_female_3.copy()
	lm4_50 = light_skinned_faces_male_4.copy()
	lf4_50 = light_skinned_faces_female_4.copy()

	dm0_50 = dark_skinned_faces_male_0.copy()
	df0_50 = dark_skinned_faces_female_0.copy()
	dm1_50 = dark_skinned_faces_male_1.copy()
	df1_50 = dark_skinned_faces_female_1.copy()
	dm2_50 = dark_skinned_faces_male_2.copy()
	df2_50 = dark_skinned_faces_female_2.copy()
	dm3_50 = dark_skinned_faces_male_3.copy()
	df3_50 = dark_skinned_faces_female_3.copy()
	dm4_50 = dark_skinned_faces_male_4.copy()
	df4_50 = dark_skinned_faces_female_4.copy()

	train_50 = []

	select_for_training_set_category(lm0_50, train_50, round((dist_0 * 0.5)/ 2))
	select_for_training_set_category(lf0_50, train_50, round((dist_0 * 0.5)/ 2))
	select_for_training_set_category(lm1_50, train_50, round((dist_1 * 0.5)/ 2))
	select_for_training_set_category(lf1_50, train_50, round((dist_1 * 0.5)/ 2))
	select_for_training_set_category(lm2_50, train_50, round((dist_2 * 0.5)/ 2))
	select_for_training_set_category(lf2_50, train_50, round((dist_2 * 0.5)/ 2))
	select_for_training_set_category(lm3_50, train_50, round((dist_3 * 0.5)/ 2))
	select_for_training_set_category(lf3_50, train_50, round((dist_3 * 0.5)/ 2))
	select_for_training_set_category(lm4_50, train_50, round((dist_4 * 0.5)/ 2))
	select_for_training_set_category(lf4_50, train_50, round((dist_4 * 0.5)/ 2))

	select_for_training_set_category(dm0_50, train_50, round((dist_0 * 0.5)/ 2))
	select_for_training_set_category(df0_50, train_50, round((dist_0 * 0.5)/ 2))
	select_for_training_set_category(dm1_50, train_50, round((dist_1 * 0.5)/ 2))
	select_for_training_set_category(df1_50, train_50, round((dist_1 * 0.5)/ 2))
	select_for_training_set_category(dm2_50, train_50, round((dist_2 * 0.5)/ 2))
	select_for_training_set_category(df2_50, train_50, round((dist_2 * 0.5)/ 2))
	select_for_training_set_category(dm3_50, train_50, round((dist_3 * 0.5)/ 2))
	select_for_training_set_category(df3_50, train_50, round((dist_3 * 0.5)/ 2))
	select_for_training_set_category(dm4_50, train_50, round((dist_4 * 0.5)/ 2))
	select_for_training_set_category(df4_50, train_50, round((dist_4 * 0.5)/ 2))

	############################################################################################
	# 75 percent dark-skinned faces
	lm0_75 = light_skinned_faces_male_0.copy()
	lf0_75 = light_skinned_faces_female_0.copy()
	lm1_75 = light_skinned_faces_male_1.copy()
	lf1_75 = light_skinned_faces_female_1.copy()
	lm2_75 = light_skinned_faces_male_2.copy()
	lf2_75 = light_skinned_faces_female_2.copy()
	lm3_75 = light_skinned_faces_male_3.copy()
	lf3_75 = light_skinned_faces_female_3.copy()
	lm4_75 = light_skinned_faces_male_4.copy()
	lf4_75 = light_skinned_faces_female_4.copy()

	dm0_75 = dark_skinned_faces_male_0.copy()
	df0_75 = dark_skinned_faces_female_0.copy()
	dm1_75 = dark_skinned_faces_male_1.copy()
	df1_75 = dark_skinned_faces_female_1.copy()
	dm2_75 = dark_skinned_faces_male_2.copy()
	df2_75 = dark_skinned_faces_female_2.copy()
	dm3_75 = dark_skinned_faces_male_3.copy()
	df3_75 = dark_skinned_faces_female_3.copy()
	dm4_75 = dark_skinned_faces_male_4.copy()
	df4_75 = dark_skinned_faces_female_4.copy()

	train_75 = []

	select_for_training_set_category(lm0_75, train_75, round((dist_0 * 0.25)/ 2))
	select_for_training_set_category(lf0_75, train_75, round((dist_0 * 0.25)/ 2))
	select_for_training_set_category(lm1_75, train_75, round((dist_1 * 0.25)/ 2))
	select_for_training_set_category(lf1_75, train_75, round((dist_1 * 0.25)/ 2))
	select_for_training_set_category(lm2_75, train_75, round((dist_2 * 0.25)/ 2))
	select_for_training_set_category(lf2_75, train_75, round((dist_2 * 0.25)/ 2))
	select_for_training_set_category(lm3_75, train_75, round((dist_3 * 0.25)/ 2))
	select_for_training_set_category(lf3_75, train_75, round((dist_3 * 0.25)/ 2))
	select_for_training_set_category(lm4_75, train_75, round((dist_4 * 0.25)/ 2))
	select_for_training_set_category(lf4_75, train_75, round((dist_4 * 0.25)/ 2))

	select_for_training_set_category(dm0_75, train_75, round((dist_0 * 0.75)/ 2))
	select_for_training_set_category(df0_75, train_75, round((dist_0 * 0.75)/ 2))
	select_for_training_set_category(dm1_75, train_75, round((dist_1 * 0.75)/ 2))
	select_for_training_set_category(df1_75, train_75, round((dist_1 * 0.75)/ 2))
	select_for_training_set_category(dm2_75, train_75, round((dist_2 * 0.75)/ 2))
	select_for_training_set_category(df2_75, train_75, round((dist_2 * 0.75)/ 2))
	select_for_training_set_category(dm3_75, train_75, round((dist_3 * 0.75)/ 2))
	select_for_training_set_category(df3_75, train_75, round((dist_3 * 0.75)/ 2))
	select_for_training_set_category(dm4_75, train_75, round((dist_4 * 0.75)/ 2))
	select_for_training_set_category(df4_75, train_75, round((dist_4 * 0.75)/ 2))

	############################################################################################
	# 0 percent dark-skinned faces
	# 75 percent dark-skinned faces
	lm0_0 = light_skinned_faces_male_0.copy()
	lf0_0 = light_skinned_faces_female_0.copy()
	lm1_0 = light_skinned_faces_male_1.copy()
	lf1_0 = light_skinned_faces_female_1.copy()
	lm2_0 = light_skinned_faces_male_2.copy()
	lf2_0 = light_skinned_faces_female_2.copy()
	lm3_0 = light_skinned_faces_male_3.copy()
	lf3_0 = light_skinned_faces_female_3.copy()
	lm4_0 = light_skinned_faces_male_4.copy()
	lf4_0 = light_skinned_faces_female_4.copy()
	train_0 = []

	select_for_training_set_category(lm0_0, train_0, round((dist_0)/ 2))
	select_for_training_set_category(lf0_0, train_0, round((dist_0)/ 2))
	select_for_training_set_category(lm1_0, train_0, round((dist_1)/ 2))
	select_for_training_set_category(lf1_0, train_0, round((dist_1)/ 2))
	select_for_training_set_category(lm2_0, train_0, round((dist_2)/ 2))
	select_for_training_set_category(lf2_0, train_0, round((dist_2)/ 2))
	select_for_training_set_category(lm3_0, train_0, round((dist_3)/ 2))
	select_for_training_set_category(lf3_0, train_0, round((dist_3)/ 2))
	select_for_training_set_category(lm4_0, train_0, round((dist_4)/ 2))
	select_for_training_set_category(lf4_0, train_0, round((dist_4)/ 2))

	return train_0, train_25, train_50, train_75, train_100, testing_set

def hold_back_testing_set(training_set):
	for i in range(NUM_OF_TESTING_PER_CATEGORY):
		random_index = random.randint(0, len(training_set) - 1)
		testing_set.append(training_set[random_index])
		del training_set[random_index]

def select_for_training_set_category(set_to_choose_from, set_to_add_to, num_images_to_select):
	for i in range(num_images_to_select):
		random_index = random.randint(0, len(set_to_choose_from) - 1)
		set_to_add_to.append(set_to_choose_from[random_index])
		del set_to_choose_from[random_index]

def test_data(train_0, train_25, train_50, train_75, train_100, testing_set):
	# Should each be around 3870 images
	sets = [train_0, train_25, train_50, train_75, train_100, testing_set]
	for data_set in sets:
		print(len(data_set))

	# Check distribution of light and dark skinned faces
	for data_set in sets:
		dark = []
		light = []
		for image in data_set:
			if (re.match(r'UTKFace/([0-9]|[0-9][0-9]|[0-9][0-9][0-9])_[0-1]_1_(.*)', image)):
				# All dark skinned faces, for testing purposes
				dark.append(image)
			if (re.match(r'UTKFace/([0-9]|[0-9][0-9]|[0-9][0-9][0-9])_[0-1]_0_(.*)', image)):
				# All dark skinned faces, for testing purposes
				light.append(image)
		print("Check distribution of dark and light skin")
		print(len(dark))
		print(len(light))

	# Check equal gender distrbution 
	for data_set in sets:
		dark_m = []
		light_m = []
		dark_f = []
		light_f = []
		for image in data_set:
			if (re.match(r'UTKFace/([0-9]|[0-9][0-9]|[0-9][0-9][0-9])_0_1_(.*)', image)):
				dark_m.append(image)
			if (re.match(r'UTKFace/([0-9]|[0-9][0-9]|[0-9][0-9][0-9])_1_1_(.*)', image)):
				dark_f.append(image)
			if (re.match(r'UTKFace/([0-9]|[0-9][0-9]|[0-9][0-9][0-9])_0_0_(.*)', image)):
				light_m.append(image)
			if (re.match(r'UTKFace/([0-9]|[0-9][0-9]|[0-9][0-9][0-9])_1_0_(.*)', image)):
				light_f.append(image)
		print("Check gender distrbution")
		print(len(dark_m) == len(dark_f))
		print(len(light_m) == len(light_f))


def main():
    train_0, train_25, train_50, train_75, train_100, test = get_training_sets()
    # test_data(train_0, train_25, train_50, train_75, train_100, test)

    # Save data
    # sets = [train_0, train_25, train_50, train_75, train_100, test]
    # sets_names = ["train_0/", "train_25/", "train_50/", "train_75/", "train_100/", "tests/"]
    # for i in range(len(sets)):
    # 	folder_name = "data/" + sets_names[i]
    # 	os.makedirs(folder_name)
    # 	for im in sets[i]:
    # 		face = cv2.imread(im, cv2.IMREAD_GRAYSCALE)
    # 		cv2.imwrite(os.path.join(folder_name, im[8:]), face)
		
if __name__ == '__main__':
    main()



