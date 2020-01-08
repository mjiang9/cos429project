from test_face_classifier import test_face_classifier_l

print('train 0%')
test_face_classifier_l(300, 300, 9, False, 'train_0', 'hard_nonfaces')
print('train 25%')
test_face_classifier_l(300, 300, 9, False, 'train_25', 'hard_nonfaces')
print('train 50%')
test_face_classifier_l(300, 300, 9, False, 'train_50', 'hard_nonfaces')
print('train 75%')
test_face_classifier_l(300, 300, 9, False, 'train_75', 'hard_nonfaces')
print('train 100%')
test_face_classifier_l(300, 300, 9, False, 'train_100', 'hard_nonfaces')
