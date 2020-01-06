from test_face_classifier import test_face_classifier

print('train 0%')
test_face_classifier(300, 300, 9, False, 'train_0')
print('train 25%')
test_face_classifier(300, 300, 9, False, 'train_25')
print('train 50%')
test_face_classifier(300, 300, 9, False, 'train_50')
print('train 75%')
test_face_classifier(300, 300, 9, False, 'train_75')
print('train 100%')
test_face_classifier(300, 300, 9, False, 'train_100')
