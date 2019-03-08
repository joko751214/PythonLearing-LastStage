import os
import sys
import numpy as np
import cv2 as cv
import sklearn.preprocessing as sp


def search_faces(directory):
    if not os.path.isdir(directory):
        raise IOError(
            "The directory '" + directory + "' doesn't exist!")
    faces = {}
    for curdir, subdirs, files in os.walk(directory):
        for jpeg in (file for file in files if file.endswith('.jpg')):
            path = os.path.join(curdir, jpeg)
            label = path.split(os.path.sep)[-2]
            if label not in faces:
                faces[label] = []
            faces[label].append(path)
    return faces


def train_codec(labels):
    # 將字符串變成數據
    codec = sp.LabelEncoder()
    codec.fit(labels)
    return codec


def load_detectors():
    face_detector = cv.CascadeClassifier('haar\\face.xml')
    return face_detector


def read_image(filename):
    image = cv.imread(filename)
    return image


def bgr2gray(image):
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return image


def detect_regions(detector, image):
    
    regions = detector.detectMultiScale(image, 1.1, 2,
                                        minSize=(100, 100))
    return regions


def encode(codec, label):
    code = int(codec.transform([label])[0])
    return code


def show_image(title, image):
    cv.imshow(title, image)


def decode(codec, code):
    label = codec.inverse_transform(code)
    return label


def wait_escape(delay=0):
    return cv.waitKey(delay) == 27


def read_data(directory):
    faces = search_faces(directory)
    codec = train_codec(list(faces.keys()))
    face_detector = load_detectors()
    x, y, z = [], [], []
    for label, filenames in faces.items():
        for filename in filenames:
            print(filename, '->', label)
            original = read_image(filename)
            gray = bgr2gray(original)
            faces = detect_regions(face_detector, gray)
            for l, t, w, h in faces:
                x.append(gray[t:t + h, l:l + w])
                y.append(encode(codec, label))
                a, b = int(w / 2), int(h / 2)
                cv.ellipse(original, (l + a, t + b), (a, b),
                           0, 0, 360, (255, 0, 255), 2)
                z.append(original)
    y = np.array(y)
    return codec, x, y, z


def train_model(x, y):
    # 人臉識別器
    model = cv.face.LBPHFaceRecognizer_create()
    model.train(x, y)
    return model


def pred_model(model, x):
    y = []
    for face in x:
        y.append(model.predict(face)[0])
    return y


def show_labels(codec, codes, pred_codes, images):
    escape = False
    while not escape:
        for code, pred_code, image in zip(codes, pred_codes, images):
            # 在圖片上放上文本
            cv.putText(image, '{} {} {}'.format(
                decode(codec, code),
                '==' if code == pred_code else '!=',
                decode(codec, pred_code)), (10, 60),
                cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 6)
            show_image('Recognizing Face...', image)
            if wait_escape(1000):
                escape = True
                break


def main(argc, argv, envp):
    codec, train_x, train_y, train_z = read_data('faces\\training')
    _, test_x, test_y, test_z = read_data('faces\\testing')
    model = train_model(train_x, train_y)
    pred_test_y = pred_model(model, test_x)
    show_labels(codec, test_y, pred_test_y, test_z)
    return 0


if __name__ == '__main__':
    sys.exit(main(len(sys.argv), sys.argv, os.environ))
