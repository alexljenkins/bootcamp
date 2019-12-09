"""
A simple VG16 CNN example that takes in an image and returns the
top 5 classifications and their probabilities.
Image must be of one of the 1,000 VGG initially trained on
(see ImageNet for more details and full list).
"""
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions


def image_loader(filename):
    """
    Loads the filename image in and coverts it
    to the format required for vgg16
    """
    # load an image from file as numpy array
    image = img_to_array(load_img(filename, target_size=(224, 224)))
    # reshape image dimensions
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)

    return image


def vgg16_standard_classifier(image):
    """
    Classifies images based on the 1000 outputs VGG16
    was designed to classify.
    """
    # load the model
    model = VGG16()
    print(model.summary())
    # predict the probability across all output classes
    yhat = model.predict(image)
    # convert the probabilities to class labels
    label = decode_predictions(yhat)

    return label[0]


if __name__ == '__main__':

    image = image_loader('data/mug.jfif')

    outcome = vgg16_standard_classifier(image)

    #print out the top 5 results and their probablities
    for i in range(5):
        print(f'{outcome[i][1]}: {round(outcome[i][2]*100,2)}%')
