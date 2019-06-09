import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from horse_or_human_loader import prepare_dataset
from image_classifier import load_model

if __name__ == '__main__':
    directory = prepare_dataset('validation-horse-or-human')
    # Directory with our training horse pictures
    validation_horse_dir = os.path.join(directory, 'horses')

    # Directory with our training human pictures
    validation_human_dir = os.path.join(directory, 'humans')

    validation_datagen = ImageDataGenerator(rescale=1 / 255)
    # Flow training images in batches of 128 using train_datagen generator
    validation_generator = validation_datagen.flow_from_directory(
        directory,
        target_size=(300, 300),  # All images will be resized to 150x150
        batch_size=32,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

    model = load_model('horse_or_human_cnn.h5')
    eval = model.evaluate_generator(validation_generator, steps = 8, workers=10)
    print(eval)