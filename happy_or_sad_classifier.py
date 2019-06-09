import os
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from dataset_loader import prepare_dataset
from image_classifier import save_model

def build_small_model():
    return tf.keras.models.Sequential([
        # Note the input shape is the desired size of the image 300x300 with 3 bytes color
        # This is the first convolution
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(300, 300, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The second convolution
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The third convolution
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The fourth convolution
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The fifth convolution
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # Flatten the results to feed into a DNN
        tf.keras.layers.Flatten(),
        # 512 neuron hidden layer
        tf.keras.layers.Dense(512, activation='relu'),
        # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('horses') and 1 for the other ('humans')
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

if __name__ == '__main__':
    directory = prepare_dataset('happy-or-sad')

    # Directory with our training horse pictures
    train_happy_dir = os.path.join(directory, 'happy')

    # Directory with our training human pictures
    train_sad_dir = os.path.join(directory, 'sad')

    # build the model
    model = build_small_model()
    model.summary()

    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(lr=0.001),
                  metrics=['acc'])

    # All images will be rescaled by 1./255
    train_datagen = ImageDataGenerator(rescale=1 / 255)

    # Flow training images in batches of 128 using train_datagen generator
    train_generator = train_datagen.flow_from_directory(
        directory,  # This is the source directory for training images
        target_size=(300, 300),  # All images will be resized to 150x150
        batch_size=128,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=5,
        epochs=10,
        verbose=1)

    save_model(model, 'happy_or_sad_cnn')