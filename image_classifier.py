import tensorflow as tf
import numpy as np
import os

def normalize(data):
    return data/data.max()


def buildSimpleMLP(hidden, outputs):
    return tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(hidden, activation=tf.nn.relu),
        tf.keras.layers.Dense(outputs, activation=tf.nn.softmax)
    ])


def buildCNN(hidden, outputs):
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(hidden, activation='relu'),
        tf.keras.layers.Dense(outputs, activation='softmax')
    ])


def reshape(data):
    sizes = np.concatenate((data.shape, 1), axis=None)
    return data.reshape(sizes)


def preprocess(data):
    return reshape(normalize(data))


def unique_labels(labels):
    return len(set(labels))


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def save_model(model, name):
    file = os.path.join(ROOT_DIR, 'models', name+'.h5')
    model.save(file)


def train_and_save(m, name, training_images, training_labels, epochs=10):
    m.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    m.fit(training_images, training_labels, epochs=epochs)
    m.summary()
    save_model(m, name)


def load_model(filename):
    file = os.path.join(ROOT_DIR, 'models', filename)
    return tf.keras.models.load_model(file)


if __name__ == '__main__':
    mnist = tf.keras.datasets.fashion_mnist
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data()

    labels = len(set(training_labels))

    training_images = preprocess(training_images)
    test_images = preprocess(test_images)


    model = buildSimpleMLP(128, labels)

    train_and_save(model, "simplemlp_128_hidden", training_images, training_labels)

    test_loss = model.evaluate(test_images, test_labels)
    print(test_loss)