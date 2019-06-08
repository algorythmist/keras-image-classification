import unittest
import tensorflow as tf

from image_classifier import preprocess, unique_labels, buildSimpleMLP, buildCNN, train_and_save

class ClassificationTestCase(unittest.TestCase):

    def test_mnist_with_simple_mlp(self):
        data = tf.keras.datasets.mnist
        (training_images, training_labels), (test_images, test_labels) = data.load_data()
        training_images = preprocess(training_images)
        test_images = preprocess(test_images)
        model = buildSimpleMLP(128, unique_labels(training_labels))
        train_and_save(model, "mnist_simpleMLP_128hidden", training_images, training_labels)

        (loss, accuracy) = model.evaluate(test_images, test_labels)
        self.assertTrue(accuracy > 0.97)

    def test_fashion_mnist_with_simple_mlp(self):
        data = tf.keras.datasets.fashion_mnist
        (training_images, training_labels), (test_images, test_labels) = data.load_data()
        training_images = preprocess(training_images)
        test_images = preprocess(test_images)
        model = buildSimpleMLP(128, unique_labels(training_labels))

        train_and_save(model, "fashion_simpleMLP_128hidden", training_images, training_labels)
        (loss, accuracy) = model.evaluate(test_images, test_labels)
        self.assertTrue(accuracy < 0.90)

    def test_fashion_mnist_with_simple_cnn(self):
        data = tf.keras.datasets.fashion_mnist
        (training_images, training_labels), (test_images, test_labels) = data.load_data()
        training_images = preprocess(training_images)
        test_images = preprocess(test_images)
        model = buildCNN(128, unique_labels(training_labels))
        train_and_save(model, "fashion_simpleCNN_128hidden", training_images, training_labels)

        (loss, accuracy) = model.evaluate(test_images, test_labels)
        print(accuracy)
        self.assertTrue(accuracy > 0.91)




if __name__ == '__main__':
    unittest.main()
