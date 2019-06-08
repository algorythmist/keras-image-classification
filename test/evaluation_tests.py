import unittest
import tensorflow as tf
from image_classifier import preprocess, load_model


class EvaluationTestCase(unittest.TestCase):

    def test_evaluate_mnist(self):
        model = load_model('mnist_simpleMLP_128hidden.h5')
        (_, _), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
        test_images = preprocess(test_images)
        (loss, accuracy) = model.evaluate(test_images, test_labels)
        self.assertTrue(accuracy > 0.97)

    def test_evaluate_fashion(self):
        model = load_model('fashion_simpleCNN_128hidden.h5')
        (_, _), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
        test_images = preprocess(test_images)
        (loss, accuracy) = model.evaluate(test_images, test_labels)
        self.assertTrue(accuracy > 0.91)


if __name__ == '__main__':
    unittest.main()
