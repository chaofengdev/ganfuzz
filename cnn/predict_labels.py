import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def display_images_with_labels(images, labels, num_examples):
    plt.figure(figsize=(10, 10))
    for i in range(num_examples):
        plt.subplot(10, 10, i + 1)
        plt.imshow(images[i, :, :, 0], cmap='gray')
        plt.title(labels[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    generated_images = np.load('generated_images.npy')

    # 使用 tf.saved_model 加载模型
    mnist_classifier = tf.keras.models.load_model('mnist_classifier')
    predicted_labels = mnist_classifier.predict(generated_images)
    predicted_labels = np.argmax(predicted_labels, axis=1)

    display_images_with_labels(generated_images, predicted_labels, 100)
