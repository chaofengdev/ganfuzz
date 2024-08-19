import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import os

# 确保使用 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 设置 GPU 内存增长  --这里都是为了缓解gpu显存不够的权宜之计。
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


# 定义MNIST分类模型
def create_mnist_classifier():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# 定义测试方法
def test_model(model, test_images, test_labels, save_dir='classifier_model_test_output'):
    # 随机挑选10张图片
    indices = np.random.choice(len(test_images), 10)
    sample_images = test_images[indices]
    sample_labels = test_labels[indices]

    # 预测类别
    predictions = model.predict(sample_images)
    predicted_labels = np.argmax(predictions, axis=1)

    # 确保保存图片的目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 创建一个大图，并将每个小图嵌入其中
    fig, axes = plt.subplots(5, 2, figsize=(10, 15))
    for i, ax in enumerate(axes.flat):
        ax.imshow(sample_images[i].reshape(28, 28), cmap=plt.cm.binary)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f'Predicted: {predicted_labels[i]}, True: {sample_labels[i]}')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'combined_image.png'))
    plt.show()

    # 计算预测的准确率
    accuracy = np.sum(predicted_labels == sample_labels) / 10
    print(f'随机抽样的10张图片的预测准确率: {accuracy:.4f}')


if __name__ == '__main__':
    # 加载MNIST数据集
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    # 预处理数据
    train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
    test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

    # 创建MNIST分类模型
    mnist_classifier = create_mnist_classifier()
    # 训练模型
    mnist_classifier.fit(train_images, train_labels, epochs=5, batch_size=64,
                         validation_data=(test_images, test_labels))

    # 在测试集上评估模型
    test_loss, test_accuracy = mnist_classifier.evaluate(test_images, test_labels)
    print(f'在测试集上的准确率: {test_accuracy:.4f}')

    # 确保保存模型的目录存在
    save_dir = 'classifier_model'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 保存模型为HDF5格式
    save_path = os.path.join(save_dir, 'mnist_classifier.h5')
    mnist_classifier.save(save_path)

    # 测试模型并保存图片
    test_model(mnist_classifier, test_images, test_labels)
