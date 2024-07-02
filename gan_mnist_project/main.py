import numpy as np
import tensorflow as tf
import random
from models.gan import make_generator_model, make_discriminator_model
from models.classifier import make_classifier_model
from mutation.mutation_algorithms import gaussian_mutation, uniform_mutation, boundary_mutation, polynomial_mutation
from utils.image_utils import save_images

# 加载MNIST数据集
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = (x_train.astype('float32') - 127.5) / 127.5
x_test = (x_test.astype('float32') - 127.5) / 127.5
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

print("数据集加载完毕")
print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")

# 定义GAN模型
generator = make_generator_model()
discriminator = make_discriminator_model()
print("GAN模型定义完毕")

# 定义分类器
classifier = make_classifier_model()
classifier.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))
print("分类器训练完毕")

# GAN训练逻辑
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


@tf.function
def train_step(images):
    noise = tf.random.normal([128, 100])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
        disc_loss = cross_entropy(tf.ones_like(real_output), real_output) + cross_entropy(tf.zeros_like(fake_output),
                                                                                          fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


# 训练GAN
def train_gan(dataset, epochs):
    for epoch in range(epochs):
        for image_batch in dataset:
            train_step(image_batch)
        print(f"Epoch {epoch + 1}/{epochs} 训练完毕")


# 创建种子向量队列
seed_queue = [tf.random.normal([100]) for _ in range(20)]
average_confidence = 0.5
print("种子向量队列创建完毕")

# 主循环
for iteration in range(500):
    print(f"迭代 {iteration + 1}/500 开始")
    # 随机选择一个向量并进行变异
    seed_vector = random.choice(seed_queue)
    print(f"选中的种子向量: {seed_vector}")
    mutated_vector = gaussian_mutation(seed_vector)
    print(f"变异后的向量: {mutated_vector}")

    # 生成图片
    generated_image = generator(np.expand_dims(mutated_vector, axis=0), training=False)
    print(f"生成的图片 shape: {generated_image.shape}")

    # 分类图片
    prediction = classifier.predict(generated_image)
    predicted_label = np.argmax(prediction)
    confidence = np.max(prediction)
    print(f"预测标签: {predicted_label}, 置信度: {confidence}")

    # 更新种子队列
    if confidence < average_confidence:
        seed_queue.append(mutated_vector)
        seed_queue.pop(0)
        average_confidence = np.mean(
            [np.max(classifier.predict(generator(np.expand_dims(vec, axis=0), training=False))) for vec in seed_queue])
        print(f"更新后的平均置信度: {average_confidence}")

    if iteration % 100 == 0:
        print(f"Iteration: {iteration}, Average Confidence: {average_confidence}")

    if average_confidence < 0.1:
        print("平均置信度低于0.2，停止迭代")
        break

# 保存生成的图片
final_images = [generator(np.expand_dims(vec, axis=0), training=False) for vec in seed_queue[-20:]]
final_images = np.array(final_images)
save_images(final_images, 'final_generated_images.png')
print("生成的图片已保存")
