import Datapack as d
import Model as m
import Train as t
import matplotlib.pyplot as plt
import random

if __name__ == '__main__':
    mnist_train_dataset = d.Datapack().download_fasion_minst(data_for_train=True)
    mnist_test_dataset = d.Datapack().download_fasion_minst(data_for_train=False)
    mnist_test_dataset_classes = mnist_test_dataset.classes
    train_data = d.Datapack().load_fasion_mnist(mnist_train_dataset)
    test_data = d.Datapack().load_fasion_mnist(mnist_test_dataset)

    model = m.Model(input_features=784, number_of_nodes=10, output_features=len(mnist_train_dataset.classes))
    t.Train().train_model(model=model, training_data=train_data, testing_data=test_data, number_of_epoch=10)

    random_number = random.randint(0, len(mnist_test_dataset))
    random_feature, random_label = mnist_test_dataset[random_number]
    test_logit = model(random_feature)
    random_feature = random_feature.permute(1, 2, 0)
    fig = plt.figure(figsize=(6,3))
    fig.add_subplot(1, 2, 1)
    plt.imshow(random_feature, cmap='gray')
    plt.title("Real Name: "+mnist_test_dataset_classes[random_label])

    fig.add_subplot(1, 2, 2)
    plt.imshow(random_feature, cmap='gray')
    plt.title("Predicted Name: "+mnist_test_dataset_classes[test_logit.argmax(dim=1)])
    plt.show()


