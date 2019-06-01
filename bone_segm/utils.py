import numpy as np
import glob
import os
np.random.seed(0)

def data_generator(images, batch_size):
    num_samples = len(images)
    while True:
        try:
            np.random.shuffle(images)
            for i in range(0, num_samples, batch_size):
                x_data = []
                y_data = []
                for im in images[i:i+batch_size]:
                    data = np.load(im)
                    x_data.append(np.stack((data[0], ) * 3, -1))
                    y_data.append(data[1])

                yield np.array(x_data), np.array(y_data)

        except Exception as err:
            print(err)


def split_train_test_data(path, validation_size):
    images = glob.glob(os.path.join(path, "**", "*"))
    np.random.shuffle(images)
    images = images[150]

    num_examples = len(images)
    valid_examples = int(num_examples * validation_size)

    return images[valid_examples:], images[:valid_examples]
