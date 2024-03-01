import time

import numpy as np
import os
import matplotlib.pyplot as plt
from random import seed, randint, shuffle
import cv2
from keras.models import Sequential
from keras.layers import Dense, Flatten, Input
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.models import load_model


def show_img(dirpath: str, count: int = 5, random: bool = True, name: str = 'picture'):
    """
    :param dirpath: Path to directory of images,
    :param count: how many pictures to show, if None: show all,
    :param random: if False: show first img, if True: show random img,
    :param name: name of img,
    :return:
    """
    if count is None:
        count = len(list(os.walk(dirpath))[2])
    if count < 5:
        columns = count
        rows = 1
    else:
        columns = 5
        if count % 5 == 0:
            rows = count // 5
        else:
            rows = 1 + count // 5
    fig_size_y = 2 + rows * 3
    fig = plt.figure(figsize=(16, fig_size_y))
    picture_name_list = list(os.walk(dirpath))[0][2]
    selected_picture = {}
    if random:
        i = 0
        while i < count:
            index = randint(0, len(picture_name_list) - 1)
            if selected_picture.get(index) is None:
                selected_picture[index] = picture_name_list[index]
                i += 1
    else:
        for i in range(count):
            selected_picture[i] = picture_name_list[i]
    iter_picture = iter(selected_picture.values())
    for i in range(rows):
        for j in range(columns):
            fig.add_subplot(rows, columns, 1+i*columns+j)
            img = plt.imread(os.path.join(dirpath, next(iter_picture)))
            plt.imshow(img)
            plt.axis('off')
            plt.title(f'{name} {1+i*columns+j}')
    random_str = 'random' if random else "first"
    fig.suptitle(f"""Here we see {count} {random_str} picture from {name}
There are {len(picture_name_list)} in total""", fontsize=14, bbox={'facecolor': 'grey', 'alpha': 0.5})
    plt.show()


def get_and_transform_img(path, need_resize=False, size=(32, 32)):
    images = []
    for filename in list(os.walk(path))[0][2]:
        img = cv2.imread(os.path.join(path, filename))
        if need_resize:
            img = cv2.resize(img, size)
        img = img.astype('float32') / 255.0
        images.append(img)
    images = np.array(images)
    return images


def create_vec_y(i, n):
    v = [0 for i in range(n)]
    v[i] = 1
    return v


def create_x_and_y(*arrays, mix=True, random_seed=7, classes_count=2):
    x_array = []
    y_array = []
    index_ranges = []
    start_range = 0
    end_range = 0
    for array in arrays:
        start_range = end_range
        end_range += len(array)
        index_ranges.append([start_range, end_range])
    list_of_index = [x for x in range(end_range)]
    if mix:
        seed(random_seed)
        shuffle(list_of_index)
    for index in list_of_index:
        index_array = 0
        for index_range in index_ranges:
            if index_range[0] <= index < index_range[1]:
                index -= index_range[0]
                break
            index_array += 1
        y_array.append(create_vec_y(index_array, classes_count))
        x_array.append(arrays[index_array][index])
    return np.array(x_array), np.array(y_array)


def create_model(neurons_num=1024, activate_func='relu'):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Flatten())
    model.add(Dense(neurons_num, activation=activate_func))
    model.add(Dense(classes_count, activation='softmax'))
    model._name = 'Seq'
    return model


def result(model):
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=4, mode='max', min_delta=0.0001,
                                  cooldown=0, min_lr=0, verbose=1)
    early_stop = EarlyStopping(monitor='val_accuracy', mode='max', patience=5, verbose=1)
    save_best = ModelCheckpoint(filepath=f'models/best_{model._name}_of_train.keras', monitor='val_accuracy', mode='max',
                                save_best_only=True, save_weights_only=False)
    start_time = time.time()
    history = model.fit(x_train, y_train, callbacks=[reduce_lr, early_stop, save_best], batch_size=batch, epochs=epochs,
                        validation_data=(x_test, y_test), shuffle=True, verbose=1)
    train_time = time.time() - start_time
    start_time = time.time()
    test_loss, test_acc = model.evaluate(x_test, y_test)
    test_time = time.time() - start_time
    info_about_learning = ""
    if os.path.exists(f'models/best_{model._name}.keras'):
        model_best = load_model(f'models/best_{model._name}.keras')
        acc_best = model_best.evaluate(x_test, y_test)[1]
        if test_acc > acc_best:
            model.save(f'models/best_{model._name}.keras')
            info_about_learning = f"New best accuracy is {round(100 * test_acc, 2)}% and was {round(100 * acc_best, 2)}%"
        else:
            info_about_learning = f"Best accuracy is {round(100 * acc_best, 2)}% and current is {round(100 * test_acc, 2)}%"

    else:
        model.save(f'models/best_{model._name}.keras')

    model.summary()

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()
    print('\nTrain time: ', train_time)
    print('Test loss:', test_loss)
    print('Test time: ', test_time)
    print('Test accuracy: ' + str(round(100 * test_acc, 2)) + "%")
    print()
    print(info_about_learning)


need_show_data = True
need_show_img = True
need_resize = True
size_to_resize = (32, 32)
if need_resize:
    input_shape = (32, 32, 3)
else:
    input_shape = (1000, 1000, 3)
random_seed = 7
classes_count = 2
need_mix_train = True
need_mix_test = False

fruit_train_path = "fruitvsvegetable/train/0"
veg_train_path = "fruitvsvegetable/train/1"
fruit_test_path = "fruitvsvegetable/test/0"
veg_test_path = "fruitvsvegetable/test/1"

if need_show_img:
    show_img(fruit_train_path, 5, name='fruit_train')
    show_img(veg_train_path, 5, name='veg_train')
    show_img(fruit_test_path, 5, name='fruit_test')
    show_img(veg_test_path, 5, name='veg_test')

start_time = time.time()
fruit_train_img = get_and_transform_img(fruit_train_path, need_resize=need_resize, size=size_to_resize)
veg_train_img = get_and_transform_img(veg_train_path, need_resize=need_resize, size=size_to_resize)
fruit_test_img = get_and_transform_img(fruit_test_path, need_resize=need_resize, size=size_to_resize)
veg_test_img = get_and_transform_img(veg_test_path, need_resize=need_resize, size=size_to_resize)
print(f'Загрузка данных {time.time() - start_time}')
start_time = time.time()
x_train, y_train = create_x_and_y(fruit_train_img, veg_train_img, mix=need_mix_train, random_seed=random_seed,
                                  classes_count=classes_count)
x_test, y_test = create_x_and_y(fruit_test_img, veg_test_img, mix=need_mix_test, random_seed=random_seed,
                                classes_count=classes_count)
print(f'Формирование Х и Y {time.time() - start_time}')

if need_show_data:
    print(f'Количество элементов в x_train - {len(x_train)}')
    print(f'Количество элементов в y_train - {len(y_train)}')
    print(f'Количество элементов в x_test - {len(x_test)}')
    print(f'Количество элементов в y_test - {len(y_test)}')
    print(f'Нулевой элемент в x_train размер {np.shape(x_train[0])}')
    print(f'Нулевой элемент в x_test размер {np.shape(x_test[0])}')
    print(f'Нулевой элемент в x_train {x_train[0]}')
    print(f'Нулевой элемент в y_train {y_train[0]}')
    print(f'Нулевой элемент в x_test {x_test[0]}')
    print(f'Нулевой элемент в y_test {y_test[0]}')
    fig = plt.figure(figsize=(16, 5))
    fig.add_subplot(1, 2, 1)
    img = x_train[0]
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'Нулевая картинка в тренировочной выборке')
    fig.add_subplot(1, 2, 2)
    img = x_test[0]
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'Нулевая картинка в тестовой выборке')
    fig.suptitle('Нулевые картинки в выборках', fontsize=14)
    plt.show()

epochs = 10
batch = 10
neurons_num = 1024
activate_funcs = ['relu', 'tanh', 'sigmoid', 'softmax', 'softplus', 'softsign', 'selu', 'elu', 'exponential',
                  'leaky_relu', 'relu6', 'silu', 'hard_silu', 'gelu', 'hard_sigmoid', 'linear', 'mish', 'log_softmax']
activate_func = activate_funcs[0]
optimizers = ['Nadam', 'SGD', 'RMSprop', 'Adam', 'Adadelta', 'Adagrad', 'Adamax', 'Ftrl']
optimizer = optimizers[3]
model = create_model(neurons_num=neurons_num, activate_func=activate_func)
result(model)