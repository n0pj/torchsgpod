import matplotlib.pyplot as plt
import numpy as np
import time
import math
from functools import reduce
import torch, os, glob
import torch.nn as nn
from termcolor import cprint
from tqdm import tqdm

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    try:
        plt.imshow(np.transpose(npimg, (2, 3, 1)))
        plt.show()
    except KeyboardInterrupt:
        quit()

def progress(start, end, loss, acc, vloss=0, vacc=0, e=0, epoch=0, epochs=0, length=25, fill='#'):
    # percent = str("{:.1f}".format(100 * (start / float(end))))
    # percent = percent.rjust(5, " ")
    filledLength = int(length * start // end)
    bar = fill * filledLength + '-' * (length - filledLength)
    fillzero = str(start)
    end_str = len(str(end))
    prog = "{}/{}".format(fillzero.rjust(end_str, " "), str(end).rjust(6, " "))
    prosecced = e / (start + 1e-10)
    estimate = (end - start) * prosecced
    if start == 0:
        print("\nEpoch: {}/{}".format(epoch+1, epochs))
    if loss != 0:
        if start != end:
            print('\r{} [{}]  ETA: {}s - loss: {:.4f} - acc: {:.4f}  '.format(prog, bar, int(estimate), loss, acc), end = '\r')
        else:
            print('\r{} [{}] Time: {}s - loss: {:.4f} - acc: {:.4f} - vloss: {:.4f} - vacc: {:.4f}'.format(prog, bar, int(e), loss, acc, vloss, vacc), end = '\n')
    else:
        pass
        # if start != end:
        #     print('\r{} [{}]  ETA: {}s - loss: {:.4f} - acc: {:.4f}  '.format(prog, bar, int(estimate), loss, acc), end = '\r')
        # else:
        #     print('\r{} [{}] Time: {}s - vloss: {:.4f} - vacc: {:.4f}'.format(prog, bar, int(e), vloss, vacc), end = '\r')

        # print('\r{} [{}] {}% {} - loss: {:.3f} time: {:.3f}s'.format(prog, bar, percent, suffix, loss, estimate), end = '\r')

    # Print New Line on Complete
    if start == end:
        pass
        # print()

def compose(*funcs):
    print(funcs)
    return reduce(lambda f, g: lambda *args, **kwargs: g(f(*args, **kwargs)), funcs)

def print_info(info, _type=None):
    if _type is not None:
        if isinstance(info,str):
            cprint(info, _type[0], attrs=[_type[1]])
        elif isinstance(info,list):
            for i in range(info):
                cprint(i, _type[0], attrs=[_type[1]])
    else:
        print(info)

def exchange_data(h, w):
    base_dir = "rawdata/"
    labels = os.listdir(base_dir)
    labels_count = len(labels)
    
    split_p = 0.2

    x_train, x_test, y_train, y_test = [], [], [], []

    for i, label in enumerate(labels):
        image_dir = base_dir + label
        files = glob.glob(image_dir + "/*jpg")
        print(files)
        file_count = len(files)
        template = "\nLabel {}, File count {}\n"
        print(template.format(label, file_count))

        for idx, image in tqdm(enumerate(files)):
            try:
                img = Image.open(image)
            except OSError:
                print("broken image")
            else:
                img = img.convert("RGB")
                img = img.resize((h, w))
                np_img = np.asarray(img)

                testset = file_count * split_p

                if idx <= testset:
                    x_train.append(np_img)
                    y_train.append(i)
                    
                    for angle in range(-40, 40, 20):
                        rotate_img = img.rotate(angle)
                        np_img = np.asarray(rotate_img)
                        x_train.append(np_img)
                        y_train.append(i)
                else:
                    
                    for angle2 in range(-40, 40, 20):
                        rotate_img = img.rotate(angle2)
                        np_img = np.asarray(rotate_img)
                        x_test.append(np_img)
                        y_test.append(i)

    x_train = np.asarray(x_train)
    x_test = np.asarray(x_test)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    dataset = (x_train, y_train, x_test, y_test)
    npy = "npy/"
    os.makedirs(npy, exist_ok=True)
    np.save(npy+"data.npy", dataset)
    print("\nDatasets created done.")

# 40000/40000  - 24s 600us/sample - loss: 2.1855 - accuracy: 0.2448 - val_loss: 1.8212 - val_accuracy: 0.3034
# Learning rate:  0.001
# Epoch 2/40
# 4620/40000  - ETA: 18s - loss: 1.8521 - accuracy: 0.2745^CTraceback (most recent call


# def progress(start, end, loss=0, prefix='Progress', suffix='Complete', length=25, fill='#'):
#     """
#     Call in a loop to create terminal progress bar
#     @params:
#         start   - Required  : current start (Int)
#         end       - Required  : end iterations (Int)
#         prefix      - Optional  : prefix string (Str)
#         suffix      - Optional  : suffix string (Str)
#         decimals    - Optional  : positive number of decimals in percent complete (Int)
#         length      - Optional  : character length of bar (Int)
#         fill        - Optional  : bar fill character (Str)
#     """
#     percent = ("{0:.1f}").format(100 * (start / float(end)))
#     filledLength = int(length * start // end)
#     bar = fill * filledLength + '-' * (length - filledLength)
#     fillzero = str(start)
#     prog = "{}/{}".format(fillzero.rjust(len(str(end)), " "), end)
#     # print('\r%s |%s| %s%% %s %3f' % (prefix, bar, percent, suffix, loss), end = '\r')
#     if loss != 0:
#         print('\r{} [{}] {}% {} - loss: {:.3f}'.format(prog, bar, percent, suffix, loss), end = '\r')
#     else:
#         print('\r{} [{}] {}% {}'.format(prefix, bar, percent, suffix), end = '\r')


#     # Print New Line on Complete
#     if start == end: 
#         print()