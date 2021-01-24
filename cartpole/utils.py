
import psutil
import os 
import matplotlib.pyplot as plt

def show_RAM_usage():
    py = psutil.Process(os.getpid())
    print('RAM usage: {} GB {} %'.format(round(py.memory_info()[0]/2. ** 30, 2), dict(psutil.virtual_memory()._asdict())['percent']))

def get_RAM_usage():
    return dict(psutil.virtual_memory()._asdict())['percent']

def show_image(image):
    # imgplot = plt.imshow(np.rollaxis(image, 0, 3))
    plt.imshow(image*255, cmap = plt.get_cmap('gray'))
    plt.show()

