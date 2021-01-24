
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

    

def generate_gif(frame_number, frames_for_gif, reward, path):
    """
        Args:
            frame_number: Integer, determining the number of the current frame
            frames_for_gif: A sequence of (210, 160, 3) frames of an Atari game in RGB
            reward: Integer, Total reward of the episode that es ouputted as a gif
            path: String, path where gif is saved
    
    for idx, frame_idx in enumerate(frames_for_gif): 
        frames_for_gif[idx] = resize(frame_idx, (420, 320, 3), 
                                     preserve_range=True, order=0).astype(np.uint8)
        
    imageio.mimsave(f'{path}{"ATARI_frame_{0}_reward_{1}.gif".format(frame_number, reward)}', frames_for_gif, duration=1/30)
    """
    pass
