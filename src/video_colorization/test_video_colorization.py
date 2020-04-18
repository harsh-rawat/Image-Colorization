from src.video_colorization.video_utils import *

path = 'C:/Users/harsh/Downloads/Assignments/Spring 2020/CS 766 Computer Vision/Project/Data/Animated/Videos'
filename = 'short.mp4'
save_ = 'short_gray.mp4';

ob = video_utils(path)
ob.convert_to_grayscale_video(filename, save_)