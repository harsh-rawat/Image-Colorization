import cv2
import os
import pathlib
import shutil
import pickle
import collections


class video_utils:
    def __init__(self, path, size=256, image_type='jpg'):
        self.path = path
        self.size = size
        self.image_type = image_type

    def convert_to_grayscale_video(self, load_filename, save_filename):
        self.delete_temp_files('grayscale_gray')
        self.delete_temp_files('grayscale_color')

        self.extract_images('grayscale', load_filename)
        self.combine_images('grayscale', load_filename, save_filename)
        self.delete_temp_files('grayscale_gray')

    def combine_images(self, folder_name, load_filename, save_filename):
        video_name = '{}/{}'.format(self.path, save_filename)
        images_path = '{}/{}_gray'.format(self.path, folder_name)
        images = []
        image_dict = {}
        for img in os.listdir(images_path):
            if img.endswith(self.image_type):
                key = int(img.split('.')[0])
                image_dict[key] = img

        if len(image_dict) == 0:
            return False;
        imgs = collections.OrderedDict(sorted(image_dict.items()))
        for k, v in imgs.items():
            images.append(v)

        with open('{}/{}_fps'.format(images_path, load_filename), 'rb') as file:
            save_dict = pickle.load(file)
        fps = save_dict['fps']
        fourcc = save_dict['fourcc']

        frame = cv2.imread(os.path.join(images_path, images[0]))
        height, width, layers = frame.shape

        video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

        for image in images:
            video.write(cv2.imread(os.path.join(images_path, image)))

        print('Images have been combined!')

        cv2.destroyAllWindows()
        video.release()

        return True

    def delete_temp_files(self, folder_name):
        path = '{}/{}'.format(self.path, folder_name)
        folder = pathlib.Path(path)
        if folder.exists():
            shutil.rmtree(path)

    def extract_images(self, folder_name, filename):
        video_path = '{}/{}'.format(self.path, folder_name)
        video_path_gray = '{}_gray'.format(video_path)
        video_path_color = '{}_color'.format(video_path)

        try:
            if not os.path.exists(video_path_gray):
                os.makedirs(video_path_gray)
            if not os.path.exists(video_path_color):
                os.makedirs(video_path_color)
        except OSError:
            print('Error: Creating directory of data')

        cam = cv2.VideoCapture('{}/{}'.format(self.path, filename));
        frame_per_second = int(cam.get(cv2.CAP_PROP_FPS))
        save_dict = {'fps': frame_per_second, 'fourcc': int(cam.get(cv2.CAP_PROP_FOURCC))}
        with open('{}/{}_fps'.format(video_path_gray, filename), 'wb') as file:
            pickle.dump(save_dict, file)

        currentframe = 0

        while True:
            ret, frame = cam.read()
            if ret:
                name_gray = '{}/{}.{}'.format(video_path_gray, currentframe, 'jpg')
                name_color = '{}/{}.{}'.format(video_path_color, currentframe, 'jpg')
                new_frame = cv2.resize(frame, (self.size, self.size))
                gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(name_gray, gray)
                cv2.imwrite(name_color, new_frame)
                currentframe += 1
            else:
                break

        print('Grayscale images have been saved!')

        cam.release()
        cv2.destroyAllWindows()

        return True
