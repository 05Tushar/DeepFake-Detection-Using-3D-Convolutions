import subprocess
import argparse
from pathlib import Path

import cv2
from joblib import Parallel, delayed

def constrain(C,D):
    l = int(C - (D/2))
    if l<0:
        return 0
    else:
        return l


def video_process(video_file_path, dst_root_path, ext, fps=-1):
    if ext != video_file_path.suffix:
        return

    name = video_file_path.stem
    dst_dir_path = dst_root_path / name
    dst_dir_path.mkdir(exist_ok=True)

    pt = str(video_file_path)
    vidObj = cv2.VideoCapture(pt)
    count = 1
    success = 1
    W = 120
    H = 120
    while success:
        success, image = vidObj.read()
        if success:
            # Down sample
            scale_percent = 50
            width = int(image.shape[1] * scale_percent / 100)
            height = int(image.shape[0] * scale_percent / 100)
            dsize = (width, height)
            image = cv2.resize(image, dsize)

            # Extract faces
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=3,
                minSize=(30, 30)
            )

            for (x, y, w, h) in faces:
                c_x = x + w / 2
                c_y = y + h / 2
                x = constrain(c_x, W)
                y = constrain(c_y, H)
                w = W
                h = H
                roi_color = image[y:y + h, x:x + w]
                print('Creating Image {}/image_%05d.jpg'.format(dst_dir_path) % count)
                cv2.imwrite('{}/image_%05d.jpg'.format(dst_dir_path) % count, roi_color)

        count += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'dir_path', default=None, type=Path, help='Directory path of videos')
    parser.add_argument(
        'dst_path',
        default=None,
        type=Path,
        help='Directory path of jpg videos')
    parser.add_argument(
        'dataset',
        default='',
        type=str,
        help='Dataset name (kinetics | mit | ucf101 | hmdb51 | activitynet | faceforensics)')
    parser.add_argument(
        '--n_jobs', default=-1, type=int, help='Number of parallel jobs')
    parser.add_argument(
        '--fps',
        default=-1,
        type=int,
        help=('Frame rates of output videos. '
              '-1 means original frame rates.'))
    #parser.add_argument(
    #    '--size', default=240, type=int, help='Frame size of output videos.')
    args = parser.parse_args()

    if args.dataset in ['kinetics', 'mit', 'activitynet', 'faceforensics']:
        ext = '.mp4'
    else:
        ext = '.avi'

    if (args.dataset=='faceforensics'):
        video_file_paths = [x for x in sorted(args.dir_path.iterdir())]
        status_list = Parallel(
            n_jobs=args.n_jobs,
            backend='threading')(delayed(video_process)(
                video_file_path, args.dst_path, ext, args.fps)
                                 for video_file_path in video_file_paths)
