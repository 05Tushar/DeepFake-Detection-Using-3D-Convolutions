import sys
import json
import subprocess
from pathlib import Path
from .utils import get_n_frames

if __name__ == '__main__':
    video_dir_path = Path(sys.argv[1])
    json_path = Path(sys.argv[2])
    if len(sys.argv) > 3:
        dst_json_path = Path(sys.argv[3])
    else:
        dst_json_path = json_path

    with json_path.open('r') as f:
        json_data = json.load(f)

    dst_data = {}
    labels = []
    labels.append('FAKE')
    labels.append('REAL')
    dst_data['labels'] = labels
    dst_data['database'] = {}
    it = 0
    total_len = 1335
    val_num = 1200
    for video_file_path in sorted(video_dir_path.iterdir()):
        name = video_file_path.name
        file_name = name+'.mp4'
        dst_data['database'][name] ={}
        if it<val_num:
            dst_data['database'][name]['subset'] = 'training'
        else:
            dst_data['database'][name]['subset'] = 'validation'

        it=it+1
        dst_data['database'][name]['annotations'] = {}
        dst_data['database'][name]['annotations']['label'] = json_data[file_name]['label']
        n_frames = get_n_frames(video_file_path)
        dst_data['database'][name]['annotations']['segment'] = (1, n_frames + 1)

    with dst_json_path.open('w') as f:
        json.dump(dst_data, f)