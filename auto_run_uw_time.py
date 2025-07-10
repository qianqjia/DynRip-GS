import os
import glob

train_base_cmd = 'python train_uw_time.py -s {} -m {} -r 2 --eval --iteration 15_000 --port 1210'
render_base_cmd = 'python render_uw_time.py -m {} --eval -r 2 --port 1210'
metric_base_cmd = 'python metrics_qq.py -m {}'
video_base_cmd = 'python generate_video_time.py -m {} --eval --port 1210'


# BVI-Coras
dataset_folder = r'/public/home/qianqianjia/dataset/BVI-Coras'
folders = [f for f in glob.glob(os.path.join(dataset_folder, '*')) if '.' not in os.path.split(f)[-1]]

for folder in folders:
    # if '11417' not in folder:
    #     continue

    output_folder = os.path.join('/public/home/qianqianjia/experiments/DynRip-GS/BVI-Coras/time', os.path.split(folder)[-1] + '_appdim0')
    train_cmd = train_base_cmd.format(folder, output_folder) + ' --appearance_dim 0'
    print(train_cmd)
    os.system(train_cmd)

    render_cmd = render_base_cmd.format(output_folder)
    print(render_cmd)
    os.system(render_cmd)

    metric_cmd = metric_base_cmd.format(output_folder)
    print(metric_cmd)
    os.system(metric_cmd)

    video_cmd = video_base_cmd.format(output_folder)
    print(video_cmd)
    os.system(video_cmd)
    # exit()

# IW
dataset_folder = r'/public/home/qianqianjia/dataset/IW'
folders = [f for f in glob.glob(os.path.join(dataset_folder, '*')) if '.' not in os.path.split(f)[-1]]
for folder in folders:
    output_folder = os.path.join('/public/home/qianqianjia/experiments/DynRip-GS/IW/time', os.path.split(folder)[-1] + '_appdim0')
    train_cmd = train_base_cmd.format(folder, output_folder) + ' --appearance_dim 0'
    print(train_cmd)
    os.system(train_cmd)

    render_cmd = render_base_cmd.format(output_folder)
    print(render_cmd)
    os.system(render_cmd)

    metric_cmd = metric_base_cmd.format(output_folder)
    print(metric_cmd)
    os.system(metric_cmd)

    video_cmd = video_base_cmd.format(output_folder)
    print(video_cmd)
    os.system(video_cmd)


# Seathru-metashape
dataset_folder = r'/public/home/qianqianjia/dataset/Seathru-metashape'
folders = [f for f in glob.glob(os.path.join(dataset_folder, '*')) if '.' not in os.path.split(f)[-1]]
for folder in folders:
    output_folder = os.path.join('/public/home/qianqianjia/experiments/DynRip-GS/Seathru-metashape/time', os.path.split(folder)[-1] + '_appdim0')
    train_cmd = train_base_cmd.format(folder, output_folder) + ' --appearance_dim 0'
    print(train_cmd)
    os.system(train_cmd)

    render_cmd = render_base_cmd.format(output_folder)
    print(render_cmd)
    os.system(render_cmd)

    metric_cmd = metric_base_cmd.format(output_folder)
    print(metric_cmd)
    os.system(metric_cmd)

    video_cmd = video_base_cmd.format(output_folder)
    print(video_cmd)
    os.system(video_cmd)