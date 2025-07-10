import os
import glob



train_base_cmd = 'python train_w_dinov2_time_contrib.py -s {} -m {} -r 2 --eval --iteration 15_000'
render_base_cmd = 'python render_uw_time_contrib.py -m {} --eval -r 2'
metric_base_cmd = 'python metrics_qq.py -m {}'
video_base_cmd = 'python generate_video_time_contrib.py -m {} --eval --out_path {}'


# BVI-Coras
dataset_folder = r'/home/tuf/datasets/WaterDataset/BVI-Coras'
folders = [f for f in glob.glob(os.path.join(dataset_folder, '*')) if '.' not in os.path.split(f)[-1]]
for folder in folders:
    if '11417' not in folder:
        continue
    output_folder = os.path.join('output_w_dinov2_time_contrib/BVI-Coras', os.path.split(folder)[-1] + '_appdim0')
    # train_cmd = train_base_cmd.format(folder, output_folder) + ' --appearance_dim 0'
    # print(train_cmd)
    # os.system(train_cmd)

    # # exit()

    # render_cmd = render_base_cmd.format(output_folder)
    # print(render_cmd)
    # os.system(render_cmd)

    # metric_cmd = metric_base_cmd.format(output_folder)
    # print(metric_cmd)
    # os.system(metric_cmd)

    video_cmd = video_base_cmd.format(output_folder, output_folder)
    print(video_cmd)
    os.system(video_cmd)

    # exit()



