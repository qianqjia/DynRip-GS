import os
import glob


# # DTU_r2
# dataset_folder = r'/home/tuf/datasets/DTU/DTU'
# folders = [f for f in glob.glob(os.path.join(dataset_folder, '*')) if '.' not in os.path.split(f)[-1]]

# train_base_cmd = 'python train.py -s {} -m {} -r 2 --eval -i images_no_mask --iteration 30_000'
# render_base_cmd = 'python render.py -m {} --eval -r 2'
# metric_base_cmd = 'python metrics.py -m {}'

# for folder in folders:
#     output_folder = os.path.join('output/DTU_r2', os.path.split(folder)[-1])
#     # train_cmd = train_base_cmd.format(folder, output_folder)
#     # print(train_cmd)
#     # os.system(train_cmd)

#     # render_cmd = render_base_cmd.format(output_folder)
#     # print(render_cmd)
#     # os.system(render_cmd)

#     metric_cmd = metric_base_cmd.format(output_folder)
#     print(metric_cmd)
#     os.system(metric_cmd)

#     exit()

# BVI-Coras
dataset_folder = r'/home/tuf/datasets/WaterDataset/BVI-Coras'
folders = [f for f in glob.glob(os.path.join(dataset_folder, '*')) if '.' not in os.path.split(f)[-1]]

train_base_cmd = 'python train.py -s {} -m {} -r 2 --eval --iteration 30_000'
render_base_cmd = 'python render.py -m {} --eval -r 2'
metric_base_cmd = 'python metrics.py -m {}'

for folder in folders:
    output_folder = os.path.join('output/BVI-Coras', os.path.split(folder)[-1])
    train_cmd = train_base_cmd.format(folder, output_folder)
    print(train_cmd)
    os.system(train_cmd)

    render_cmd = render_base_cmd.format(output_folder)
    print(render_cmd)
    os.system(render_cmd)

    metric_cmd = metric_base_cmd.format(output_folder)
    print(metric_cmd)
    os.system(metric_cmd)


# IW
dataset_folder = r'/home/tuf/datasets/WaterDataset/IW'
folders = [f for f in glob.glob(os.path.join(dataset_folder, '*')) if '.' not in os.path.split(f)[-1]]

train_base_cmd = 'python train.py -s {} -m {} -r 2 --eval --iteration 30_000'
render_base_cmd = 'python render.py -m {} --eval -r 2'
metric_base_cmd = 'python metrics.py -m {}'

for folder in folders:
    output_folder = os.path.join('output/IW', os.path.split(folder)[-1])
    train_cmd = train_base_cmd.format(folder, output_folder)
    print(train_cmd)
    os.system(train_cmd)

    render_cmd = render_base_cmd.format(output_folder)
    print(render_cmd)
    os.system(render_cmd)

    metric_cmd = metric_base_cmd.format(output_folder)
    print(metric_cmd)
    os.system(metric_cmd)


# IW
dataset_folder = r'/home/tuf/datasets/WaterDataset/Seathru-metashape'
folders = [f for f in glob.glob(os.path.join(dataset_folder, '*')) if '.' not in os.path.split(f)[-1]]

train_base_cmd = 'python train.py -s {} -m {} -r 2 --eval --iteration 30_000'
render_base_cmd = 'python render.py -m {} --eval -r 2'
metric_base_cmd = 'python metrics.py -m {}'

for folder in folders:
    output_folder = os.path.join('output/Seathru-metashape', os.path.split(folder)[-1])
    train_cmd = train_base_cmd.format(folder, output_folder)
    print(train_cmd)
    os.system(train_cmd)

    render_cmd = render_base_cmd.format(output_folder)
    print(render_cmd)
    os.system(render_cmd)

    metric_cmd = metric_base_cmd.format(output_folder)
    print(metric_cmd)
    os.system(metric_cmd)

