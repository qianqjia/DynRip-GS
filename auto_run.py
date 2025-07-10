import os
import glob

# --lambda_normal 0.01 --lambda_dssim 0.3
train_base_cmd = 'python train.py -s {} -m {}  --eval  -i images_no_mask --iteration 30_000 -r 2'
render_base_cmd = 'python render.py -s {} -m {}  --iteration 30_000 --eval -r 2'
metric_base_cmd = 'python metrics.py -m {}'
# video_base_cmd = 'python generate_video.py -s {} -m {}  --eval --out_path {}'

paths = glob.glob('/home/tuf/datasets/DTU/DTU/*')
out_folder = 'output/DTU'

for path in paths:
    if os.path.isdir(path):
        folder_name = os.path.basename(path)
        out_path = os.path.join(out_folder, folder_name)
        train_cmd = train_base_cmd.format(path, out_path)
        render_cmd = render_base_cmd.format(path, out_path)
        metric_cmd = metric_base_cmd.format(out_path)

        print("Training command: ", train_cmd)
        print("Rendering command: ", render_cmd)
        print("Metric command: ", metric_cmd)

        # Execute the commands
        os.system(train_cmd)
        os.system(render_cmd)
        os.system(metric_cmd)


# # seathru
# dataset_folder = r'/home/tuf/datasets/water_dump_lab'
# output_folder = r'output_uw/water_dump_lab'
# folders = glob.glob(os.path.join(dataset_folder, '*'))
# for folder in folders:
#     if '.' in folder:
#         continue

#     res_cmd = ' -r 4'
#     name = os.path.split(folder)[-1]

#     train_cmd = train_base_cmd.format(
#         folder, os.path.join(output_folder, name)
#     ) + res_cmd
#     print(train_cmd)
#     os.system(train_cmd)

#     render_cmd = render_base_cmd.format(
#         folder, os.path.join(output_folder, name)
#     ) + res_cmd
#     print(render_cmd)
#     os.system(render_cmd)

#     metric_cmd = metric_base_cmd.format(
#         os.path.join(output_folder, name)
#     )
#     print(metric_cmd)
#     os.system(metric_cmd)

#     video_cmd = video_base_cmd.format(
#         folder, os.path.join(output_folder, name), os.path.join(output_folder, name)
#     ) + res_cmd
#     print(video_cmd)
#     os.system(video_cmd)


# # seathru
# dataset_folder = r'/home/tuf/datasets/seathru'
# output_folder = r'output_uw/seathru'
# folders = glob.glob(os.path.join(dataset_folder, '*'))
# for folder in folders:
#     if '.' in folder:
#         continue

#     name = os.path.split(folder)[-1]
#     res_cmd = ' -r 1'
#     train_cmd = train_base_cmd.format(
#         folder, os.path.join(output_folder, name)
#     ) + res_cmd
#     print(train_cmd)
#     os.system(train_cmd)

#     render_cmd = render_base_cmd.format(
#         folder, os.path.join(output_folder, name)
#     ) + res_cmd
#     print(render_cmd)
#     os.system(render_cmd)

#     metric_cmd = metric_base_cmd.format(
#         os.path.join(output_folder, name)
#     )
#     print(metric_cmd)
#     os.system(metric_cmd)

#     video_cmd = video_base_cmd.format(
#         folder, os.path.join(output_folder, name), os.path.join(output_folder, name)
#     ) + res_cmd
#     print(video_cmd)
#     os.system(video_cmd)


# underwater_dataset_uwnerf 
# dataset_folder = r'/home/tuf/datasets/BVI-Coras'
# output_folder = r'output_uw/BVI-Coras'
# folders = glob.glob(os.path.join(dataset_folder, '*'))
# for folder in folders:
#     if '.' in folder:
#         continue

#     res_cmd = ' -r 2'
#     name = os.path.split(folder)[-1]

#     train_cmd = train_base_cmd.format(
#         folder, os.path.join(output_folder, name)
#     ) + res_cmd
#     print(train_cmd)
#     os.system(train_cmd)

#     render_cmd = render_base_cmd.format(
#         folder, os.path.join(output_folder, name)
#     ) + res_cmd
#     print(render_cmd)
#     os.system(render_cmd)

#     metric_cmd = metric_base_cmd.format(
#         os.path.join(output_folder, name)
#     )
#     print(metric_cmd)
#     os.system(metric_cmd)

#     video_cmd = video_base_cmd.format(
#         folder, os.path.join(output_folder, name), os.path.join(output_folder, name)
#     ) + res_cmd
#     print(video_cmd)
#     os.system(video_cmd)

