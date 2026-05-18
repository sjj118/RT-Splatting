python train.py -s data/rt-splatting/van -r 4 --eval --env_scope_center -0.5 -1 0.5 --env_scope_radius 2.5 --init_until_iter 700 --norm_loss_from_iter 700 --xyz_axis 2.0 1.0 0.0
python train.py -s data/rt-splatting/swab -r 4 --eval --env_scope_center 0 2 2.5 --env_scope_radius 2 --init_until_iter 6000 --norm_loss_from_iter 5000 --xyz_axis 2.0 1.0 0.0
python train.py -s data/nerf-casting/compact -r 4 --eval --env_scope_center -0.5 0 -0.5 --env_scope_radius 1.9 --init_until_iter 1000 --norm_loss_from_iter 700 --xyz_axis 2.0 1.0 0.0
python train.py -s data/nerf-casting/hatchback -r 4 --eval --env_scope_center 0 0 0 --env_scope_radius 2.3 --mask_loss_from_iter 2000 --init_until_iter 1000 --norm_loss_from_iter 0 --xyz_axis 2.0 1.0 0.0
python train.py -s data/tandt/truck --eval --env_scope_center -0.943 -0.083 0.514 --env_scope_radius 1 --init_until_iter 700 --norm_loss_from_iter 700 --xyz_axis 2.0 1.0 0.0
python train.py -s data/envgs/audi -r 4 --eval --env_scope_center 3.383 2.323 3.527 --env_scope_radius 4.0 --init_until_iter 1500 --norm_loss_from_iter 0 --xyz_axis 2.0 1.0 0.0
python train.py -s data/ref_real/sedan -r 8 --eval --iterations 31000 --env_scope_center -0.032 0.808 0.751 --env_scope_radius 2.138 --init_until_iter 700 --norm_loss_from_iter 700 --xyz_axis 2.0 1.0 0.0
python train.py -s data/ref_real/toycar -r 4 --eval --iterations 31000 --env_scope_center 0.486 1.108 3.72 --env_scope_radius 2.507 --init_until_iter 1500 --norm_loss_from_iter 1500 --xyz_axis 0.0 2.0 1.0

python render.py -m output/van
python render.py -m output/swab
python render.py -m output/compact
python render.py -m output/hatchback
python render.py -m output/truck
python render.py -m output/audi
python render.py -m output/sedan
python render.py -m output/toycar

python metrics.py -m output/van
python metrics.py -m output/swab
python metrics.py -m output/compact
python metrics.py -m output/hatchback
python metrics.py -m output/truck
python metrics.py -m output/audi
python metrics.py -m output/sedan
python metrics.py -m output/toycar