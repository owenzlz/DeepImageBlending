python run.py \
       --source_file data/1_source.png \
       --mask_file data/1_mask.png \
       --target_file data/1_target.png \
       --output_dir results/1 \
       --ss 300 --ts 512 --x 200 --y 235 \
       --gpu 0  --num_steps 1000 --save_video True

python run.py \
       --source_file data/3_source.png \
       --mask_file data/3_mask.png \
       --target_file data/3_target.png \
       --output_dir results/3 \
       --ss 384 --ts 512 --x 320 --y 256 \
       --gpu 0  --num_steps 1000 --save_video True

python run.py \
       --source_file data/5_source.png \
       --mask_file data/5_mask.png \
       --target_file data/5_target.png \
       --output_dir results/5 \
       --ss 320 --ts 512 --x 160 --y 256 \
       --gpu 0  --num_steps 1000 --save_video True





