

local_save_dir=/ssd2/zhizhou/workspace/rotation-project/shared_folder/math_datasets

python train_math.py --local_save_dir $local_save_dir/train_math
python test_math500.py --local_dir $local_save_dir/test_math500
python test_gsm8k.py --local_dir $local_save_dir/test_gsm8k


