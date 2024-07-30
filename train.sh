nohup python -u train.py --gpu_list=0 --max_ratio=0.1 > r0.1g0.txt &
nohup python -u train.py --gpu_list=1 --max_ratio=0.3 > r0.3g1.txt &
nohup python -u train.py --gpu_list=2 --max_ratio=0.5 > r0.5g2.txt &