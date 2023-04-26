# Run the resnet18 models
python src/eval_conditioning.py -m store_path=outs/resnet18_0 model/net=resnet18 ckpt_path=logs/train/multiruns/2023-04-24_19-03-20/0/checkpoints/epoch_020.ckpt alpha=0.5
python src/eval_conditioning.py -m store_path=outs/resnet18_1 model/net=resnet18 ckpt_path=logs/train/multiruns/2023-04-24_19-03-20/1/checkpoints/epoch_035.ckpt alpha=0.5
python src/eval_conditioning.py -m store_path=outs/resnet18_2 model/net=resnet18 ckpt_path=logs/train/multiruns/2023-04-24_19-03-20/2/checkpoints/epoch_040.ckpt alpha=0.5
python src/eval_conditioning.py -m store_path=outs/resnet18_3 model/net=resnet18 ckpt_path=logs/train/multiruns/2023-04-24_19-03-20/3/checkpoints/epoch_020.ckpt alpha=0.5
python src/eval_conditioning.py -m store_path=outs/resnet18_4 model/net=resnet18 ckpt_path=logs/train/multiruns/2023-04-24_19-03-20/4/checkpoints/epoch_022.ckpt alpha=0.5

# Then resnet34
python src/eval_conditioning.py -m store_path=outs/resnet34_0 model/net=resnet34 ckpt_path=logs/train/multiruns/2023-04-24_19-03-20/0/checkpoints/epoch_041.ckpt alpha=0.5
python src/eval_conditioning.py -m store_path=outs/resnet34_1 model/net=resnet34 ckpt_path=logs/train/multiruns/2023-04-24_19-03-20/1/checkpoints/epoch_032.ckpt alpha=0.5
python src/eval_conditioning.py -m store_path=outs/resnet34_2 model/net=resnet34 ckpt_path=logs/train/multiruns/2023-04-24_19-03-20/2/checkpoints/epoch_033.ckpt alpha=0.5
python src/eval_conditioning.py -m store_path=outs/resnet34_3 model/net=resnet34 ckpt_path=logs/train/multiruns/2023-04-24_19-03-20/3/checkpoints/epoch_046.ckpt alpha=0.5
python src/eval_conditioning.py -m store_path=outs/resnet34_4 model/net=resnet34 ckpt_path=logs/train/multiruns/2023-04-24_19-03-20/4/checkpoints/epoch_048.ckpt alpha=0.5