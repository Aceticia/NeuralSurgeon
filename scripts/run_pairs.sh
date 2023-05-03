# Run the resnet18 models
python src/eval_all_pairs.py -m store_path=outs/resnet18_64_0 model/net=resnet18 ckpt_path=checkpoints/src.models.components.resnet.ResNet18Model_64_0.ckpt
python src/eval_all_pairs.py -m store_path=outs/resnet18_64_1 model/net=resnet18 ckpt_path=checkpoints/src.models.components.resnet.ResNet18Model_64_1.ckpt
python src/eval_all_pairs.py -m store_path=outs/resnet18_64_2 model/net=resnet18 ckpt_path=checkpoints/src.models.components.resnet.ResNet18Model_64_2.ckpt
python src/eval_all_pairs.py -m store_path=outs/resnet18_64_3 model/net=resnet18 ckpt_path=checkpoints/src.models.components.resnet.ResNet18Model_64_3.ckpt
python src/eval_all_pairs.py -m store_path=outs/resnet18_64_4 model/net=resnet18 ckpt_path=checkpoints/src.models.components.resnet.ResNet18Model_64_4.ckpt

python src/eval_all_pairs.py -m store_path=outs/resnet18_128_0 model.net.subspace_size=128 model/net=resnet18 ckpt_path=checkpoints/src.models.components.resnet.ResNet18Model_128_0.ckpt
python src/eval_all_pairs.py -m store_path=outs/resnet18_128_1 model.net.subspace_size=128 model/net=resnet18 ckpt_path=checkpoints/src.models.components.resnet.ResNet18Model_128_1.ckpt
python src/eval_all_pairs.py -m store_path=outs/resnet18_128_2 model.net.subspace_size=128 model/net=resnet18 ckpt_path=checkpoints/src.models.components.resnet.ResNet18Model_128_2.ckpt
python src/eval_all_pairs.py -m store_path=outs/resnet18_128_3 model.net.subspace_size=128 model/net=resnet18 ckpt_path=checkpoints/src.models.components.resnet.ResNet18Model_128_3.ckpt
python src/eval_all_pairs.py -m store_path=outs/resnet18_128_4 model.net.subspace_size=128 model/net=resnet18 ckpt_path=checkpoints/src.models.components.resnet.ResNet18Model_128_4.ckpt

# Then resnet34
python src/eval_all_pairs.py -m store_path=outs/resnet34_64_0 model/net=resnet34 ckpt_path=checkpoints/src.models.components.resnet.ResNet34Model_64_0.ckpt
python src/eval_all_pairs.py -m store_path=outs/resnet34_64_1 model/net=resnet34 ckpt_path=checkpoints/src.models.components.resnet.ResNet34Model_64_1.ckpt
python src/eval_all_pairs.py -m store_path=outs/resnet34_64_2 model/net=resnet34 ckpt_path=checkpoints/src.models.components.resnet.ResNet34Model_64_2.ckpt
python src/eval_all_pairs.py -m store_path=outs/resnet34_64_3 model/net=resnet34 ckpt_path=checkpoints/src.models.components.resnet.ResNet34Model_64_3.ckpt
python src/eval_all_pairs.py -m store_path=outs/resnet34_64_4 model/net=resnet34 ckpt_path=checkpoints/src.models.components.resnet.ResNet34Model_64_4.ckpt

python src/eval_all_pairs.py -m store_path=outs/resnet34_128_0 model.net.subspace_size=128 model/net=resnet34 ckpt_path=checkpoints/src.models.components.resnet.ResNet34Model_128_0.ckpt
python src/eval_all_pairs.py -m store_path=outs/resnet34_128_1 model.net.subspace_size=128 model/net=resnet34 ckpt_path=checkpoints/src.models.components.resnet.ResNet34Model_128_1.ckpt
python src/eval_all_pairs.py -m store_path=outs/resnet34_128_2 model.net.subspace_size=128 model/net=resnet34 ckpt_path=checkpoints/src.models.components.resnet.ResNet34Model_128_2.ckpt
python src/eval_all_pairs.py -m store_path=outs/resnet34_128_3 model.net.subspace_size=128 model/net=resnet34 ckpt_path=checkpoints/src.models.components.resnet.ResNet34Model_128_3.ckpt
python src/eval_all_pairs.py -m store_path=outs/resnet34_128_4 model.net.subspace_size=128 model/net=resnet34 ckpt_path=checkpoints/src.models.components.resnet.ResNet34Model_128_4.ckpt