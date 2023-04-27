python src/train.py -m seed=0,1,2,3,4 hydra/launcher=submitit_slurm \
  model/net=resnet18,resnet34 model.net.subspace_size=64,128 \
  hydra.launcher.partition=oermannlab hydra.launcher.gres="gpu:a100:1" +hydra.launcher.additional_parameters.time="3-00:00:00" \
  hydra.launcher.cpus_per_task=8 hydra.launcher.mem_per_cpu=10G \
 +hydra.launcher.additional_parameters.qos=qos_free