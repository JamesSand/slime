
# alias slime="docker exec -it slime_zhizhou bash"
# alias killslime="docker rm -f slime_zhizhou"

# setup docker
ssd=ssd1 docker run --privileged --gpus all --ipc=host --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /$ssd/zhizhou/workspace/rotation-project/slime:/root/slime \
  -v /$ssd/zhizhou/workspace/rotation-project/shared_folder:/root/shared_folder \
  -v /$ssd/zhizhou/tmp:/tmp \
  -v ~/.tmux.conf:/root/.tmux.conf \
  --name slime_zhizhou \
  -itd jamessand42/slime:szz-rl /bin/bash

# docker commit






