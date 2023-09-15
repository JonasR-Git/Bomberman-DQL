# bomberman_rl
Setup for a project/competition amongst students to train a winning Reinforcement Learning agent for the classic game Bomberman.

tensorboard --logdir=./logs

python main.py play --agents deepql --train 1 --n-rounds 10000 --no-gui --scenario coin-heaven