# RL Agent for Atari Games

A comprehensive implementation of Deep Q-Network (DQN) and its variants for playing Atari games using PyTorch and Gymnasium.

## Features

- **Standard DQN**: Basic Deep Q-Network implementation
- **Double DQN**: Reduces overestimation bias
- **Dueling DQN**: Separates value and advantage streams
- **Prioritized Experience Replay**: Samples important transitions more frequently
- **Atari Environment Wrapper**: Preprocessing pipeline for Atari games
- **Training & Evaluation**: Complete pipeline with logging and visualization

## Project Structure

```
RLAgent_AtariGames/
├── src/
│   ├── agents/
│   │   ├── dqn_agent.py        # Main DQN agent implementation
│   │   └── dqn_network.py      # Neural network architectures
│   ├── envs/
│   │   └── atari_wrapper.py    # Atari environment preprocessing
│   └── utils/
│       └── replay_buffer.py    # Experience replay buffers
├── models/                     # Saved model checkpoints
├── logs/                       # Training logs and tensorboard files
├── train.py                    # Training script
├── evaluate.py                 # Evaluation script
└── requirements.txt            # Dependencies
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd RLAgent_AtariGames
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install Atari ROMs (required for Gymnasium):
```bash
pip install "gymnasium[atari,accept-rom-license]"
```

## Usage

### Training

Train a DQN agent on Breakout with default settings:
```bash
python train.py --game breakout
```

Train with Double DQN and Dueling architecture:
```bash
python train.py --game breakout --double_dqn --dueling
```

Train with all enhancements:
```bash
python train.py --game breakout --double_dqn --dueling --prioritized
```

### Available Games

The wrapper supports these popular Atari games:
- `breakout` - ALE/Breakout-v5
- `pong` - ALE/Pong-v5
- `space_invaders` - ALE/SpaceInvaders-v5
- `ms_pacman` - ALE/MsPacman-v5
- `qbert` - ALE/Qbert-v5
- `seaquest` - ALE/Seaquest-v5
- `beam_rider` - ALE/BeamRider-v5
- `enduro` - ALE/Enduro-v5
- `alien` - ALE/Alien-v5
- `asterix` - ALE/Asterix-v5

### Training Parameters

Key training parameters you can adjust:

```bash
python train.py \
    --game breakout \
    --episodes 2000 \
    --learning_rate 1e-4 \
    --gamma 0.99 \
    --epsilon_start 1.0 \
    --epsilon_end 0.01 \
    --epsilon_decay 0.995 \
    --buffer_size 100000 \
    --batch_size 32 \
    --target_update_freq 1000 \
    --double_dqn \
    --dueling \
    --prioritized
```

### Evaluation

Evaluate a trained agent:
```bash
python evaluate.py --game breakout --model_path models/breakout_best_dqn.pth --num_episodes 10
```

Evaluate with rendering (watch the agent play):
```bash
python evaluate.py --game breakout --model_path models/breakout_best_dqn.pth --render --num_episodes 5
```

Save a video of gameplay:
```bash
python evaluate.py --game breakout --model_path models/breakout_best_dqn.pth --render --save_video
```

Human play mode (for comparison):
```bash
python evaluate.py --game breakout --human_play
```

## Algorithm Details

### DQN (Deep Q-Network)
- Uses experience replay to break correlation between consecutive samples
- Employs a separate target network for stable learning
- Updates target network every 1000 steps by default

### Double DQN
- Uses the main network to select actions and target network to evaluate them
- Reduces overestimation bias present in standard DQN

### Dueling DQN
- Separates the Q-function into value and advantage streams
- Better handles scenarios where action choice doesn't significantly affect value

### Prioritized Experience Replay
- Samples transitions based on their TD-error magnitude
- Focuses learning on more "surprising" or important experiences

## Environment Preprocessing

The Atari wrapper applies several preprocessing steps:
- **Frame skipping**: Repeats actions for 4 frames by default
- **Frame stacking**: Stacks 4 consecutive frames for temporal information
- **Grayscale conversion**: Converts RGB to grayscale
- **Frame resizing**: Resizes to 84x84 pixels
- **Reward clipping**: Clips rewards to {-1, 0, 1}
- **No-op starts**: Random no-op actions at episode start

## Monitoring Training

Training progress is logged to TensorBoard. View logs with:
```bash
tensorboard --logdir logs/
```

Key metrics tracked:
- Episode rewards
- 100-episode moving average
- Training loss
- Epsilon (exploration rate)

## Expected Results

Training times and expected scores vary by game:

| Game | Training Episodes | Expected Score | Human Score |
|------|------------------|----------------|-------------|
| Breakout | 1000-2000 | 300-400 | 30 |
| Pong | 500-1000 | 15-20 | 9 |
| Space Invaders | 1500-2500 | 1000-1500 | 1670 |

## Tips for Better Performance

1. **Hyperparameter tuning**: Adjust learning rate, epsilon decay, and network architecture
2. **Longer training**: Some games require 5000+ episodes for optimal performance
3. **Environment-specific modifications**: Different games may benefit from different preprocessing
4. **Hardware**: GPU training significantly reduces training time

## Troubleshooting

**Common issues:**

1. **ROM not found**: Install with `pip install "gymnasium[atari,accept-rom-license]"`
2. **CUDA errors**: Ensure PyTorch CUDA version matches your CUDA installation
3. **Memory issues**: Reduce buffer size or batch size
4. **Slow training**: Use GPU if available, reduce frame stack/skip if needed

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Gymnasium with Atari support
- OpenCV for image processing
- Matplotlib for plotting
- TensorBoard for logging

## License

This project is provided for educational purposes. Please ensure you comply with Atari ROM licenses.