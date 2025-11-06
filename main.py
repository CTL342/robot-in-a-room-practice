import os
import random
from collections import deque

import torch
from torch import nn

# Enviornment
room = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]

maze = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 1, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 1, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]

# Variables
EPSILON = 0.3
H, W = len(room), len(room[0])
WALL = 1
PATH = 0
END = (8, 8)
ACTIONS = [(-1,0),(0,1),(1,0),(0,-1)]
MAX_STEPS = 200
MEMORY = deque()
MAX_MEMORY = 10000
FINISHED = False

steps = 0
reward = 10
current_position = (1,1)
state = [[current_position[0] / H, current_position[1] / W]]
state_tensor = torch.tensor(state)

# Functions
def is_valid(r, c):
    return 0 <= r < H and 0 <= c < W and room[r][c] == 0

# Neural Network
class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 4)
        )
    def forward(self, x):
        return self.net(x)
model = DQN()

while current_position != END and steps < MAX_STEPS:    
    steps += 1
    reward -= 0.1
    q_values = model(state_tensor)
    best_actions = q_values.argsort(dim=1, descending=True)
        
    for i in best_actions[0]:
        if random.randrange(0, 1) < EPSILON:
            best_action_index = best_actions[0][random.randint(0, 3)].item()
        else:
            best_action_index = i.item()
        next_position = (current_position[0] + ACTIONS[best_action_index][0], current_position[1] + ACTIONS[best_action_index][1])
        print(f"Current: ({current_position[0]},{current_position[1]})")
        print("Q-Values:", q_values)
        print("Best action:", best_action_index)
        print("Next position:", next_position)

        if not is_valid(next_position[0], next_position[1]):
            print("Current best action invalid..")
            print("Moving onto next best action..")
            print()
        else:
            MEMORY.append((state, ACTIONS[best_action_index], reward, next_position, FINISHED))
            
            if len(MEMORY) > MAX_MEMORY:
                MEMORY.popleft()

            current_position = next_position
            state_tensor = torch.tensor([[current_position[0]/H, current_position[1]/W]])
            break
        
if current_position == END:
    print("\nEnd goal reached!")
    FINISHED = True
    MEMORY.append((state, ACTIONS[best_action_index], reward + 10, next_position, FINISHED))