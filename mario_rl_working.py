# General Imports
from tqdm import tqdm
import pickle
import collections
import cv2
import numpy as np
from collections import namedtuple, deque
import itertools
from PIL import Image as im
import imageio
import os
import matplotlib.pyplot as plt
from matplotlib import animation

# OpenAI Gym Imports
import gym
from gym.spaces import Box
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym_super_mario_bros.actions import RIGHT_ONLY

# Torch Imports
import torch
import torch.nn as nn
import random
from torchvision import transforms as T
import torch.nn.functional as F
import torch.optim as optim

# GIF Imports
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
from IPython import display
from moviepy.editor import ImageSequenceClip

#### 1. General Setup - Random mario player and its rewards

# Initializing environment
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
done = True
frames = np.zeros((500, 240,256,3), dtype=np.uint8)
reward_tracking = []
x_position = []
data = []

# Running the first 500 random iterations of mario
for step in range(500):
    frames[step] = env.render(mode = 'rgb_array')
    if done:
        state = env.reset()
    state, reward, done, info = env.step(env.action_space.sample())
    reward_tracking.append(reward)
    data.append(info)
    x_position.append(info['x_pos'])
      
# Reward Tracking Plots
r = np.array(reward_tracking)
plt.plot(reward_tracking)
plt.title('Reward tracking')
plt.show()

# X-position Plots
plt.plot(x_position)
plt.title("Mario's Movement along the X-Axis")
plt.show
print(data[1])

# GIF Creation
clip = ImageSequenceClip(list(frames), fps=20)
clip.write_gif('mario.gif', fps=20)

# GIF Display
with open("mario.gif",'rb') as f:
    display.Image(data=f.read(), format='png')

#### 1.5 Gym Wrappers
class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        # Here we return only the every 4th frame
        super(MaxAndSkipEnv, self).__init__(env)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info

    def reset(self):
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs

class ProcessFrame84(gym.ObservationWrapper):
    # Scales the frames dow from 255 to 84 by 84
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame84.process(obs)

    @staticmethod
    def process(frame):
        if frame.size == 240 * 256 * 3:
            img = np.reshape(frame, [240, 256, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)


class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]),
                                                dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)

class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(old_space.low.repeat(n_steps, axis=0),
                                                old_space.high.repeat(n_steps, axis=0), dtype=dtype)

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer

class ScaledFloatFrame(gym.ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0

class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        # permute [H, W, C] array to [C, H, W] tensor
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation

class myClass:
	def __init__(self,val):
		self.val=val
	def getVal(self):
		return self.val

def make_env(env):
    env = MaxAndSkipEnv(env)
    env = ProcessFrame84(env)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, 4)
    env = ScaledFloatFrame(env)
    env = GrayScaleObservation(env)
    return JoypadSpace(env, RIGHT_ONLY)

#### 2. Environment Preprocessing

def preprocess(frame):
    frame = frame.sum(axis=-1)/765 # RGB reducing
    frame = frame[20:210,:] #Frame grapping
    frame = frame[::2,::2] # Group 4 frames together
    return frame

#### 3. Model Instantiation

### Dueling Double Deep Q Network

class DQNSolver(nn.Module):
# this is the start of the Deep Q-Network policy network

    def __init__(self, channels, action_size, seed=42):
        super(DQNSolver, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.conv1 = nn.Conv2d(channels, 4, 3, padding=1)
        self.conv2 = nn.Conv2d(4, 8, 3, padding=1)
        self.conv3 = nn.Conv2d(8, 16, 3, padding=1)
        self.conv4 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv5 = nn.Conv2d(16, 16, 3, padding=1)
        
        self.pool = nn.MaxPool2d(2, ceil_mode=True)

        flat_len = 16*3*4
        self.fc1 = nn.Linear(flat_len, 20)
        self.fc2 = nn.Linear(20, action_size)

    def forward(self, x):
        #Info: Network that maps state -> action values 
        # Taking in the frames and outputting the actions
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))

        x = x.reshape(x.shape[0], -1)

        x = F.relu(self.fc1(x)) # Activation function
        x = self.fc2(x)
        
        return x

# Note: Instantiation of Duelling DQN
class QNetworkDuellingCNN(nn.Module):
    # Actor (Policy) Model

    def __init__(self, channels, action_size, seed=42):

        # state_size (int): Dimension of each state
        # action_size (int): Dimension of each action
        # seed (int): Random seed

        super(QNetworkDuellingCNN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.conv1 = nn.Conv2d(channels, 4, 3, padding=1)
        self.conv2 = nn.Conv2d(4, 8, 3, padding=1)
        self.conv3 = nn.Conv2d(8, 16, 3, padding=1)
        self.conv4 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv5 = nn.Conv2d(16, 16, 3, padding=1)
        
        self.pool = nn.MaxPool2d(2, ceil_mode=True)
        
        flat_len = 16*3*4
        self.fcval = nn.Linear(flat_len, 20)
        self.fcval2 = nn.Linear(20, 1)
        self.fcadv = nn.Linear(flat_len, 20)
        self.fcadv2 = nn.Linear(20, action_size)

    def forward(self, x):
        # Building a network that maps state -> action values.
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))

        x = x.reshape(x.shape[0], -1)
        
        advantage = F.relu(self.fcadv(x))
        advantage = self.fcadv2(advantage)
        advantage = advantage - torch.mean(advantage, dim=-1, keepdim=True)
        # here we subtract the mean from the advantage function to improve funcitonality
        value = F.relu(self.fcval(x))
        value = self.fcval2(value)

        return value + advantage

#### 4. Memory Instantiation,
### Here is where the saved states, the buffer, and all the memory implementation are stored

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 #Note: implement and test cuda fucntionality

class ReplayBuffer: 
  def __init__(self, state_size, action_size, buffer_size, batch_size, priority=False):
        #Info: here I am initializing the memory network to hold the appropriate values
        #using torch arrays
        self.states = torch.zeros((buffer_size,)+state_size).to(device)
        self.next_states = torch.zeros((buffer_size,)+state_size).to(device)
        self.actions = torch.zeros(buffer_size,1,dtype=torch.long).to(device)
        self.rewards = torch.zeros(buffer_size,1,dtype=torch.float).to(device)
        self.dones = torch.zeros(buffer_size,1,dtype=torch.float).to(device)
        self.e = np.zeros((buffer_size,1),dtype=np.float)

        self.priority = priority

        self.ptr = 0
        self.n = 0
        self.buffer_size = buffer_size
        self.batch_size = batch_size

  def add(self, state, action, reward, next_state, done):
    #Info: Incrementing the memory forward with each new step
    self.states[self.ptr] = torch.from_numpy(state).to(device)
    self.next_states[self.ptr] = torch.from_numpy(next_state).to(device)
    self.actions[self.ptr] = action
    self.rewards[self.ptr] = reward
    self.dones[self.ptr] = done
        
    self.ptr += 1
    if self.ptr >= self.buffer_size:
        self.ptr = 0
        self.n = self.buffer_size

  def sample(self, get_all=False):
        #Info: Take a random section of experiences from the memory
        n = len(self)
        if get_all:
            return self.states[:n], self.actions[:n], self.rewards[:n], self.next_states[:n], self.dones[:n]
        # prioritized replay: prioritize the replay with the highest errors
        if self.priority:
            idx = np.random.choice(n, self.batch_size, replace=False, p=self.e)
        else:
            idx = np.random.choice(n, self.batch_size, replace=False)
        
        states = self.states[idx]
        next_states = self.next_states[idx]
        actions = self.actions[idx]
        rewards = self.rewards[idx]
        dones = self.dones[idx]

        return (states, actions, rewards, next_states, dones), idx

  def update_error(self, e, idx=None):
        e = torch.abs(e.detach())
        e = e / e.sum()
        if idx is not None:
            self.e[idx] = e.cpu().numpy()
        else:
            self.e[:len(self)] = e.cpu().numpy()
        
  def __len__(self):
        if self.n == 0:
            return self.ptr
        else:
            return self.n

#### 5. Agent Actions and Training

#Note: parameters for Duelling-DQN applicaiton here

buffer_size = 500 # size of training attempts
batch_size = 256
gamma = 0.99
TAU = .001
learn_rate = .0005 # rate at which the agent learns
learn_each = 10  # update_every in the duelling version

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(torch.device)

class DQNMario():
  #Info: This is the agent itself that interacts with and learns from the environment by taking advantage of the DL networks and the memory


  def __init__(self, model, state_size, action_size, seed=42, ddqn=False, priority=False):
    self.state_size = state_size
    self.action_size = action_size
    self.seed = random.seed(seed)
    self.ddqn = ddqn

    #Info: Here is the Q-Network and its saving to the online and target networks
    self.qnetwork_online = model(state_size[0], action_size, seed).to(device)
    self.qnetwork_target = model(state_size[0], action_size, seed).to(device)
    self.optimizer = optim.Adam(self.qnetwork_online.parameters(), lr=learn_rate)

    #Info: Here is the instantiation of the Replay memory
    self.memory = ReplayBuffer(state_size, (action_size,), buffer_size, batch_size)
    
    #Info: Initialize time step (for updating every UPDATE_EVERY steps)
    self.t_step = step

  def step(self, state, action, reward, next_state, done):
        #Info: Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        #Info: Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % learn_each
        if self.t_step == 0:
            #Info: If enough samples are available in memory, get random subset and learn
            if len(self.memory) > batch_size:
                experiences, idx = self.memory.sample()
                e = self.learn(experiences)
                self.memory.update_error(e, idx)

  def act(self, state, eps=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_online.eval()
        with torch.no_grad():
            action_values = self.qnetwork_online(state)
        self.qnetwork_online.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

  def update_error(self):
        states, actions, rewards, next_states, dones = self.memory.sample(get_all=True)
        with torch.no_grad():
            if self.ddqn:
                old_val = self.qnetwork_online(states).gather(-1, actions)
                actions = self.qnetwork_online(next_states).argmax(-1, keepdim=True)
                maxQ = self.qnetwork_target(next_states).gather(-1, actions)
                target = rewards+gamma*maxQ*(1-dones)
            else: # Normal DQN
                maxQ = self.qnetwork_target(next_states).max(-1, keepdim=True)[0]
                target = rewards+gamma*maxQ*(1-dones)
                old_val = self.qnetwork_online(states).gather(-1, actions)
            e = old_val - target
            self.memory.update_error(e)

  def learn(self, experiences):    
        states, actions, rewards, next_states, dones = experiences

        #Info: minimize the loss by using the optimizer
        self.optimizer.zero_grad()
        if self.ddqn:
            old_val = self.qnetwork_online(states).gather(-1, actions)
            with torch.no_grad():
                next_actions = self.qnetwork_online(next_states).argmax(-1, keepdim=True)
                maxQ = self.qnetwork_target(next_states).gather(-1, next_actions)
                target = rewards+gamma*maxQ*(1-dones) # target network
        else: # Normal DQN
            with torch.no_grad():
                maxQ = self.qnetwork_target(next_states).max(-1, keepdim=True)[0]
                target = rewards+gamma*maxQ*(1-dones)
            old_val = self.qnetwork_online(states).gather(-1, actions)   
        
        loss = F.mse_loss(old_val, target) # calculating the loss with PyTorch Functions and mean squeared error
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnetwork_online, self.qnetwork_target, TAU) 
        return old_val - target

  def soft_update(self, online_model, target_model, tau):
        for target_param, online_param in zip(target_model.parameters(), online_model.parameters()):
            target_param.data.copy_(tau*online_param.data + (1.0-tau)*target_param.data)

#### 6. Running Agents Training:
# here is where I am training the agent by running it through the Duelling DQN and the states and environment

episode = 100 #only running 100 episodes of training at the start
# add more episodes for training later
discount_rate = .99
noise = 0.05
noise_decay = 0.99
tmax = 500

# keeping track of progress
sum_rewards = []

# keeping track of frames
FRAME_SHAPE = (95, 128)
MAX_FRAMES = 4
nn_frames = deque(maxlen=MAX_FRAMES)
for i in range(MAX_FRAMES):
    nn_frames.append(np.zeros(FRAME_SHAPE))
    
action_size = 7 # length of the number of valid_actions
state_size = (MAX_FRAMES,) + FRAME_SHAPE
agent = DQNMario(QNetworkDuellingCNN, state_size, action_size, ddqn=True, priority=True)

# For loop running the training
for e in range(episode):
    obs = env.reset()
    prev_obs = None
    sum_reward = 0

    # Processing the frames
    for i in range(MAX_FRAMES):
        nn_frames.append(np.zeros(FRAME_SHAPE))
    nn_frames.append(np.copy(preprocess(obs)))
    states = np.array(nn_frames)

    # Training and rewards
    for t in range(tmax):
        actions = agent.act(states, noise)
        obs, reward, done, _ = env.step(actions)
        nn_frames.append(np.copy(preprocess(obs)))
        next_states = np.array(nn_frames)
        
        agent.step(states, int(actions), int(reward), next_states, int(done))
        sum_reward += reward
        states = next_states

        if done or reward < -10:
            break
    
    agent.update_error()
    sum_rewards.append(sum_reward)
    noise = noise * noise_decay
    
    print('\rEpisode {}\tCurrent Score: {:.2f}'.format(e, sum_rewards[-1]), end="")
    # display some progress every 20 iterations
    if (e+1) % (episode // 20) ==0:
        print(" | Episode: {0:d}, average score: {1:f}".format(e+1,np.mean(sum_rewards[-20:])))


#### 7. Visualizing and Assessing Performance

obs = env.reset()
prev_obs = None
sum_reward = 0

frames = np.zeros((tmax, 240, 256, 3), dtype=np.uint8)
for i in range(MAX_FRAMES):
  nn_frames.append(np.zeros(FRAME_SHAPE))
nn_frames.append(np.copy(preprocess(obs)))
states = np.array(nn_frames)
reward_total = []
x_total = []
y_total = []

# Running example loop
for t in range(tmax):
    frames[t] = obs
    actions = agent.act(states, noise)
    obs, reward, done, info = env.step(actions)
    nn_frames.append(np.copy(preprocess(obs)))
    next_states = np.array(nn_frames)

    sum_reward += reward
    states = next_states
    reward_total.append(reward)
    x_total.append(info['x_pos'])
    y_total.append(info['y_pos'])
    if done:
        break

#Showing reward total
print('Sum of rewards is ', sum(reward_total))

#Plotting the reward total
plt.plot(reward_total)
plt.show()

plt.plot(x_total)
plt.show()

# GIF Creation
clip = ImageSequenceClip(list(frames), fps=20)
clip.write_gif('mario_2.gif', fps=20)

# GIF Display
with open("mario_2.gif",'rb') as f:
    display.Image(data=f.read(), format='png')