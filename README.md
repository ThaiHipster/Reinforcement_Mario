# Reinforcement_Mario
A Reinforcement Learning Trained Mario Agent Overview:
I love the idea of humans competing to speed run completing video games in basically super human times. These human speed runners have inspired me to create a Reinforcement Learning Trained Super Mario Bros. agent that can hopefully compete with world record times. Currently I am using a Duelling Double Deep Q Network and my code can be found here. 

## Goals of Project
1. Train an Improving RL Agent
- Use PyTorch to implement a Duelling Double Deep Q-Network (Dueling Double DQN) that trains a mario agent to improve and perform better than random
- The mario agent can learn and improve in the completion of the level performing significantly better than random

2. Optimized Agent for Level Completion
- Edit and implement faster code in order to speed up the training of the agent
- Train the agent for a longer period of time so that it learns to complete the entire level
- Simplify the visual frame that the agent sees in order to lessen the computational requirement

3. Coached Agent for Better Performance
- Implemented two new additions to the Deep DQN: Tutor4RL and Deep Q-learning from Demonstrations
- Both these algorithms help the agent to learn the game faster and by providing human and expert guidance
- These could lead to the eventual goal of successfully creating a world-record level Mario Agent

## Algorithms Tested and Used
1. Q-Learning: Classic algorithm for Reinforcement Learning
- Model-free algorithm used to learn the value of an action given a certain state
- finds the optimal policy by maximizing the expected value of the total reward over time.
- "Q" refers to the function that the algorithm computes the expected rewards for an action taken in a given state.

2. Deep Q-Learning & Double Deep Q-Learning
- Deep Q-learning, utilizes neural networks to approximate the Q-value function
- Double DQN: Utilize a target network updated every certain number of iterations of an online network using 
- The target network has the same architecture as the online network but with frozen parameters until it is updated

3. Dueling Double Deep Q-Learning
- A dueling Double DQN has two streams that estimate state-value and the advantages separately from each other that are combined at the end
- Limits the overshooting of the model by separating state and advantages

## High Level Code Overview
1. Preprocess game images (our states) and show them to the DQN, to return the Q-values of all possible actions in the state
2. Utilize an epsilon-greedy policy implement action selection. 
3. If a random action isn’t selected select an action with a maximum online Q-value
4. Perform this action in a state s and move to a new state s’ to receive a reward. 
    The s’ is the processed image of the next game screen and we store this transition in our replay buffer as [ s,a,r,s’ ]
5. Sample some batches using prioritized replay of transitions from the replay buffer and calculate the loss
6. Perform gradient descent with respect to our actual network parameters in order to minimize this loss
7. After every C iterations, copy our actual network weights to the target network weights
8. Repeat these steps for E number of episodes

## Key Elements of Code 
### Networks: DQN and DuellingDQN
DQN: Takes in the frame dimensions as the input and outputs the representation of the state values
Duelling DQN: Takes in the same frames as input and outputs the state-dependent action advantages
Benefits: Allows the network to independently learn the values of states without having to learn the effect of each action

### Memory: Initialize, add, sample, update
Initialize the states, next_states, actions, rewards, dones, and errors
Add to memory the elements that are recorded: state, action, rewards, dones
Prioritized Replay replays experiences that were more important more frequently instead of randomly sampling them
Use the previously sampled experiences to update the Network

### Agent: Initialize, Step, Act, Update, Learn
Initialize the online and the target network. Use the adam optimizer
Step through the experiences saving the experiences to the replay buffer and moving the state forward
Act according to the policy or choose a random action depending on epsilon
Learn using the double dqn and copy the model over to the target model every certain number of iterations

## Agent Performance
### Positives
- The agent is able to avoid enemies
- Mario is able to jumps over small pipes
- Mario moves forward consistently
### Weaknesses
- Mario hasn’t learned to double jump
- Mario gets stuck easily and doesn't know his next move
- The agent is not as fast as possible

## Future Additions
### Algorithmic Additions
1. Tutor4RL
- Faster and earlier agent learning
- Ability to train agent on unlearned actions
- The impact decreases over time as to not handicap the network

2. Deep Q-Learning from Demonstration
- Pre-trained network for faster and earlier results
- Stored expert examples that can be returned to
- Mixed training on replay and expert replay
- Dynamic usage of expert replay

### Better Compute
- Better GPU optimization and usage
- Longer training periods (12 hours +)
- More Machines used 
- Parallel learning of different levels

### Different Rewards
- Adding in a stronger penalty for time elapsing
- Level specific world record rewards ex. rewarding wall glitches
- Attempts rewarding coins, enemies jumped on, or Y-height to promote jumping

### Optimized Code
- Implementing Gray Scaling
- Using different OpenAI version of the Mario Bros game
- Potential implementation of self referencing classes
- Reward clipping to provide a more uniform rewards



