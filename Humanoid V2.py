import random

class HumanoidRobot:
  def __init__(self):
    self.state = 0
    self.steps = 0
    self.reward = 0

  def take_action(self, action):
    if action == 0:
      self.state -= 1
    else:
      self.state += 1
    self.steps += 1
    self.reward = -1

  def reset(self):
    self.state = 0
    self.steps = 0
    self.reward = 0

class QTable:
  def __init__(self, num_states, num_actions):
    self.num_states = num_states
    self.num_actions = num_actions
    self.q_table = [[0.0 for i in range(num_actions)] for j in range(num_states)]

  def get_action(self, state):
    if random.uniform(0, 1) < epsilon:
      return random.choice([0, 1])
    else:
      return 0 if self.q_table[state][0] > self.q_table[state][1] else 1

  def update(self, state, action, reward, next_state):
    if state >= 0 and state < self.num_states and next_state >= 0 and next_state < self.num_states:
      alpha = 0.1
      gamma = 0.99
      next_q = max(self.q_table[next_state])
      self.q_table[state][action] = (1 - alpha) * self.q_table[state][action] + alpha * (reward + gamma * next_q)


robot = HumanoidRobot()
q_table = QTable(10, 2)

epsilon = 0.1
num_episodes = 1000

for episode in range(num_episodes):
  robot.reset()
  while robot.steps < 10:
    state = robot.state
    action = q_table.get_action(state)
    robot.take_action(action)
    next_state = robot.state
    reward = robot.reward
    q_table.update(state, action, reward, next_state)

print(q_table.q_table)
