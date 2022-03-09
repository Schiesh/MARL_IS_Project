import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time

style.use("ggplot")

SIZE = 9
HM_EPISODES = 25000
MOVE_PENALTY = 1
OUT_OF_BOUNDS = 1000
CLEAN_REWARD = 25
epsilon = 0.9 # Start at 0.9, Once trained change value to 0.0
EPS_DECAY = 0.9998
SHOW_EVERY = 300
MOVE_TIME = 100 # Do not set to zero, the > the number the slower the objects go.
DIRT_MOD = 2

start_q_table = None # Once trained put "filename.pickle" here

LEARNING_RATE = 0.1
DISCOUNT = 0.95

PLAYER_N = 1
ENEMY_N = 2

d = {1: (255, 175, 0),
     2: (0, 0, 255)}


class Vacuum:
    def __init__(self):
        self.x = np.random.randint(1,SIZE-1)
        self.y = np.random.randint(1, SIZE-1)

    def __str__(self):
        return f"{self.x}, {self.y}"

    def action(self, choice):
        # Down
        if choice == 0:
            self.move(x=0, y=1)
        # Up
        elif choice == 1:
            self.move(x=0, y=-1)
        # Right
        elif choice == 2:
            self.move(x=1, y=0)
        # Left
        elif choice == 3:
            self.move(x=-1, y=0)

    def move(self, x=False, y=False):
        self.x += x
        self.y += y

        # OUT_OF_BOUNDS
        if self.x < 0:
            self.x = 0
        elif self.x > SIZE - 1:
            self.x = SIZE - 1

        if self.y < 0:
            self.y = 0
        elif self.y > SIZE - 1:
            self.y = SIZE - 1

   

class Mold():

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return f"{self.x}, {self.y}"

    def action(self, choice):
        # Down
        if choice == 0:
            self.move(x=0, y=1)
        # Up
        elif choice == 1:
            self.move(x=0, y=-1)
        # Right
        elif choice == 2:
            self.move(x=1, y=0)
        # Left
        elif choice == 3:
            self.move(x=-1, y=0)

    def move(self, x=False, y=False):
        self.x += x
        self.y += y

        # OUT_OF_BOUNDS
        if self.x < 0:
            self.x = 0
        elif self.x > SIZE - 1:
            self.x = SIZE - 1

        if self.y < 0:
            self.y = 0
        elif self.y > SIZE - 1:
            self.y = SIZE - 1


if start_q_table is None:
    vacuum_q_table = {}
    # Vacuum x
    for i in range(0, SIZE):
        # Vacuum y
        for ii in range(0, SIZE):
            # Dirt x
            for iii in range(0, SIZE):
                # Dirt y
                for iiii in range(0, SIZE):
                    vacuum_q_table[((i, ii), (iii, iiii))] = [np.random.uniform(-5, 0) for i in range(4)] # range is # of actions
else:
    with open(start_q_table, 'rb') as f:
        vacuum_q_table = pickle.load(f)

episode_rewards = []

for episode in range(HM_EPISODES):
    vacuum = Vacuum()
    dirt = Mold(4, 4)

    if episode % SHOW_EVERY == 0:
        print(f"on #{episode}, epsilon is {epsilon}")
        print(f"{SHOW_EVERY} ep mean {np.mean(episode_rewards[-SHOW_EVERY:])}")
        show = True
    else:
        show = False

    episode_reward = 0
    dirt_wait = 0

    for i in range(200):

        obs = ((vacuum.x, vacuum.y), (dirt.x, dirt.y))
        
        if np.random.random() > epsilon:
            vacuum_action = np.argmax(vacuum_q_table[obs])
        else:
            vacuum_action = np.random.randint(0, 4)

        dirt_action = np.random.randint(0, 4)


        # Rewards
        vacuum.action(vacuum_action)
        reward = 0
        if vacuum.x == dirt.x and vacuum.y == dirt.y:
            reward = CLEAN_REWARD
        elif vacuum.x == SIZE-1 or vacuum.x == 0 or vacuum.y == SIZE-1 or vacuum.y == 0:
            reward = -OUT_OF_BOUNDS
        else:
            reward = -MOVE_PENALTY
            mod_dirt_wait = dirt_wait % DIRT_MOD
            if mod_dirt_wait == 0:
                dirt.action(dirt_action)
            dirt_wait += 1   

        new_obs = ((vacuum.x, vacuum.y), (dirt.x, dirt.y))

        vacuum_max_future_q = np.max(vacuum_q_table[new_obs])
        vacuum_current_q = vacuum_q_table[obs][vacuum_action]

        if reward == CLEAN_REWARD:
            vacuum_new_q = CLEAN_REWARD
        else:
            vacuum_new_q = (1 - LEARNING_RATE) * vacuum_current_q + LEARNING_RATE * (episode_reward + DISCOUNT + vacuum_max_future_q)


        vacuum_q_table[obs][vacuum_action] = vacuum_new_q

        if show:
            env = np.zeros((SIZE, SIZE, 3), dtype = np.uint8)
            env[vacuum.y][vacuum.x] = d[PLAYER_N]
            env[dirt.y][dirt.x] = d[ENEMY_N]

            img = Image.fromarray(env, "RGB")
            img = img.resize((500, 500), resample=Image.NEAREST)
            cv2.imshow("", np.array(img))
            cv2.waitKey(MOVE_TIME)

            if reward == CLEAN_REWARD or reward == -OUT_OF_BOUNDS:
                if cv2.waitKey(500) & 0xFF == ord("q"):
                    break
            else:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break  

        episode_reward += reward
        if reward == CLEAN_REWARD or reward == -OUT_OF_BOUNDS:
            cv2.destroyAllWindows()
            break
        
    
    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY

moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode="valid")

plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"reward {SHOW_EVERY}ma")
plt.xlabel("episode #")
plt.show()

with open(f"vacuumqtable-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(vacuum_q_table, f)
