import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import gymnasium as gym
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import matplotlib.pyplot as plt
from collections import deque
import random
def epsilon_greedy(Q_s, n_actions, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)
    else:
        return np.argmax(Q_s, axis=1)[0]
def demo_run(sess, env, Q_values, State, state_size, n_actions,
             norm_mean, norm_std):
    state, info = env.reset()
    steps = 0
    done = False
    while not done:
        s = (state - norm_mean) / (norm_std + 1e-8)
        s_cur = s.reshape(1, state_size)
        Q_s = Q_values.eval(feed_dict={State: s_cur}, session=sess)
        a = epsilon_greedy(Q_s, n_actions, 0.0)
        state, reward, terminated, truncated, info = env.step(a)
        env.render()
        done = terminated or truncated
        steps += 1
    return steps
def build_q_network(State, name, hidden=64):
    with tf.compat.v1.variable_scope(name):
        W1 = tf.Variable(tf.random.truncated_normal([4, hidden], stddev=0.1))
        b1 = tf.Variable(tf.zeros([hidden]))
        W2 = tf.Variable(tf.random.truncated_normal([hidden, hidden],
                                                     stddev=0.1))
        b2 = tf.Variable(tf.zeros([hidden]))
        W3 = tf.Variable(tf.random.truncated_normal([hidden, 2], stddev=0.1))
        b3 = tf.Variable(tf.zeros([2]))
        h1 = tf.nn.relu(tf.matmul(State, W1) + b1)
        h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)
        Q_values = tf.matmul(h2, W3) + b3
    trainable_vars = tf.compat.v1.get_collection(
        tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=name)
    return Q_values, trainable_vars
class ReplayBuffer:
    def __init__(self, capacity=20000):
        self.buffer = deque(maxlen=capacity)
    def push(self, s, a, r, s_next, done):
        self.buffer.append((s.copy(), a, r, s_next.copy(), done))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_next, done = zip(*batch)
        return (np.vstack(s), np.array(a), np.array(r, dtype=np.float32),
                np.vstack(s_next), np.array(done, dtype=np.float32))
    def __len__(self):
        return len(self.buffer)
State = tf.compat.v1.placeholder(tf.float32, shape=[None, 4])
Q_online, online_vars = build_q_network(State, "online", hidden=64)
Q_target, target_vars = build_q_network(State, "target", hidden=64)
copy_ops = [tv.assign(ov) for tv, ov in zip(target_vars, online_vars)]
Target = tf.compat.v1.placeholder(tf.float32, shape=[None])
Action = tf.compat.v1.placeholder(tf.int32, shape=[None])
Q_selected = tf.reduce_sum(
    Q_online * tf.one_hot(Action, 2), axis=1)
loss = tf.reduce_mean(tf.square(Target - Q_selected))
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0005)
training_op = optimizer.minimize(loss)
gamma = 0.99
n_episodes = 500
epsilon_max = 1.0
epsilon_min = 0.01
epsilon_decay = 0.997
batch_size = 32
replay_start = 1000
train_every = 4
target_update_every = 1000
demo_interval = 250
norm_mean = np.array([0.0, 0.0, 0.0, 0.0])
norm_std  = np.array([2.4, 2.0, 0.24, 2.0])
train_env = gym.make("CartPole-v1")
demo_env = gym.make("CartPole-v1", render_mode="human")
episode_steps_list = []
episode_losses = []
episode_epsilons = []
with tf.compat.v1.Session() as sess:
    tf.compat.v1.global_variables_initializer().run()
    sess.run(copy_ops)
    replay = ReplayBuffer(20000)
    epsilon = epsilon_max
    total_steps = 0
    train_steps = 0
    for ep in range(n_episodes):
        state, info = train_env.reset()
        done = False
        steps = 0
        loss_sum = 0.0
        update_count = 0
        while not done:
            s_norm = (state - norm_mean) / (norm_std + 1e-8)
            s_cur = s_norm.reshape(1, 4)
            Q_s = Q_online.eval(feed_dict={State: s_cur}, session=sess)
            a_cur = epsilon_greedy(Q_s, 2, epsilon)
            state, reward, terminated, truncated, info = train_env.step(a_cur)
            done = terminated or truncated
            s_next_norm = (state - norm_mean) / (norm_std + 1e-8)
            s_next = s_next_norm.reshape(1, 4)
            steps += 1
            total_steps += 1
            replay.push(s_cur, a_cur, 0.0 if done else 1.0, s_next, done)
            if len(replay) >= replay_start and total_steps % train_every == 0:
                s_batch, a_batch, r_batch, s_next_batch, done_batch = \
                    replay.sample(batch_size)
                Q_next = Q_target.eval(
                    feed_dict={State: s_next_batch}, session=sess)
                target = r_batch + gamma * np.max(Q_next, axis=1) * \
                         (1 - done_batch)
                feed_dict = {State: s_batch, Action: a_batch, Target: target}
                _, loss_val = sess.run([training_op, loss], feed_dict=feed_dict)
                loss_sum += loss_val
                update_count += 1
                train_steps += 1
                if train_steps % target_update_every == 0:
                    sess.run(copy_ops)
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
        episode_steps_list.append(steps)
        episode_epsilons.append(epsilon)
        episode_losses.append(loss_sum / max(update_count, 1))
        if (ep + 1) % demo_interval == 0:
            demo_steps = demo_run(sess, demo_env, Q_online, State,
                                  4, 2, norm_mean, norm_std)
            avg_loss = loss_sum / max(update_count, 1)
            print(f"Episode {ep + 1:4d} | Train: {steps:3d} | "
                  f"Demo: {demo_steps:3d} | Epsilon: {epsilon:.4f} | "
                  f"Loss: {avg_loss:.4f}")
train_env.close()
demo_env.close()
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(10, 5))
plt.plot(episode_steps_list, linewidth=0.5, color='#2196F3')
plt.xlabel('训练轮次')
plt.ylabel('步数')
plt.title('DQN — 每轮步数')
plt.grid(True, alpha=0.3)
plt.show()
plt.figure(figsize=(10, 5))
plt.plot(episode_epsilons, linewidth=0.7, color='#FF9800')
plt.xlabel('训练轮次')
plt.ylabel('Epsilon')
plt.title('DQN — 探索率衰减')
plt.grid(True, alpha=0.3)
plt.show()
window = 50
if len(episode_losses) >= window:
    smoothed = np.convolve(episode_losses, np.ones(window) / window,
                           mode='valid')
    plt.figure(figsize=(10, 5))
    plt.plot(range(window - 1, len(episode_losses)), smoothed,
             linewidth=1.2, color='#4CAF50')
    plt.xlabel('训练轮次')
    plt.ylabel('平均损失')
    plt.title('DQN — 平均损失')
    plt.grid(True, alpha=0.3)
    plt.show()
