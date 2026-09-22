import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
def softmax(scores):
    e = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)
def demo_run(env, W, state_size, n_actions, norm_mean, norm_std):
    state, info = env.reset()
    steps = 0
    done = False
    while not done:
        s = (state - norm_mean) / (norm_std + 1e-8)
        s = s.reshape(1, state_size)
        probs = softmax(s.dot(W))
        a = np.argmax(probs)
        state, reward, terminated, truncated, info = env.step(a)
        env.render()
        done = terminated or truncated
        steps += 1
    return steps
def actor_critic(train_env, demo_env, state_size, n_actions, n_iter, gamma,
                 eta_u, eta_W, demo_interval=200):
    norm_mean = np.array([0.0, 0.0, 0.0, 0.0])
    norm_std  = np.array([2.4, 2.0, 0.24, 2.0])
    W = np.random.randn(state_size, n_actions) * 0.01
    u = np.random.randn(state_size, 1) * 0.01
    episode_steps = []
    for ep in range(n_iter):
        state, info = train_env.reset()
        done = False
        S = []
        A = []
        R = []
        while not done:
            s_raw = state.reshape(1, state_size)
            s_norm = (s_raw - norm_mean) / (norm_std + 1e-8)
            probs = softmax(s_norm.dot(W))
            a_cur = np.random.choice(n_actions, p=probs.reshape(-1))
            S.append(s_norm)
            A.append(a_cur)
            state, reward, terminated, truncated, info = train_env.step(a_cur)
            done = terminated or truncated
            R.append(0.0 if done else reward)
        T = len(R)
        episode_steps.append(T)
        G = np.zeros(T)
        G[-1] = R[-1]
        for t in range(T - 2, -1, -1):
            G[t] = R[t] + gamma * G[t + 1]
        advantages = np.array([G[t] - float(S[t].dot(u).item())
                               for t in range(T)])
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - advantages.mean()) / adv_std
        dW = np.zeros_like(W)
        du = np.zeros_like(u)
        for t in range(T):
            s_norm = S[t]
            a_cur  = A[t]
            A_t    = float(advantages[t])
            probs = softmax(s_norm.dot(W))
            y = np.zeros((1, n_actions))
            y[0, a_cur] = 1.0
            grad_log = s_norm.T.dot(y - probs)
            dW += eta_W * A_t * grad_log
            du += eta_u * A_t * s_norm.T
        W += dW
        u += du
        if (ep + 1) % demo_interval == 0:
            demo_steps = demo_run(demo_env, W, state_size, n_actions,
                                  norm_mean, norm_std)
            print(f"Iteration {ep + 1:5d} | Train: {T:3d} steps | "
                  f"Demo: {demo_steps:3d} steps")
    return W, u, episode_steps, norm_mean, norm_std
if __name__ == "__main__":
    train_env = gym.make("CartPole-v1")
    demo_env = gym.make("CartPole-v1", render_mode="human")
    _, _, episode_steps, _, _ = actor_critic(
        train_env, demo_env, state_size=4, n_actions=2, n_iter=500,
        gamma=0.99, eta_u=0.05, eta_W=0.01)
    train_env.close()
    demo_env.close()
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(10, 5))
    plt.plot(episode_steps, linewidth=0.5, color='#2196F3')
    plt.xlabel('训练轮次')
    plt.ylabel('步数')
    plt.title('Actor-Critic — 每轮步数')
    plt.grid(True, alpha=0.3)
    plt.show()
    window = 50
    if len(episode_steps) >= window:
        smoothed = np.convolve(episode_steps, np.ones(window) / window,
                               mode='valid')
        plt.figure(figsize=(10, 5))
        plt.plot(range(window - 1, len(episode_steps)), smoothed,
                 linewidth=1.2, color='#FF5722')
        plt.xlabel('训练轮次')
        plt.ylabel('步数（滑动平均）')
        plt.title('Actor-Critic — 步数滑动平均')
        plt.grid(True, alpha=0.3)
        plt.show()
