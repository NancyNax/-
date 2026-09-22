import gymnasium as gym
env = gym.make('CartPole-v1', render_mode='human')
action = 1
while True:
    state, info = env.reset()
    while True:
        state, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            break
