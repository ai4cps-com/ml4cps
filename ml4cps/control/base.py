import gymnasium as gym
import simpy
from gymnasium import spaces
import numpy as np
from ml4cps import automata, vis
from plotly import graph_objects as go


class Agent:
    def __init__(self, policy, gamma):
        self.policy = policy
        self.gamma = gamma


class EnvironmentTA(gym.Env):
    """Custom Environment that follows Gymnasium interface"""
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, automaton):
        super().__init__()
        self.ta = automaton

        # Define action and observation spaces
        # For example, actions: discrete (0 or 1)
        # self.action_space = spaces.Discrete(2)

        # Observations: continuous space with shape (3,)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # Define state variables
        self.max_episode_length = 100  # Maximum steps in an episode
        self._dash = None

    @property
    def action_space(self):
        return spaces.Discrete(self.ta.num_events)

    @property
    def state(self):
        return self.ta.state

    @property
    def episode_length(self):
        return len(self.ta._discrete_state_data) - 1

    def reset(self, seed=None, options=None):
        """Reset the environment to an initial state and return the initial observation."""
        super().reset(seed=seed)

        self.ta.reinitialize(t=0, state=self.ta.sample_initial())
        finish_time = np.inf
        self.ta._env = simpy.Environment()
        self.ta.simulation_process_simpy(self.ta._env, finish_time, verbose=False)
        while self.ta._env.peek() != float('inf'):  # While there are events in the queue
            self.ta._env.step()
        return self.ta.state, {}

    def step(self, action):
        """Take an action and return the next observation, reward, done, and info."""
        # Update the state based on action
        valid_actions = set([x[3]['event'] for x in self.ta.out_transitions(self.ta._q)])
        action = self.ta.Sigma[action]
        if action not in valid_actions:
            # Invalid action penalty
            reward = -100
            done = False
            info = {}
            print('Rejected action', action)
            return self.ta.state, reward, done, False, info
        print(action)

        rewards = {e[3]['event']: e[3].get('r', 0) for e in self.ta.out_transitions(self.ta._q, action)}
        self.ta.apply_sim_event(action)
        reward = rewards.get(action, 0)

        # self.ta.simulation_process_simpy(self.ta._env, np.inf)
        while self.ta._env.peek() != float('inf'):  # While there are events in the queue
            self.ta._env.step()

        # Compute reward (example logic)
        # reward = self.ta._discrete_state_data[-1].get('r', 0)

        # Check if the episode is done
        done = self.episode_length >= self.max_episode_length

        # Optionally provide debug information
        # info = {"example_key": "example_value"}
        info = {}
        return self.ta.state, reward, done, False, info

    def render(self, mode="human"):
        """Render the environment. No-op in this example."""
        if mode == "human":
            if self._dash is None:
                self._dash = vis.plot_cps_component(self.ta, output="dash")
            else:
                print('Updating dash cytoscape figure')
            # self.fig.update_traces(x=[self.state[0]], y=[self.state[1]], selector=dict(mode="markers"))
            # self.fig.update_layout(
            #     title="Environment State",
            #     xaxis=dict(range=[-1, 1], title="X"),
            #     yaxis=dict(range=[-1, 1], title="Y"),
            #     showlegend=False
            # )

    def close(self):
        """Clean up resources if necessary."""
        pass

    def expected_cum_reward(self, policy, discounted=True, max_depth=2):
        q = next(iter(self.ta.q0))
        ev = policy[q]

        # for out in self.ta.out_transitions(q, event=ev):


    # def step_reward(self):

if __name__ == "__main__":
    # Apply Q learning

    env = automata.Automaton(states=["s0", "s1", "s2", "s3"], initial_q="s0", final_q="s0",
                       transitions=[dict(source="s0", event="A", dest="s1", p=0.1, r=17),
                                    dict(source="s1", event="B", dest="s2", p=3, r=12),
                                    dict(source="s0", event="A", dest="s2", p=0.9, r=1),
                                    dict(source="s2", event="C", dest="s3", p=1, r=10),
                                    dict(source="s0", event="C", dest="s3", p=1, r=3),
                                    dict(source="s3", event="C", dest="s0", p=1, r=50)])
    env = EnvironmentTA(automaton=env)
    vis.plot_cps_component(env.ta, output="dash", color="hsu", min_zoom=3, max_zoom=3, node_labels=True, edge_labels=True,
                        center_node_labels=True, show_transition_data=['p', 'r'])

    exit()
    # Initialize Q-table
    q_table = np.zeros([bins] * 8 + [env.action_space.n])

    # Q-learning algorithm
    rewards = []
    for episode in range(3):
        state = discretize_state(env.reset()[0])  # Reset environment
        total_reward = 0
        for step in range(max_steps):
            # Choose an action using epsilon-greedy policy
            if np.random.rand() < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(q_table[state])  # Exploit

            # Take action and observe the next state and reward
            next_state, reward, done, truncated, _ = env.step(action)
            next_state = discretize_state(next_state)
            total_reward += reward

            delta = gamma * np.max(q_table[next_state]) - q_table[state][action]
            # Q-value update
            q_table[state][action] += alpha * (reward + delta)

            # Transition to the next state
            state = next_state

            if done:
                break

        # Decay epsilon to reduce exploration over time
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Store total reward for this episode
        rewards.append(total_reward)

        # Print progress
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

    # Close environment
    env.close()


    exit()
    # Register the custom environment (optional)
    # gym.envs.registration.register(
    #     id="CustomEnv-v0",
    #     entry_point="path.to.module:EnvironmentTA"
    # )

    # env = automata.Automaton(states=["s1", "s2", "s3"], initial_q="s1",
    #                          transitions=[dict(source="s1", event="a1", dest="s2", timing_dist=1, p=0.8, r=10),
    #                                       dict(source="s1", event="a1", dest="s3", timing_dist=1, p=0.2, r=5),
    #                                       dict(source="s1", event="a2", dest="s3", p=1, r=7),
    #                                       dict(source="s2", event="a1", dest="s3", p=0.9, r=0),
    #                                       dict(source="s2", event="a2", dest="s1", p=1, r=20),
    #                                       dict(source="s2", event="a1", dest="s1", p=0.1, r=15),
    #                                       dict(source="s3", event="a1", dest="s2", p=0.3, r=8),
    #                                       dict(source="s3", event="a2", dest="s2", p=1, r=-5),
    #                                       dict(source="s3", event="a1", dest="s1", p=0.7, r=12)])
    # env = EnvironmentTA(automaton=env)
    #
    # policy = {'s1': 'a1',
    #           's2': 'a2',
    #           's3': 'a1'}
    #
    # print(env.expected_cum_reward(policy=policy, discounted=True))
    # vis.plot_cps_component(env, output="dash", color="hsu", min_zoom=3, max_zoom=3, node_labels=True, edge_labels=True,
    #                        center_node_labels=True, show_transition_data=['p', 'r'])
    # exit()

    env = automata.Automaton(states=["s1", "s2", "s3"], initial_q="s1",
                             transitions=[dict(source="s1", event="a1", dest="s2", p=0.8, r=10),
                                          dict(source="s1", event="a1", dest="s3", p=0.2, r=5),
                                          dict(source="s1", event="a2", dest="s3", p=1, r=7),
                                          dict(source="s2", event="a1", dest="s3", p=0.9, r=0),
                                          dict(source="s2", event="a2", dest="s1", p=1, r=20),
                                          dict(source="s2", event="a1", dest="s1", p=0.1, r=15),
                                          dict(source="s3", event="a1", dest="s2", p=0.3, r=8),
                                          dict(source="s3", event="a2", dest="s2", p=1, r=-5),
                                          dict(source="s3", event="a1", dest="s1", p=0.7, r=12)])



    env = EnvironmentTA(automaton=env)

    observation, info = env.reset()
    print(observation)
    for step_ind in range(10):
        action = env.action_space.sample()
        print('Chosen action: ', action)
        observation, reward, done, _, info = env.step(action)
        print(observation)
        print('Reward: ', reward)
        # env.render()
        if done:
            break


    env.close()







