import numpy as np
from collections import defaultdict

SEED = 42
np.random.seed(SEED)

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
    
        self.nA = nA
        self.Q = defaultdict(lambda: np.random.normal(size=self.nA))

        self.td_methods = ["SARSA", "EXPECTED SARSA", "Q-LEARNING"]
        self.selected_method = self.td_methods[2]

        self.policies = ["GREEDY", "EPSILON GREEDY"]
        self.selected_policy = self.policies[1]

        #self.alpha = lambda x: 1.0
        self.alpha = lambda n: (1/(1+np.log(n)))

        self.gamma = 0.9
        self.epsilon = 1.0
        self.count_episode = 1

        self.next_action = None

        print(f"""
            --- AGENT INITIALIZED ---
                - method: {self.selected_method}
                - policy: {self.selected_policy}
                - alpha: {self.alpha}
                - gamma: {self.gamma}
                - epsilon: {self.epsilon}
            """)

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """

        action = None

        # We save computed next action from execute td_method to have coherent information for SARSA simulation
        if (self.next_action != None):
            action = self.next_action
            self.next_action = None
        else:
            if (self.selected_policy == "GREEDY"):
                action = self._greedy_policy(state)
            elif (self.selected_policy == "EPSILON GREEDY"):
                action = self._epsilon_greedy_policy(state)

        if action == None:
            raise NoPolicySpecifiedException()

        return action

    def _epsilon_greedy_policy(self, state):
        eps = self.epsilon/self.count_episode
        rdm = np.random.random()

        # Exploitation
        if (rdm > eps):
            action = self._greedy_policy(state)
        # Exploration
        else:
            action = np.random.choice(self.nA)
        
        return action

    def _greedy_policy(self, state):
        return np.argmax(self.Q[state])

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        self.count_episode += 1
        self._execute_td_method(state, action, reward, next_state, self.alpha(self.count_episode), self.gamma)

    def _execute_td_method(self, state, action, reward, next_state, alpha=0.2, gamma=1.0):
        """
            Generic method used to execute the selected td method
        """
        if (self.selected_method == "SARSA"):
            self._sarsa(state, action, reward, next_state, alpha, gamma)
        
        elif (self.selected_method == "EXPECTED SARSA"):
            self._expected_sarsa(state, action, reward, next_state, alpha, gamma)
    
        elif (self.selected_method == "Q-LEARNING"):
            self._q_learning(state, action, reward, next_state, alpha, gamma)

    def _sarsa(self, state, action, reward, next_state, alpha, gamma):
        next_action = self.select_action(next_state)
        self.Q[state][action] = self.Q[state][action] + alpha * (reward + gamma*self.Q[next_state][next_action] - self.Q[state][action])
    
    def _expected_sarsa(self, state, action, reward, next_state, alpha, gamma):
        eps = self.epsilon/self.count_episode
        probs = np.ones(self.nA) * (1 - eps) / (self.nA - 1)
        probs[np.argmax(self.Q[state])] = eps 
        self.Q[state][action] = self.Q[state][action] + alpha * (reward + gamma*np.dot(self.Q[next_state], probs) - self.Q[state][action])

    def _q_learning(self, state, action, reward, next_state, alpha, gamma):
        self.Q[state][action] = self.Q[state][action] + alpha * (reward + gamma*np.max(self.Q[next_state]) - self.Q[state][action])
        

class NoPolicySpecifiedException(Exception):

    def __init__(self):
        pass