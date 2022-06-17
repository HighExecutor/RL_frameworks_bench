import matplotlib.pyplot as plt
import numpy as np

class ActionPlotter:
    def __init__(self, n_actions):
        plt.ion()
        self.x = np.arange(n_actions)
        self.bar = plt.bar(self.x, np.zeros(n_actions))
        self.min = 0.0
        self.max = 1.0
        self.update_data(np.zeros(n_actions), 0)
        plt.autoscale(True, tight=True)
        plt.show()


    def update_data(self, action_values, action):
        if action_values  is not None:
            min = action_values.min()
            max = action_values.max()
            if self.min > min:
                self.min = min
            if self.max < max:
                self.max = max
            for rect, h in zip(self.bar, action_values):
                rect.set_height(h)
            plt.ylim(self.min, self.max)
            plt.pause(0.00001)

    def close(self):
        plt.ioff()

if __name__ == '__main__':
    act_plot = ActionPlotter(4)
    for _ in range(10000):
        act_plot.update_data(np.random.random(4))


