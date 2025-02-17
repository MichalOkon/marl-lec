import matplotlib.pyplot as plt
import os


class RewardPlotter:
    def __init__(self, evaluation_interval, baseline_train_reward=None, baseline_test_reward=None,
                 l_curve_save_path=None):
        self.evaluation_interval = evaluation_interval
        self.l_curve_save_path = l_curve_save_path

        # Set up plot
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.line1, = self.ax.plot([], [], label='Training Rewards')
        self.line2, = self.ax.plot([], [], label='Evaluation Rewards')

        # Add baseline train and test reward lines if provided
        if baseline_train_reward is not None:
            self.ax.axhline(y=baseline_train_reward, color='blue', linestyle='--', label='Baseline Train Balance')

        if baseline_test_reward is not None:
            self.ax.axhline(y=baseline_test_reward, color='orange', linestyle='--', label='Baseline Test Balance')

        self.ax.set_xlabel('Training Iterations')
        self.ax.set_ylabel('Average Reward')
        self.ax.set_title('Training and Evaluation Rewards Over Time')
        self.ax.legend()
        self.ax.grid(True)

        self.training_std_fill = None
        self.evaluation_std_fill = None

    def update_plot(self, training_rewards, training_rewards_stds, evaluation_rewards, evaluation_rewards_stds):
        print("Training rewards: ", training_rewards)
        print("Training rewards stds: ", training_rewards_stds)
        print("Evaluation rewards: ", evaluation_rewards)
        print("Evaluation rewards stds: ", evaluation_rewards_stds)

        # Update line data
        self.line1.set_data(range(len(training_rewards)), training_rewards)
        self.line2.set_data(
            [i for i in range(0, len(evaluation_rewards) * self.evaluation_interval, self.evaluation_interval)],
            evaluation_rewards
        )

        # Filter out invalid data for training rewards
        valid_training_data = [(i, r, s) for i, r, s in
                               zip(range(len(training_rewards)), training_rewards, training_rewards_stds) if
                               r is not None and s is not None]
        if valid_training_data:
            x_training, training_rewards_filtered, training_rewards_stds_filtered = zip(*valid_training_data)
            y1_training = [r - s for r, s in zip(training_rewards_filtered, training_rewards_stds_filtered)]
            y2_training = [r + s for r, s in zip(training_rewards_filtered, training_rewards_stds_filtered)]

            if self.training_std_fill:
                self.training_std_fill.remove()
            self.training_std_fill = self.ax.fill_between(x_training, y1_training, y2_training, color='blue', alpha=0.2)

        # Filter out invalid data for evaluation rewards
        valid_evaluation_data = [(i, r, s) for i, r, s in zip(
            [i for i in range(0, len(evaluation_rewards) * self.evaluation_interval, self.evaluation_interval)],
            evaluation_rewards, evaluation_rewards_stds) if r is not None and s is not None]
        if valid_evaluation_data:
            x_evaluation, evaluation_rewards_filtered, evaluation_rewards_stds_filtered = zip(*valid_evaluation_data)
            y1_evaluation = [r - s for r, s in zip(evaluation_rewards_filtered, evaluation_rewards_stds_filtered)]
            y2_evaluation = [r + s for r, s in zip(evaluation_rewards_filtered, evaluation_rewards_stds_filtered)]

            if self.evaluation_std_fill:
                self.evaluation_std_fill.remove()
            self.evaluation_std_fill = self.ax.fill_between(x_evaluation, y1_evaluation, y2_evaluation, color='orange',
                                                            alpha=0.2)

        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)

        if self.l_curve_save_path:
            if os.path.exists(self.l_curve_save_path):
                os.remove(self.l_curve_save_path)
            plt.savefig(self.l_curve_save_path)

    def finalize(self):
        if self.l_curve_save_path:
            if os.path.exists(self.l_curve_save_path):
                os.remove(self.l_curve_save_path)
            plt.savefig(self.l_curve_save_path)
        plt.ioff()
        plt.show()
