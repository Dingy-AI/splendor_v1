import matplotlib.pyplot as plt


class TrainingPlot:

    def __init__(self):
        plt.ion()

        self.games = []

        self.win_rates = []
        self.total_losses = []
        self.policy_losses = []
        self.value_losses = []
        self.policy_kls = []
        self.replay_sizes = []
        self.avg_game_lengths = []

        self.fig, self.axes = plt.subplots(
            3,
            2,
            figsize=(12, 10),
        )

        self.fig.tight_layout()

    def update(
        self,
        games_played,
        win_rate=None,
        total_loss=None,
        policy_loss=None,
        value_loss=None,
        policy_kl=None,
        replay_size=None,
        avg_game_length=None,
    ):
        self.games.append(games_played)

        self.win_rates.append(win_rate)
        self.total_losses.append(total_loss)
        self.policy_losses.append(policy_loss)
        self.value_losses.append(value_loss)
        self.policy_kls.append(policy_kl)
        self.replay_sizes.append(replay_size)
        self.avg_game_lengths.append(avg_game_length)

        self._draw()

    def _draw(self):

        axes = self.axes.flatten()

        for ax in axes:
            ax.clear()

        self._plot_metric(
            axes[0],
            self.win_rates,
            "Win Rate vs Random",
            "Win Rate",
        )

        axes[0].set_ylim(0, 1)

        self._plot_losses(
            axes[1]
        )

        self._plot_metric(
            axes[2],
            self.policy_kls,
            "Policy KL",
            "KL Divergence",
        )

        self._plot_metric(
            axes[3],
            self.replay_sizes,
            "Replay Buffer Size",
            "Positions",
        )

        self._plot_metric(
            axes[4],
            self.avg_game_lengths,
            "Average Evaluation Game Length",
            "Steps",
        )

        axes[5].axis("off")

        self.fig.tight_layout()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        plt.pause(0.01)

    def _plot_metric(
        self,
        ax,
        values,
        title,
        ylabel,
    ):
        x, y = self._valid_points(values)

        ax.plot(
            x,
            y,
            marker="o",
        )

        ax.set_title(title)
        ax.set_xlabel("Games Played")
        ax.set_ylabel(ylabel)
        ax.grid(True)

    def _plot_losses(
        self,
        ax,
    ):
        total_x, total_y = self._valid_points(
            self.total_losses
        )

        policy_x, policy_y = self._valid_points(
            self.policy_losses
        )

        value_x, value_y = self._valid_points(
            self.value_losses
        )

        ax.plot(
            total_x,
            total_y,
            label="Total",
        )

        ax.plot(
            policy_x,
            policy_y,
            label="Policy",
        )

        ax.plot(
            value_x,
            value_y,
            label="Value",
        )

        ax.set_title("Training Loss")
        ax.set_xlabel("Games Played")
        ax.set_ylabel("Loss")

        ax.grid(True)
        ax.legend()

    def _valid_points(
        self,
        values,
    ):
        x = []
        y = []

        for games_played, value in zip(
            self.games,
            values,
        ):
            if value is not None:
                x.append(games_played)
                y.append(value)

        return x, y

    def save(
        self,
        path="checkpoints/training_progress.png",
    ):
        self.fig.savefig(path)