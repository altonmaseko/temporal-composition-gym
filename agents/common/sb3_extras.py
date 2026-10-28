"""SB3 additions for protocol v3 logging.

LoggedDQN is stable-baselines3 2.9.0's DQN with an identical training step that additionally
records the mean Q-value of the taken actions (train/q_mean) and the pre-clipping gradient norm
(train/grad_norm), which SB3 computes but discards. The update itself is unchanged: same
sampling, targets, loss, clipping and optimiser step (checked by tests/test_protocol_v3.py).
"""
import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3 import DQN


class LoggedDQN(DQN):
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses, q_means, grad_norms = [], [], []
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            with th.no_grad():
                next_q_values = self.q_net_target(replay_data.next_observations)
                next_q_values, _ = next_q_values.max(dim=1)
                next_q_values = next_q_values.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            current_q_values = self.q_net(replay_data.observations)
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())
            q_means.append(current_q_values.detach().mean().item())

            self.policy.optimizer.zero_grad()
            loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            grad_norms.append(float(grad_norm))
            self.policy.optimizer.step()

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
        self.logger.record("train/q_mean", np.mean(q_means))
        self.logger.record("train/grad_norm", np.mean(grad_norms))
