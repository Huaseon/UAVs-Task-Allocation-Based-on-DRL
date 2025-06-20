# %%
import numpy as np
import time, datetime
from visdom import Visdom

# %%
from . import SEED
np.random.seed(SEED)

class AgentLogger:
    def __init__(self, save_dir, name, top):
        self.save_log = save_dir / f"log_{top}"
        self.name = name
        self.top = top
        self.record_every = (top + 99) // 100

        with open(self.save_log, "w") as f:
            f.write(
                f"{'Episode':>8}{'Step':>8}{'Epsilon':>10}{'MeanReward':>15}"
                f"{'MeanLength':>15}{'MeanLoss':>15}{'MeanQValue':>15}"
                f"{'TimeDelta':>15}{'Time':>20}{'ExpectTime':>20}\n"
            )

        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []
        self.ep_avg_qs = []

        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_lengths = []
        self.moving_avg_ep_avg_losses = []
        self.moving_avg_ep_avg_qs = []

        self.init_episode()

        self.record_time = time.time()

        self.wind = Visdom(env=self.name)
        for win in ['length_plot', 'q_plot']:
            self.wind.win_exists(win) or self.wind.line(
                [0], [0], win=win, opts=dict(title=win, legend=[f'{self.name}']), name=f'{self.name}', update='append'
            )

    def log_step(self, reward, loss, q):
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        if loss:
            self.curr_ep_loss += loss
            self.curr_ep_q += q
            self.curr_ep_loss_length += 1
    
    def log_episode(self):
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)
        if self.curr_ep_loss_length == 0:
            ep_avg_loss = 0
            ep_avg_q = 0
        else:
            ep_avg_loss = np.round(self.curr_ep_loss / self.curr_ep_loss_length, 5)
            ep_avg_q = np.round(self.curr_ep_q / self.curr_ep_loss_length, 5)
            self.ep_avg_losses.append(ep_avg_loss)
            self.ep_avg_qs.append(ep_avg_q)

        self.init_episode()
    
    def init_episode(self):
        self.curr_ep_reward = .0
        self.curr_ep_length = 0
        self.curr_ep_loss = .0
        self.curr_ep_q = .0
        self.curr_ep_loss_length = 0
    
    def record(self, episode, epsilon, step):
        mean_ep_reward = np.round(np.mean(self.ep_rewards[-100:] or [0]), 3)
        mean_ep_length = np.round(np.mean(self.ep_lengths[-100:] or [0]), 3)
        mean_ep_loss = np.round(np.mean(self.ep_avg_losses[-100:] or [0]), 3)
        mean_ep_q = np.round(np.mean(self.ep_avg_qs[-100:] or [0]), 3)

        self.moving_avg_ep_rewards.append(mean_ep_reward)
        self.moving_avg_ep_lengths.append(mean_ep_length)
        self.moving_avg_ep_avg_losses.append(mean_ep_loss)
        self.moving_avg_ep_avg_qs.append(mean_ep_q)

        d = [(self.ep_avg_qs[-1:] or [0], mean_ep_q, 'q_plot'), (self.ep_lengths[-1:] or [0], mean_ep_length, 'length_plot')]
        
        for y, m, win in d:
            self.wind.line(
                [m], [episode], win=win, name=f'{self.name}', update='append'
            )

        if episode + 1 == self.top or episode % self.record_every == 0:
            last_record_time = self.record_time
            self.record_time = time.time()
            time_since_last_record = np.round(self.record_time - last_record_time, 3)
            with open(self.save_log, 'a') as f:
                f.write(
                    f"{episode:8d}{step:8d}{epsilon:10.3f}"
                    f"{mean_ep_reward:15.3f}{mean_ep_length:15.3f}{mean_ep_loss:15.3f}{mean_ep_q:15.3f}"
                    f"{time_since_last_record:15.3f}"
                    f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}{(datetime.datetime.now() + datetime.timedelta(seconds=time_since_last_record / self.record_every * (self.top - 1 - episode))).strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
                )

class TestLogger:
    def __init__(self, save_dir, top, name):
        self.save_log = save_dir / f"test-log_{top}"
        self.top = top
        self.name = name
        self.record_every = (top + 99) // 100

        with open(self.save_log, "w") as f:
            f.write(
                f"{'Episode':>8}{'MeanReward':>15}"
                f"{'MeanLength':>15}{'MeanQValue':>15}"
                f"{'TimeDelta':>15}{'Time':>20}{'ExpectTime':>20}"
                f"{'MeanPrecision':>20}{'MeanPayloadLimit':>20}"
                f"{'MeanTemporalLimit':>20}{'MeanOptimization':>20}\n"
            )

        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_qs = []
        self.ep_avg_rs = []

        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_lengths = []
        self.moving_avg_ep_avg_qs = []
        self.moving_avg_ep_avg_rs = []

        self.init_episode()

        self.record_time = time.time()

        self.wind = Visdom(env=self.name)
        for win in ['reward_plot', 'q_plot']:
            self.wind.win_exists(win) or self.wind.line(
                [0], [0], win=win, opts=dict(title=win, legend=[f'{self.name}']), name=f'{self.name}', update='append'
            )
        

        self.wind.win_exists('r_plot') or self.wind.line(
                [0], [0], win='r_plot', opts=dict(title='r_plot', legend=['precision_avg']), name=f'precision_avg', update='append'
        )

    def log_step(self, reward, q, r):
        self.curr_ep_reward += reward
        self.curr_ep_q += q
        self.curr_ep_length += 1
        self.curr_ep_r = r
    
    def log_episode(self):
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)
        self.ep_avg_qs.append(np.round(self.curr_ep_q / self.curr_ep_length, 5))
        self.ep_avg_rs.append(self.curr_ep_r)
        self.init_episode()
    
    def init_episode(self):
        self.curr_ep_reward = .0
        self.curr_ep_length = 0
        self.curr_ep_q = .0
        self.curr_ep_r = np.zeros(shape=(4,))
    
    def record(self, episode):
        mean_ep_reward = np.round(np.mean(self.ep_rewards[-100:] or [0]), 3)
        mean_ep_length = np.round(np.mean(self.ep_lengths[-100:] or [0]), 3)
        mean_ep_q = np.round(np.mean(self.ep_avg_qs[-100:] or [0]), 3)
        mean_ep_r = np.round(np.mean(self.ep_avg_rs[-100:] or [[0] * 4], axis=0), 3)

        self.moving_avg_ep_rewards.append(mean_ep_reward)
        self.moving_avg_ep_lengths.append(mean_ep_length)
        self.moving_avg_ep_avg_qs.append(mean_ep_q)
        self.moving_avg_ep_avg_rs.append(mean_ep_r)

        d = [(self.ep_rewards[-1:] or [0], mean_ep_reward, 'reward_plot') , (self.ep_avg_qs[-1:] or [0], mean_ep_q, 'q_plot')]
        
        for y, m, win in d:
            self.wind.line(
                [m], [episode], win=win, name=f'{self.name}', update='append'
            )
        for y, m, name_part in zip(self.ep_avg_rs[-1], mean_ep_r, ['precision', 'payload', 'temporal', 'optimization']):
            self.wind.line(
                [m], [episode], win='r_plot', name=f'{name_part}_avg', update='append'
            )

        if episode + 1 == self.top or episode % self.record_every == 0:
            last_record_time = self.record_time
            self.record_time = time.time()
            time_since_last_record = np.round(self.record_time - last_record_time, 3)
            with open(self.save_log, 'a') as f:
                f.write(
                    f"{episode:8d}{mean_ep_reward:15.3f}"
                    f"{mean_ep_length:15.3f}{mean_ep_q:15.3f}"
                    f"{time_since_last_record:15.3f}"
                    f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}{(datetime.datetime.now() + datetime.timedelta(seconds=time_since_last_record / self.record_every * (self.top - 1 - episode))).strftime('%Y-%m-%dT%H:%M:%S'):>20}"
                    f"{mean_ep_r[0]:>20}{mean_ep_r[1]:>20}"
                    f"{mean_ep_r[2]:>20}{mean_ep_r[3]:>20}\n"
                )

# class HyperLogger:
#     def __init__(self, save_dir, name, top):
#         self.save_log = save_dir / f"log_{top}"
#         self.name = name
#         self.top = top
#         self.record_every = (top + 99) // 100

#         with open(self.save_log, "w") as f:
#             f.write(
#                 f"{'Episode':>8}{'Step':>8}{'Epsilon':>10}{'MeanReward':>15}"
#                 f"{'MeanLength':>15}{'MeanLoss':>15}{'MeanQValue':>15}"
#                 f"{'TimeDelta':>15}{'Time':>20}{'ExpectTime':>20}\n"
#             )

#         self.ep_rewards = []
#         self.ep_lengths = []
#         self.ep_avg_losses = []
#         self.ep_avg_qs = []

#         self.moving_avg_ep_rewards = []
#         self.moving_avg_ep_lengths = []
#         self.moving_avg_ep_avg_losses = []
#         self.moving_avg_ep_avg_qs = []

#         self.init_episode()

#         self.record_time = time.time()

#         self.wind = Visdom(env=self.name)
#         for win in ['reward_plot', 'length_plot', 'loss_plot', 'q_plot']:
#             self.wind.win_exists(win) or self.wind.line(
#                 [0], [0], win=win, opts=dict(title=win, legend=['realtime']), name=f'realtime', update='append'
#             )

#     def log_step(self, reward, loss, q):
#         self.curr_ep_reward += reward
#         self.curr_ep_length += 1
#         if loss:
#             self.curr_ep_loss += loss
#             self.curr_ep_q += q
#             self.curr_ep_loss_length += 1
    
#     def log_episode(self):
#         self.ep_rewards.append(self.curr_ep_reward)
#         self.ep_lengths.append(self.curr_ep_length)
#         if self.curr_ep_loss_length == 0:
#             ep_avg_loss = 0
#             ep_avg_q = 0
#         else:
#             ep_avg_loss = np.round(self.curr_ep_loss / self.curr_ep_loss_length, 5)
#             ep_avg_q = np.round(self.curr_ep_q / self.curr_ep_loss_length, 5)
#             self.ep_avg_losses.append(ep_avg_loss)
#             self.ep_avg_qs.append(ep_avg_q)

#         self.init_episode()
    
#     def init_episode(self):
#         self.curr_ep_reward = .0
#         self.curr_ep_length = 0
#         self.curr_ep_loss = .0
#         self.curr_ep_q = .0
#         self.curr_ep_loss_length = 0
    
#     def record(self, episode, epsilon, step):
#         mean_ep_reward = np.round(np.mean(self.ep_rewards[-100:] or [0]), 3)
#         mean_ep_length = np.round(np.mean(self.ep_lengths[-100:] or [0]), 3)
#         mean_ep_loss = np.round(np.mean(self.ep_avg_losses[-100:] or [0]), 3)
#         mean_ep_q = np.round(np.mean(self.ep_avg_qs[-100:] or [0]), 3)

#         self.moving_avg_ep_rewards.append(mean_ep_reward)
#         self.moving_avg_ep_lengths.append(mean_ep_length)
#         self.moving_avg_ep_avg_losses.append(mean_ep_loss)
#         self.moving_avg_ep_avg_qs.append(mean_ep_q)

#         d = [(self.ep_rewards[-1:] or [0], mean_ep_reward, 'reward_plot') , (self.ep_avg_qs[-1:] or [0], mean_ep_q, 'q_plot'), (self.ep_lengths[-1:] or [0], mean_ep_length, 'length_plot'), (self.ep_avg_losses[-1:] or [0], mean_ep_loss, 'loss_plot')]
        
#         for y, m, win in d:
#             self.wind.line(
#                 y, [episode], win=win, name=f'realtime', update='append'
#             )
#             self.wind.line(
#                 [m], [episode], win=win, name=f'avg', update='append'
#             )

#         if episode + 1 == self.top or episode % self.record_every == 0:
#             last_record_time = self.record_time
#             self.record_time = time.time()
#             time_since_last_record = np.round(self.record_time - last_record_time, 3)
#             with open(self.save_log, 'a') as f:
#                 f.write(
#                     f"{episode:8d}{step:8d}{epsilon:10.3f}"
#                     f"{mean_ep_reward:15.3f}{mean_ep_length:15.3f}{mean_ep_loss:15.3f}{mean_ep_q:15.3f}"
#                     f"{time_since_last_record:15.3f}"
#                     f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}{(datetime.datetime.now() + datetime.timedelta(seconds=time_since_last_record / self.record_every * (self.top - 1 - episode))).strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
#                 )
