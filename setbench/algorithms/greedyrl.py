import hydra
import wandb
import time
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from torch.nn import functional as F
import os

from setbench.algorithms.base import BaseAlgorithm
from setbench.tools.pareto_op import pareto_frontier
from setbench.tools.data_op import str_to_tokens, tokens_to_str
from setbench.metrics import get_all_metrics, get_hv

from torch.distributions import Categorical
from tqdm import tqdm
from functools import partial
from pymoo.indicators.hv import HV

def combine_arrays(arr1, arr2):
    return np.array([np.append(arr1, x) for x in arr2])

class TaskHelper():
    def __init__(self, task, ref_point):
        self.task = task
        self.obj_dim = task.obj_dim
        self.ref_point = ref_point

    def score(self, x):
        return -self.task.score(x)

    def get_batch_cache(self, batch_seqs):
        # For caching batch_seqs' redundant computation
        with torch.inference_mode():
            if len(batch_seqs) == 0:
                batch_seqs_cache = np.empty([0, self.obj_dim])
            else:
                batch_seqs_cache = self.score(batch_seqs)
        return batch_seqs_cache
    
    def score_step(self, batch_seqs, seqs, batch_seqs_cache=None):
        with torch.inference_mode():
            seqs_score = self.score(seqs).reshape(len(seqs), 1, -1) # len(seqs) x 1 x obj_dim
            batch_score = batch_seqs_cache if batch_seqs_cache is not None else self.get_batch_cache(batch_seqs)
            batch_score_repeated = np.tile(batch_score, (seqs_score.shape[0], 1, 1)) # len(seqs) x len(batch_seqs) x obj_dim

            eval_batch_score = np.concatenate([batch_score_repeated, seqs_score], axis=1)
        return torch.tensor(get_hv(torch.tensor(eval_batch_score), hv_ref=self.ref_point))

class SeqBoard:
    def __init__(self, task, obj_dim, max_size, task_max_len):
        self.task = task
        self.obj_dim = obj_dim
        self.max_size = max_size
        self.task_max_len = task_max_len
        self.reset()
    
    def add(self, seq):
        seq = np.array(seq)
        self.cur_hash_vecs = np.concatenate([self.cur_hash_vecs, self.task.score(np.array([seq]))])
        self.cur_reward = self.task.score_step(self.cur_seqs, seq, self.cur_batch_cache).item()
        self.cur_seqs = np.concatenate([self.cur_seqs, seq])
        self.cur_batch_cache = self.cur_hash_vecs 
    
    def reset(self):
        self.cur_hash_vecs = np.empty([0, self.obj_dim])
        self.cur_seqs = np.empty(0)
        self.cur_reward = 0
        self.cur_batch_cache = self.cur_hash_vecs

    def get_set_var(self):
        return torch.tensor(self.cur_hash_vecs).view(-1, self.obj_dim).float()

    def __len__(self):
        return len(self.cur_seqs)
    
    def get_rewards(self, seqs):
        return self.task.score_step(self.cur_seqs, seqs, self.cur_batch_cache).numpy()
    
    def get_loc_seq_board(self):
        return SeqBoard(self.task, self.obj_dim, self.max_size, self.task_max_len)

    def reduce_set(self, k):
        assert k <= len(self)
        if k == len(self):
            return
        if k == 0:
            self.reset()
        elif k == 1:
            seq = self.cur_seqs[0:1]
            self.reset()
            self.add(seq)
        else:
            self.cur_seqs = self.cur_seqs[:k]
            self.cur_hash_vecs = self.cur_hash_vecs[:k]
            self.cur_batch_cache = self.cur_hash_vecs
            self.cur_reward = self.task.score_step(self.cur_seqs[:-1], self.cur_seqs[-1:], self.cur_batch_cache[:-1]).item()

class GreedyRL(BaseAlgorithm):
    def __init__(self, cfg, task, tokenizer, task_cfg, **kwargs):
        super(GreedyRL, self).__init__(cfg, task, tokenizer, task_cfg)
        self.setup_vars(kwargs)
        self.init_policy()

    def setup_vars(self, kwargs):
        cfg = self.cfg
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # Task stuff
        self.max_len = cfg.max_len
        self.min_len = cfg.min_len
        self.obj_dim = self.task.obj_dim

        # Alg stuff
        self.train_steps = cfg.train_steps
        self.random_action_prob = cfg.random_action_prob
        self.batch_size = cfg.batch_size
        self.gen_clip = cfg.gen_clip
        self.always_reinit = cfg.always_reinit

        self.save_samples = getattr(cfg, 'save_samples', False)

        save_name = 'greedyrl'
        os.makedirs(cfg.state_save_path + f'{self.task.task_name}/{save_name}_{self.always_reinit}/', exist_ok=True)
        self.state_save_path = cfg.state_save_path + f'{self.task.task_name}/{save_name}_{self.always_reinit}/{cfg.pi_lr}_{cfg.random_action_prob}_{cfg.max_size}_{kwargs["seed"]}.pkl.gz'

        # Eval Stuff
        self._hv_ref = None
        self._ref_point = np.array([0] * self.obj_dim)
        self.eval_metrics = cfg.eval_metrics
        self.eval_freq = cfg.eval_freq
        self.eos_char = "[SEP]"
        self.pad_tok = self.tokenizer.convert_token_to_id("[PAD]")
        self.max_size = cfg.max_size
        self.num_samples = cfg.num_samples #* self.max_size
        self.use_tqdm = cfg.use_tqdm

        # Adapt model config to task
        self.cfg.model.vocab_size = len(self.tokenizer.full_vocab)
        self.cfg.model.num_actions = len(self.tokenizer.non_special_vocab) + 1

    def init_policy(self):
        cfg = self.cfg
        cond_dim = None
        self.model = hydra.utils.instantiate(cfg.model, cond_dim=cond_dim)

        self.model.to(self.device)

        self.opt = torch.optim.Adam(self.model.model_params(), cfg.pi_lr, weight_decay=cfg.wd,
                            betas=(0.9, 0.999))
        
    def optimize(self, task, init_data=None):
        """
        optimize the task involving multiple objectives (all to be maximized) with 
        optional data to start with
        """
        task_helper = TaskHelper(task, self._ref_point)
        self.seq_board = SeqBoard(task_helper, self.obj_dim, self.max_size, task_max_len=self.max_len)
        losses, rewards = [], []

        train_steps_per_loop = int(np.ceil(self.train_steps / self.max_size))

        if self.max_size == 256:
            self.eval_freq = 2
        elif self.max_size == 64:
            self.eval_freq = 8
        elif self.max_size == 16:
            self.eval_freq = 32
        elif self.max_size == 4:
            self.eval_freq = 125

        for j in range(self.max_size):
            print("stage", j, "/", self.max_size)
            if self.always_reinit:
                self.init_policy()
            
            best_candidate, best_reward = self.evaluation()
            for i in range(train_steps_per_loop):
                loss, r = self.train_step(self.batch_size)

                losses.append(loss)
                rewards.append(r)
                
                if (i != 0 and i % self.eval_freq == self.eval_freq-1) or i == train_steps_per_loop-1:
                    new_candidate, new_reward = self.evaluation()
                    if new_reward > best_reward:
                        best_reward = new_reward
                        best_candidate = new_candidate
            print("best", best_reward)
            self.seq_board.add(best_candidate)
        
        all_rewards = self.seq_board.cur_hash_vecs
        new_candidates = self.seq_board.cur_seqs

        # filter to get current pareto front 
        pareto_candidates, pareto_targets = pareto_frontier(new_candidates, all_rewards, maximize=True)
        
        mo_metrics = get_all_metrics(pareto_targets, self.eval_metrics, hv_ref=self._ref_point, num_obj=self.obj_dim)
        hv = mo_metrics["hypervolume"]
        self.update_state(dict(
            step=0,
            hv=mo_metrics["hypervolume"],
            num_samples=len(new_candidates),
        ))
        self.save_state()

        return {
            'hypervol': hv
        }

    def train_step(self, batch_size):
        states, logprobs = self.sample(episodes=batch_size)
        r = self.process_reward(self.seq_board, states)
        Reward = torch.tensor(r).to(self.device)

        self.opt.zero_grad()
        
        loss = -(logprobs * (Reward-Reward.mean()) / (Reward.std() + 1e-8)).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gen_clip)
        self.opt.step()
        return loss.item(), r.mean()

    def sample(self, episodes, set_var=None, train=True):
        # set_var - (n, set_dim)
        states = [''] * episodes
        traj_logprob = torch.zeros(episodes).to(self.device)
        active_mask = torch.ones(episodes).bool().to(self.device)
        x = str_to_tokens(states, self.tokenizer).to(self.device).t()[:1] # Init with [CLS] tokens shape (1,n)
        lens = torch.zeros(episodes).long().to(self.device)
        uniform_pol = torch.empty(episodes).fill_(self.random_action_prob).to(self.device)

        for t in (range(self.max_len) if episodes > 0 else []):
            logits = self.model(x, set_var, lens=lens, mask=None)
            
            if t <= self.min_len:
                logits[:, 0] = -1000 

            cat = Categorical(logits=logits)
            actions = cat.sample()
            if train and self.random_action_prob > 0:
                uniform_mix = torch.bernoulli(uniform_pol).bool()
                actions = torch.where(uniform_mix, torch.randint(int(t <= self.min_len), logits.shape[1], (episodes, )).to(self.device), actions)
            
            log_prob = cat.log_prob(actions) * active_mask
            traj_logprob += log_prob

            actions_apply = torch.where(torch.logical_not(active_mask), torch.zeros(episodes).to(self.device).long(), actions + 4)
            active_mask = torch.where(active_mask, actions != 0, active_mask)

            x = torch.cat((x, actions_apply.unsqueeze(0)), axis=0)
            if active_mask.sum() == 0:
                break
        states = tokens_to_str(x.t(), self.tokenizer)
        return states, traj_logprob

    def process_reward(self, seq_board, seqs):
        return seq_board.get_rewards(seqs)

    def evaluation(self):
        samples, _ = self.sample(episodes=self.num_samples, train=False) 

        r = self.process_reward(self.seq_board, samples)
        
        # top 1 metrics
        max_idx = r.argmax()

        new_candidate = samples[max_idx:max_idx+1]
        new_reward = r[max_idx]

        return new_candidate, new_reward
