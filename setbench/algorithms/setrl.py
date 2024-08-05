import hydra
import numpy as np
import torch
from torch.nn import functional as F
import os

from setbench.algorithms.base import BaseAlgorithm
from setbench.tools.pareto_op import pareto_frontier
from setbench.tools.data_op import str_to_tokens, tokens_to_str
from setbench.metrics import get_all_metrics, get_hv

from torch.distributions import Categorical
from tqdm import tqdm

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

class SeqBoardBatch:
    def __init__(self, seq_board, batch_size):
        self.seq_boards = []
        for i in range(batch_size):
            self.seq_boards.append(seq_board.get_loc_seq_board())
    
    def add(self, seqs):
        for i in range(len(seqs)):
            self.seq_boards[i].add(seqs[i:i+1])
    
    def reset(self):
        for i in range(len(self.seq_boards)):
            self.seq_boards[i].reset()

    def get_set_var(self):
        return torch.cat([sb.get_set_var().unsqueeze(0) for sb in self.seq_boards], dim=0)

    def reduce_set(self, cardinalities):
        for i, k in enumerate(cardinalities):
            self.seq_boards[i].reduce_set(k)


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

class SetRL(BaseAlgorithm):
    def __init__(self, cfg, task, tokenizer, task_cfg, **kwargs):
        super(SetRL, self).__init__(cfg, task, tokenizer, task_cfg)
        self.setup_vars(kwargs)
        self.init_policy()

    def setup_vars(self, kwargs):
        cfg = self.cfg
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # Task stuff
        self.max_len = cfg.max_len
        self.min_len = cfg.min_len
        self.obj_dim = self.task.obj_dim
        self.train_steps = cfg.train_steps
        self.random_action_prob = cfg.random_action_prob
        self.batch_size = cfg.batch_size
        self.gen_clip = cfg.gen_clip

        self.n_set_samples = cfg.n_set_samples if cfg.n_set_samples > 0 else int(np.ceil(cfg.train_max_size / 16))
        self.train_max_size = cfg.train_max_size

        save_name = 'setrl'
        os.makedirs(cfg.state_save_path + f'{self.task.task_name}/{save_name}/', exist_ok=True)
        self.state_save_path = cfg.state_save_path + f'{self.task.task_name}/{save_name}/{cfg.pi_lr}_{cfg.random_action_prob}_{cfg.max_size}_{cfg.train_max_size}_{cfg.n_set_samples}_{kwargs["seed"]}.pkl.gz'

        # Eval Stuff
        self._hv_ref = None
        self._ref_point = np.array([0] * self.obj_dim)
        self.eval_metrics = cfg.eval_metrics
        self.eval_freq = cfg.eval_freq
        self.num_samples = cfg.num_samples
        self.eos_char = "[SEP]"
        self.pad_tok = self.tokenizer.convert_token_to_id("[PAD]")
        self.max_size = [cfg.max_size] if cfg.max_size>0 else [4, 16, 64, 256]

        if cfg.max_size == -1 and self.train_max_size not in self.max_size:
            self.max_size.append(self.train_max_size)

        # Adapt model config to task
        self.cfg.model.vocab_size = len(self.tokenizer.full_vocab)
        self.cfg.model.num_actions = len(self.tokenizer.non_special_vocab) + 1

    def init_policy(self):
        cfg = self.cfg
        self.set_dim = self.obj_dim
        cond_dim = self.set_dim 
        self.model = hydra.utils.instantiate(cfg.model, cond_dim=cond_dim)

        self.model.to(self.device)

        self.opt = torch.optim.Adam(self.model.model_params(), cfg.pi_lr, weight_decay=cfg.wd,
                            betas=(0.9, 0.999))
        self.opt_Z = torch.optim.Adam(self.model.Z_param(), cfg.z_lr, weight_decay=cfg.wd,
                            betas=(0.9, 0.999))
        
    def state_iter(self, task, step):
        hv_dict = {}

        samples_dict = {}
        all_rews_dict = {}
        for max_size in self.max_size:
            with torch.no_grad():
                samples, all_rews, mo_metrics = self.evaluation(task, max_size, plot=True)
            hv = mo_metrics["hypervolume"]

            hv_dict[max_size] = hv
            samples_dict[max_size] = samples
            all_rews_dict[max_size] = all_rews
            
        self.update_state(dict(
            step=step,
            hv=hv_dict,
        ))
        self.save_state()
        return hv_dict

    def make_desc_str(self, hv_max, hv_dict, losses, rewards):
        desc_str = "Eval. "
        for ms in self.max_size:
            hv = hv_dict[ms]
            hvm = hv_max[ms]
            desc_str += "HV-{}: {:.3f} ({:.3f}) ".format(ms, hv, hvm)
        desc_str += "| Train := Loss: {:.3f} Rewards: {:.3e}".format(sum(losses[-10:]) / 10, sum(rewards[-10:]) / 10)
        return desc_str

    def optimize(self, task, init_data=None):
        """
        optimize the task involving multiple objectives (all to be maximized) with 
        optional data to start with
        """
        task_helper = TaskHelper(task, self._ref_point)
        self.seq_board = SeqBoard(task_helper, self.obj_dim, self.max_size[0], task_max_len=self.max_len)
        losses, rewards = [], []
        pb = tqdm(range(self.train_steps)) 

        hv_dict = self.state_iter(task, 0)
        hv_max = {ms: hv_dict[ms] for ms in self.max_size} 

        pb.set_description(self.make_desc_str(hv_max, hv_dict, losses, rewards))

        for i in pb:
            if i % self.n_set_samples == 0:
                cardinalities = np.random.choice(range(self.train_max_size), size=self.n_set_samples, replace=True)
                seq_board_batch = self.sample_multiple_batch(cardinalities)
            self.seq_board = seq_board_batch.seq_boards[i % self.n_set_samples]
            set_var = self.seq_board.get_set_var()
            loss, r = self.train_step(self.batch_size, set_var)

            losses.append(loss)
            rewards.append(r)
            
            if i != 0 and i % self.eval_freq == self.eval_freq-1:
                hv_dict = self.state_iter(task, i+1)
                for ms in self.max_size:
                    if hv_max[ms] < hv_dict[ms]:
                        hv_max[ms] = hv_dict[ms]
            pb.set_description(self.make_desc_str(hv_max, hv_dict, losses, rewards))
        
        if self.cfg.max_size == -1:
            print("===============================================================")
            for ms in self.max_size:
                print(f"constraint n : {ms}", f"hypervol : {hv_max[ms]}")
            print("===============================================================")
        
        hv_ret = hv_max[self.cfg.max_size] if self.cfg.max_size > 0 else hv_max[self.train_max_size]
        return {
            'losses': losses,
            'train_rs': rewards,
            'hypervol': hv_ret,
        }
    
    def sample_multiple_batch(self, cardinalities):
        max_cardinality = max(cardinalities)
        seq_board_batch = SeqBoardBatch(self.seq_board, len(cardinalities))
        set_var_ = [[0 for _ in range(self.obj_dim)]]
        set_var = torch.tensor(set_var_).view(1, -1).float()
        set_var = torch.tile(set_var.unsqueeze(0), (len(cardinalities), 1, 1)).to(self.device)
        with torch.inference_mode():
            for i in range(max_cardinality):
                states, _ = self.sample(episodes=len(cardinalities), set_var=set_var, train=False)
                seq_board_batch.add(states)
                set_var = seq_board_batch.get_set_var().to(self.device)
        seq_board_batch.reduce_set(cardinalities)
        return seq_board_batch


    def sample_single_batch(self, cardinality):
        set_var_ = [[0 for _ in range(self.obj_dim)]]
        set_var = torch.tensor(set_var_).view(1, -1).float()
        self.seq_board.reset()
        with torch.inference_mode():
            for i in range(cardinality):
                set_var = torch.tile(set_var.unsqueeze(0), (1, 1, 1)).to(self.device)
                state, _ = self.sample(episodes=1, set_var=set_var, train=False)
                self.seq_board.add(state)
                set_var = self.seq_board.get_set_var()
        return set_var

    def train_step(self, batch_size, set_var):
        if len(set_var) == 0:
            set_var_ = [[0 for _ in range(self.obj_dim)]]
            set_var = torch.tensor(set_var_).view(1, -1).float()
        set_var = torch.tile(set_var.unsqueeze(0), (batch_size, 1, 1)).to(self.device)
        states, logprobs = self.sample(episodes=batch_size, set_var=set_var)
        r = self.process_reward(self.seq_board, states)
        Reward = torch.tensor(r).to(self.device)

        self.opt.zero_grad()
        loss = -(logprobs * (Reward-Reward.mean()) / (Reward.std() + 1e-8)).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gen_clip)
        self.opt.step()

        chosen_state = np.random.choice(states, size=1, replace=True)
        self.seq_board.add(chosen_state)
        return loss.item(), r.mean()


    def sample(self, episodes, set_var, train=True):
        states = [''] * episodes
        traj_logprob = torch.zeros(episodes).to(self.device)
        active_mask = torch.ones(episodes).bool().to(self.device)
        x = str_to_tokens(states, self.tokenizer).to(self.device).t()[:1] # Init with [CLS] tokens shape (1,n)
        lens = torch.zeros(episodes).long().to(self.device)
        uniform_pol = torch.empty(episodes).fill_(self.random_action_prob).to(self.device)

        for t in (range(self.max_len) if episodes > 0 else []):
            logits = self.model(x, set_var, lens=lens, mask=None)
            
            if t <= self.min_len:
                logits[:, 0] = -1000 # Prevent model from stopping
                                     # without having output anything

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
    
    def evaluation(self, task, max_size, plot=False):
        loc_seq_board = self.seq_board.get_loc_seq_board()
        set_var_ = [[0 for _ in range(self.obj_dim)]]
        set_var = torch.tensor(set_var_).view(1, -1).float().to(self.device)

        while len(loc_seq_board) < max_size:
            set_var = torch.tile(set_var.unsqueeze(0), (self.num_samples, 1, 1)).to(self.device)

            samples, _ = self.sample(episodes=self.num_samples, set_var=set_var, train=False) 
            r = self.process_reward(loc_seq_board, samples)
            
            # top 1 metrics
            max_idx = r.argmax()
            loc_seq_board.add(samples[max_idx: max_idx+1])

            set_var = loc_seq_board.get_set_var()

        all_rewards = loc_seq_board.cur_hash_vecs
        new_candidates = loc_seq_board.cur_seqs

        # filter to get current pareto front 
        pareto_candidates, pareto_targets = pareto_frontier(new_candidates, all_rewards, maximize=True)
        
        mo_metrics = get_all_metrics(pareto_targets, self.eval_metrics, hv_ref=self._ref_point, num_obj=self.obj_dim)

        return new_candidates, all_rewards, mo_metrics
