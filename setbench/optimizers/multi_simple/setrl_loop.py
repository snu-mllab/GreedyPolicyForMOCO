import hydra
import time
import numpy as np
from regex import P
import torch
import torch.nn.functional as F
import random
import queue

from torch.distributions import Categorical
from setbench.tools.data_op import str_to_tokens, tokens_to_str
from setbench.tools.misc_op import NewTQDM, batched_call
from setbench.models.mlm import get_mlm_synonym_masks
from tqdm import tqdm
from pymoo.indicators.hv import HV

def combine_arrays(arr1, arr2):
    return np.array([np.append(arr1, x) for x in arr2])

class AcqTask():
    def __init__(self, acq_fn, obj_dim):
        self.acq_fn = acq_fn
        self.obj_dim = obj_dim
        self.model = acq_fn.acq_fn.model

        self.beta = acq_fn.acq_fn.beta if acq_fn.tag == 'ucbhvi' else 0.1
        try:
            self.ref_point = acq_fn.acq_fn.ref_point 
        except:
            self.ref_point = acq_fn.ref_point
        self.ref_point_cpu = self.ref_point.cpu().numpy()

    def hash(self, x):
        with torch.inference_mode():
            posterior = self.model.posterior(x)
            mean = posterior.mean
            std = torch.sqrt(posterior.variance)
            ucb_vector = mean + self.beta * std - self.ref_point
        return ucb_vector

    def score(self, x):
        with torch.inference_mode():
            acq_vals = torch.cat(
                batched_call(self.acq_fn.acq_fn, x, batch_size=2)
            ).cpu()
        return acq_vals

    def get_batch_cache(self, batch_seqs):
        # For caching batch_seqs' redundant computation
        with torch.inference_mode():
            if self.acq_fn.tag == 'ucbhvi':
                if len(batch_seqs) == 0:
                    batch_seqs_cache = np.empty([0, self.obj_dim])
                else:
                    batch_seqs_cache = self.hash(batch_seqs).cpu().numpy()
            else:
                # TODO
                batch_seqs_cache = None
        return batch_seqs_cache
    
    def score_step(self, batch_seqs, seqs, batch_seqs_cache=None):
        with torch.inference_mode():
            if self.acq_fn.tag == 'ucbhvi':
                seqs_ucb = self.hash(seqs).view(len(seqs),1,-1) + self.ref_point
                batch_ucb = torch.tensor(batch_seqs_cache).cuda()
                batch_ucb_repeated = torch.tile(batch_ucb + self.ref_point, (seqs_ucb.shape[0], 1, 1))

                eval_batch_ucb = torch.cat([batch_ucb_repeated, seqs_ucb], axis=1)
                hvs = torch.cat(
                        batched_call(self.acq_fn.acq_fn._compute_qehvi, eval_batch_ucb.unsqueeze(0), batch_size=2)
                    ).cpu()
                hvs = torch.tensor(hvs)
                return hvs
            else:
                # TODO implement cached version of nehvi, ehvi
                eval_batch = combine_arrays(batch_seqs, seqs)
                return self.score(eval_batch)


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
        self.cur_hash_vecs = np.concatenate([self.cur_hash_vecs, self.task.hash(seq).cpu().numpy()])
        self.cur_reward = self.task.score_step(self.cur_seqs, seq, self.cur_batch_cache).item()
        self.cur_seqs = np.concatenate([self.cur_seqs, seq])
        self.cur_batch_cache = self.cur_hash_vecs if self.task.acq_fn.tag == 'ucbhvi' else self.task.get_batch_cache(self.cur_seqs)
        # self.cur_batch_cache = self.task.get_batch_cache(self.cur_seqs)
    
    def reset(self):
        self.cur_hash_vecs = np.empty([0, self.obj_dim])
        self.cur_seqs = np.empty(0)
        self.cur_reward = 0
        self.cur_batch_cache = self.cur_hash_vecs if self.task.acq_fn.tag == 'ucbhvi' else self.task.get_batch_cache(self.cur_seqs)
        # self.cur_batch_cache = self.task.get_batch_cache(self.cur_seqs)

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
            self.cur_hash_vecs = self.cur_hash_vecs[:k]
            self.cur_seqs = self.cur_seqs[:k]
            self.cur_batch_cache = self.cur_hash_vecs if self.task.acq_fn.tag == 'ucbhvi' else self.task.get_batch_cache(self.cur_seqs)
            self.cur_reward = self.task.score_step(self.cur_seqs[:-1], self.cur_seqs[-1:], self.cur_batch_cache[:-1]).item()

class SetRLSeqLoop(object):
    def __init__(self, bb_task, tokenizer, num_opt_steps, model, print_op=False, **kwargs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.bb_task = bb_task
        self.tokenizer = tokenizer

        self.batch_size = self.bb_task.batch_size
        self.obj_dim = self.bb_task.obj_dim

        self.num_opt_steps = num_opt_steps
        self.load_alg_params(kwargs, model)

        self.print_op = print_op
        self.mlm_masks = None

    def load_alg_params(self, kwargs, model):
        self.max_num_edits = self.bb_task.max_num_edits 
        self.min_num_edits = kwargs["min_num_edits"] if kwargs["min_num_edits"] else 1
        # Alg stuff
        self.random_action_prob = kwargs["random_action_prob"]
        self.train_batch_size = kwargs["train_batch_size"]
        self.gen_clip = kwargs["gen_clip"]

        self.max_size = self.bb_task.batch_size # Cardinality Constraint
        self.num_initial_states = kwargs["num_initial_states"]

        # We only consider batch_size = max_size = train_max_size = 16, n_set_samples = 1 for active learning scenario.
        self.train_max_size = self.max_size #kwargs.get("train_max_size", min(64, self.max_size))
        self.n_set_samples = self.max_size // 16 #kwargs.get("n_set_samples", 1)
        print("n_set_samples", self.n_set_samples)
        print("train_max_size", self.train_max_size)
        
        self.mlm_top_p = kwargs["mlm_top_p"]
        self.mlm_top_k = kwargs["mlm_top_k"]
        self.use_mlm_logits = kwargs["use_mlm_logits"]

        # Eval stuff
        self._hv_ref = None
        self._ref_point = np.array([0] * self.obj_dim)
        self.eval_freq = kwargs["eval_freq"]
        self.num_samples = kwargs["num_eval_samples"]
        self.eos_char = "[SEP]"
        self.pad_tok = self.tokenizer.convert_token_to_id("[PAD]")
        set_dim = self.obj_dim
        self.cond_dim = set_dim 
        
        self.model_cfg = model
        self.vocab_size = len(self.tokenizer.full_vocab)
        self.num_actions = len(self.tokenizer.non_special_vocab) + 1
        self.pi_lr = kwargs["pi_lr"]
        self.wd = kwargs["wd"]
        self.model = hydra.utils.instantiate(model, num_actions=self.num_actions, vocab_size=self.vocab_size, cond_dim=self.cond_dim)
        self.model.to(self.device)
        self.opt = torch.optim.Adam(params=self.model.model_params(), lr=self.pi_lr, weight_decay=self.wd,
                            betas=(0.9, 0.999))
        print("num_actions", self.num_actions)
        print("vocab_size", self.vocab_size)
    
    def reinit_model(self):
        self.model = hydra.utils.instantiate(self.model_cfg, num_actions=self.num_actions, vocab_size=self.vocab_size, cond_dim=self.cond_dim)
        self.model.to(self.device)
        self.opt = torch.optim.Adam(params=self.model.model_params(), lr=self.pi_lr, weight_decay=self.wd,
                            betas=(0.9, 0.999))

    def optimize(self, surrogate_model, acq_fn, active_candidates, active_targets, mlm_model= None, log_prefix=''):
        self.surrogate_model = surrogate_model
        self.mlm_model = surrogate_model.encoder if mlm_model is None else mlm_model
        self.acq_fn = acq_fn
        self.active_candidates = active_candidates
        self.active_targets = active_targets
        self.start_states = np.array([c.mutant_residue_seq for c in self.active_candidates])
        
        self.mlm_masks, self.mlm_logits = self._get_mlm_synonym_masks()

        task = AcqTask(self.acq_fn, self.obj_dim)
        self.seq_board = SeqBoard(task, self.obj_dim, self.max_size, task_max_len=self.bb_task.max_len)
        
        train_losses, train_rewards = [], []

        with torch.inference_mode():
            best_seqs, acq_val, best_base_candidates  = self.evaluation()
                    
        acq_val_max = acq_val

        pb = tqdm(range(self.num_opt_steps)) if self.print_op else NewTQDM(range(self.num_opt_steps), frequency=self.eval_freq)
        desc_str = "Evaluation := Acq: {:.3f} ({:.3f}) | Train := Loss: {:.3f} Rewards: {:.3f}"
        pb.set_description(desc_str.format(acq_val, acq_val_max, sum(train_losses[-10:]) / 10, sum(train_rewards[-10:]) / 10))

        for i in pb:
            # sample step_idx
            # step_idx = random.choice(range(self.max_size))
            # loss, r = self.train_step(self.train_batch_size, step_idx)
            if i % self.n_set_samples == 0:
                cardinalities = np.random.choice(range(self.train_max_size), size=self.n_set_samples, replace=True)
                seq_board_batch = self.sample_multiple_batch(cardinalities)
            self.seq_board = seq_board_batch.seq_boards[i % self.n_set_samples]
            set_var = self.seq_board.get_set_var()
            loss, r = self.train_step(self.train_batch_size, set_var)


            train_losses.append(loss)
            train_rewards.append(r)

            if i != 0 and i % self.eval_freq == self.eval_freq-1:
                with torch.inference_mode():
                    samples, acq_val, base_candidates = self.evaluation()

                if acq_val_max is None or acq_val_max < acq_val:
                    acq_val_max = acq_val
                    best_seqs = samples
                    best_base_candidates = base_candidates

            pb.set_description(desc_str.format(acq_val, acq_val_max, sum(train_losses[-10:]) / 10, sum(train_rewards[-10:]) / 10))

        metrics = {}
        metrics['best_acq_val'] = self.acq_fn(best_seqs[None, :]).mean().item()
        metrics['best_seqs'] = best_seqs
        metrics['base_candidates'] = best_base_candidates
        print("best_acq_val", metrics['best_acq_val'])
        self.mlm_masks, self.start_states = None, None
        return metrics, (best_base_candidates, best_seqs)

    def _get_mlm_synonym_masks(self):
        return get_mlm_synonym_masks(
                    start_states=self.start_states, 
                    tokenizer=self.tokenizer, 
                    mlm_model=self.mlm_model, 
                    mlm_top_p=self.mlm_top_p, 
                    mlm_top_k=self.mlm_top_k,
                    device=self.device,
                    return_logits=True,
                    )
    
    def sample_multiple_batch(self, cardinalities):
        max_cardinality = max(cardinalities)
        seq_board_batch = SeqBoardBatch(self.seq_board, len(cardinalities))
        set_var_ = [[0 for _ in range(self.obj_dim)]]
        set_var = torch.tensor(set_var_).view(1, -1).float()
        set_var = torch.tile(set_var.unsqueeze(0), (len(cardinalities), 1, 1)).to(self.device)
        with torch.inference_mode():
            for i in range(max_cardinality):
                cur_idxs = np.random.choice(np.arange(len(self.active_candidates)), size=len(cardinalities), replace=True)
                states, _ = self.sample(cur_idxs, episodes=len(cardinalities), set_var=set_var, train=False)
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
                cur_idxs = np.random.choice(np.arange(len(self.active_candidates)), size=1, replace=True)
                state, _ = self.sample(cur_idxs, episodes=1, set_var=set_var, train=False)
                self.seq_board.add(state)
                set_var = self.seq_board.get_set_var()
        return set_var

    # def train_step(self, batch_size, step_idx):
    def train_step(self, batch_size, set_var):
        # set_var = self.sample_single_batch(step_idx)
        # set_var = torch.tile(set_var.unsqueeze(0), (batch_size*self.num_initial_states, 1, 1)).to(self.device)
        if len(set_var) == 0:
            set_var_ = [[0 for _ in range(self.obj_dim)]]
            set_var = torch.tensor(set_var_).view(1, -1).float()
        set_var = torch.tile(set_var.unsqueeze(0), (batch_size, 1, 1)).to(self.device)

        cur_idxs = np.random.choice(np.arange(len(self.active_candidates)), size=self.num_initial_states, replace=True).repeat(batch_size)
        states, logprobs = self.sample(cur_idxs, episodes=batch_size*self.num_initial_states, set_var=set_var)
        r = self.process_reward(self.seq_board, states)
        # normalize by states
        for i in range(self.num_initial_states):
            r[i*batch_size:(i+1)*batch_size] = r[i*batch_size:(i+1)*batch_size] - r[i*batch_size:(i+1)*batch_size].mean()
            r[i*batch_size:(i+1)*batch_size] = r[i*batch_size:(i+1)*batch_size] / (r[i*batch_size:(i+1)*batch_size].std() + 1e-8)
        r = torch.tensor(r).to(self.device)

        self.opt.zero_grad()
        loss = -(logprobs * r).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gen_clip)
        self.opt.step()

        chosen_state = np.random.choice(states, size=1, replace=True)
        self.seq_board.add(chosen_state)
        return loss.item(), r.mean()

    def sample(self, cur_idxs, set_var=None, episodes=16, train=True):
        start_states = self.start_states[cur_idxs]
        states = start_states
        cur_mlm_masks = self.mlm_masks[cur_idxs]
        cur_mlm_logits = self.mlm_logits[cur_idxs]
        traj_logprob = torch.zeros(episodes).to(self.device)
        active_mask = torch.ones(episodes).bool().to(self.device)
        x, lens = str_to_tokens(states, self.tokenizer, return_len=True)
        x = x.to(self.device).t()
        lens = lens.long().to(self.device) - 2
        traj_lens = torch.zeros(episodes).long().to(self.device)
        uniform_pol = torch.zeros(episodes).fill_(self.random_action_prob).to(self.device)
        updated = torch.zeros(episodes).long().to(self.device)
        for t in (range(self.max_num_edits) if episodes > 0 else []):
            pos_logits, tok_logits = self.model(x, set_var, lens=lens, mask=None)
            pos_logits[torch.arange(len(lens)), lens+1] = -1e6 # can't change last token 
            pos_logits[x.t() == 0] = -1e6
            if t > 0:
                pos_logits = pos_logits.scatter(1, updated, -1e6)
            if t < self.min_num_edits:
                pos_logits[:, 0] = -1e6 # Prevent model from stopping

            pos_dist = Categorical(logits=pos_logits)
            pos_actions = pos_dist.sample()

            if train and self.random_action_prob > 0:
                pos_logits_rand = torch.zeros_like(pos_logits.detach())
                pos_logits_rand[:, lens+1] = -1e6 # can't change last token
                pos_logits_rand[x.t() == 0] = -1e6
                if t > 0:
                    pos_logits_rand = pos_logits_rand.scatter(1, updated, -1e6)
                if t < self.min_num_edits:
                    pos_logits_rand[:, 0] = -1e6 # Prevent model from stopping
                pos_dist_rand = Categorical(logits=pos_logits_rand)
                pos_actions_rand = pos_dist_rand.sample()
                uniform_mix = torch.bernoulli(uniform_pol).bool()
                pos_actions = torch.where(uniform_mix, pos_actions_rand, pos_actions)
            if (pos_actions > lens).any():
                raise ValueError("pos action out of bounds")
            tok_logits = tok_logits[torch.arange(tok_logits.shape[0]), pos_actions, :]
            if self.use_mlm_logits:
                with torch.no_grad():
                    tok_logits = tok_logits + cur_mlm_logits[torch.arange(len(pos_actions)), pos_actions]
            tok_logits[torch.arange(tok_logits.shape[0]), x.t()[torch.arange(x.shape[1]), pos_actions] - 4] = -1e6 # block same token
            tok_logits[~cur_mlm_masks[torch.arange(len(pos_actions)), pos_actions]] = -1e6
            tok_dist = Categorical(logits=tok_logits)
            tok_actions = tok_dist.sample()

            if train and self.random_action_prob > 0:
                tok_logits_rand = torch.zeros_like(tok_logits.detach())
                if self.use_mlm_logits:
                    with torch.no_grad():
                        tok_logits_rand = tok_logits_rand + cur_mlm_logits[torch.arange(len(pos_actions)), pos_actions]
                tok_logits_rand[torch.arange(tok_logits.shape[0]), x.t()[torch.arange(x.shape[1]), pos_actions] - 4] = -1e6 # block same token
                tok_logits_rand[~cur_mlm_masks[torch.arange(len(pos_actions)), pos_actions]] = -1e6
                tok_dist_temp = Categorical(logits=tok_logits_rand)
                tok_actions_temp = tok_dist_temp.sample()
                uniform_mix = torch.bernoulli(uniform_pol).bool()
                tok_actions = torch.where(uniform_mix, tok_actions_temp, tok_actions)
            if (tok_dist.log_prob(tok_actions) < -1e6).any():
                raise ValueError("tok action out of bounds")
            if train:
                log_prob = (pos_dist.log_prob(pos_actions) + tok_dist.log_prob(tok_actions)) * active_mask
                
                traj_logprob += log_prob
                traj_lens += torch.where(active_mask, torch.ones_like(lens), torch.zeros_like(lens))

            active_mask = torch.where(active_mask, pos_actions != 0, active_mask)
            tok_actions = torch.where(active_mask, tok_actions+4, x.t()[torch.arange(x.shape[1]), pos_actions])
            tok_actions = torch.where(pos_actions==0, x.t()[torch.arange(x.shape[1]), 0], tok_actions)
            x = x.t().scatter(1, pos_actions.unsqueeze(1), tok_actions.unsqueeze(1)).t()
            if t > 0:
                updated = torch.column_stack((updated, pos_actions))
            else:
                updated = pos_actions.unsqueeze(1)
            if active_mask.sum() == 0:
                break
        states = tokens_to_str(x.t(), self.tokenizer)
        if [len(self.tokenizer.encode(s)) for s in start_states] != [len(self.tokenizer.encode(s)) for s in states]:
            raise ValueError("lengths don't match")
        return states, traj_logprob   

    def process_reward(self, seq_board, seqs):
        return seq_board.get_rewards(seqs)

    def evaluation(self):
        loc_seq_board = self.seq_board.get_loc_seq_board()
        set_var_ = [[0 for _ in range(self.obj_dim)]]
        set_var = torch.tensor(set_var_).view(1, -1).float().to(self.device)
        base_candidates = []
        while len(loc_seq_board) < self.max_size:
            set_var = torch.tile(set_var.unsqueeze(0), (self.num_samples, 1, 1)).to(self.device)

            cur_idxs = np.random.choice(np.arange(len(self.active_candidates)), size=self.num_samples, replace=True)

            samples, _ = self.sample(cur_idxs, episodes=self.num_samples, set_var=set_var, train=False) 
            r = self.process_reward(loc_seq_board, samples)
            
            # top 1 metrics
            max_idx = r.argmax()
            loc_seq_board.add(samples[max_idx: max_idx+1])
            base_candidates.append(self.active_candidates[cur_idxs[max_idx]])

            set_var = loc_seq_board.get_set_var()

        new_seqs = loc_seq_board.cur_seqs
        acq_val = loc_seq_board.cur_reward
        return new_seqs, acq_val, np.array(base_candidates)