import hydra
import wandb
import numpy as np
import torch
import random

from setbench.algorithms.base import BaseAlgorithm
from setbench.tools.pareto_op import thermometer, pareto_frontier
from setbench.tools.metric_op import mean_pairwise_distances
from setbench.tools.sample_op import generate_simplex
from setbench.tools.data_op import str_to_tokens, tokens_to_str
from setbench.tools.misc_op import NewTQDM
from setbench.metrics import get_all_metrics

from torch.distributions import Categorical
from tqdm import tqdm



class MOReinforce(BaseAlgorithm):
    def __init__(self, cfg, task, tokenizer, task_cfg, **kwargs):
        super(MOReinforce, self).__init__(cfg, task, tokenizer, task_cfg)
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
        self.max_size = [cfg.max_size] if cfg.max_size>0 else [4, 16, 64, 256]
        self.reward_min = cfg.reward_min
        self.therm_n_bins = cfg.therm_n_bins
        self.beta_use_therm = cfg.beta_use_therm
        self.pref_use_therm = cfg.pref_use_therm
        self.gen_clip = cfg.gen_clip
        self.sample_beta = cfg.sample_beta
        self.beta_cond = cfg.beta_cond
        self.pref_cond = cfg.pref_cond
        self.beta_scale = cfg.beta_scale
        self.beta_shape = cfg.beta_shape
        self.pref_alpha = cfg.pref_alpha
        self.beta_max = cfg.beta_max
        self.reward_type = cfg.reward_type
        self.loss_type = cfg.loss_type
        import os
        os.makedirs(cfg.state_save_path + f'{self.task.task_name}/realmorl_{self.loss_type}/', exist_ok=True)
        self.state_save_path = cfg.state_save_path + f'{self.task.task_name}/realmorl_{self.loss_type}/{cfg.pi_lr}_{cfg.random_action_prob}_{cfg.reward_type}_{cfg.max_size}_{kwargs["seed"]}.pkl.gz'
        print("state_save_path", self.state_save_path)
        self.pareto_freq = cfg.pareto_freq
        self.num_pareto_points = cfg.num_pareto_points

        # Eval Stuff
        self._hv_ref = None
        self._ref_point = np.array([0] * self.obj_dim)
        self.eval_metrics = cfg.eval_metrics
        self.eval_freq = cfg.eval_freq
        self.k = cfg.k
        self.num_samples = cfg.num_samples
        self.eos_char = "[SEP]"
        self.pad_tok = self.tokenizer.convert_token_to_id("[PAD]")
        self.simplex = generate_simplex(self.obj_dim, cfg.simplex_bins)
        self.use_tqdm = cfg.use_tqdm

        # Adapt model config to task
        self.cfg.model.vocab_size = len(self.tokenizer.full_vocab)
        self.cfg.model.num_actions = len(self.tokenizer.non_special_vocab) + 1

    def init_policy(self):
        cfg = self.cfg
        pref_dim = self.therm_n_bins * self.obj_dim if self.pref_use_therm else self.obj_dim
        beta_dim = self.therm_n_bins if self.beta_use_therm else 1
        cond_dim = pref_dim + beta_dim if self.beta_cond else pref_dim
        assert (self.beta_cond or self.pref_cond) == cfg.model.use_cond, "Model config and algorithm config do not match"
        assert (self.beta_cond or self.pref_cond) == cfg.model.update_cond, "Model config and algorithm config do not match"

        self.model = hydra.utils.instantiate(cfg.model, cond_dim=cond_dim)

        self.model.to(self.device)
        self.opt = torch.optim.Adam(self.model.model_params(), cfg.pi_lr, weight_decay=cfg.wd,
                            betas=(0.9, 0.999))

    def state_iter(self, task, step):
        hv_dict = {}
        for max_size in self.max_size:
            with torch.no_grad():
                samples, all_rews, rs, mo_metrics = self.evaluation(task, max_size, plot=True)
            hv, r2, hsri = mo_metrics["hypervolume"], mo_metrics["r2"], mo_metrics["hsri"]

            hv_dict[max_size] = hv

        self.update_state(dict(
            step=step,
            hv=hv_dict,
        ))
        self.save_state()

        return hv

    def optimize(self, task, init_data=None):
        """
        optimize the task involving multiple objectives (all to be maximized) with 
        optional data to start with
        """
        losses, rewards = [], []
        hv, r2, hsri, rs = 0., 0., 0., np.zeros(self.obj_dim)

        pb = tqdm(range(self.train_steps)) 
        desc_str = "Evaluation := HV: {:.3f} ({:.3f}) | Train := Loss: {:.3f} Rewards: {:.3f}"
        pb.set_description(desc_str.format(hv, hv, sum(losses[-10:]) / 10, sum(rewards[-10:]) / 10))

        hv = self.state_iter(task, 0)
        hv_max = hv
        
        for i in pb:
            loss, r = self.train_step(task, self.batch_size)
            losses.append(loss)
            rewards.append(r)

            if (i != 0 and i % self.eval_freq == self.eval_freq-1) or (i == self.train_steps-1):
                hv = self.state_iter(task, i+1)
                if hv > hv_max:
                    hv_max = hv
            self.log(dict(
                train_loss=loss,
                train_rewards=r,
            ))
            pb.set_description(desc_str.format(hv, hv_max, sum(losses[-10:]) / 10, sum(rewards[-10:]) / 10))


        return {
            'hypervol': hv
        }

    def train_step(self, task, batch_size):
        cond_var, (prefs, beta) = self._get_condition_var(train=True, bs=batch_size)
        states, logprobs = self.sample(batch_size, cond_var)

        r = self.process_reward(states, prefs, task).to(self.device)
        
        self.opt.zero_grad()
        
        # REINFORCE Loss
        if self.loss_type=='rl':
            loss = -(logprobs * (r - r.mean())).mean()
        elif self.loss_type=='rl_norm':
            loss = -(logprobs * (r - r.mean()) / (r.std() + 1e-8)).mean()
        else:
            raise NotImplementedError
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gen_clip)
        self.opt.step()
        return loss.item(), r.mean()


    def sample(self, episodes, cond_var=None, train=True):
        states = [''] * episodes
        traj_logprob = torch.zeros(episodes).to(self.device)
        if cond_var is None:
            cond_var, _ = self._get_condition_var(train=train, bs=episodes)
        active_mask = torch.ones(episodes).bool().to(self.device)
        x = str_to_tokens(states, self.tokenizer).to(self.device).t()[:1]
        lens = torch.zeros(episodes).long().to(self.device)
        uniform_pol = torch.empty(episodes).fill_(self.random_action_prob).to(self.device)

        for t in (range(self.max_len) if episodes > 0 else []):
            logits = self.model(x, cond_var, lens=lens, mask=None)
            
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
    

    def process_reward(self, seqs, prefs, task, rewards=None):
        if rewards is None:
            rewards = -task.score(seqs)
            # rewards = ((rewards - 0) * 2) + -1 # shape rewards to improve learning
        if self.reward_type == "convex":
            r = (torch.tensor(prefs) * (rewards)).sum(axis=1)
        elif self.reward_type == "logconvex":
            r = (torch.tensor(prefs) * torch.tensor(rewards).clamp(min=self.reward_min).log()).sum(axis=1).exp()
        elif self.reward_type == "tchebycheff":
            r = -(torch.tensor(prefs) * torch.abs(1 - torch.tensor(rewards))).max(axis=1)[0]
        return r

    def evaluation(self, task, max_size, plot=False):
        new_candidates = []
        r_scores = [] 
        all_rewards = []

        idxs_to_sample = random.sample(range(len(self.simplex)), max_size)
        prefs_to_sample = self.simplex[idxs_to_sample]
        for prefs in prefs_to_sample:
        # for prefs in self.simplex:
            cond_var, (_, beta) = self._get_condition_var(prefs=prefs, train=False, bs=self.num_samples)
            samples, _ = self.sample(self.num_samples, cond_var, train=False)
            rewards = -task.score(samples)
            r = self.process_reward(samples, prefs, task, rewards=rewards)
            
            # topk metrics
            topk_r, topk_idx = torch.topk(r, self.k)
            samples = np.array(samples)
            topk_seq = samples[topk_idx].tolist()
            edit_dist = mean_pairwise_distances(topk_seq)
            
            # top 1 metrics
            max_idx = r.argmax()
            new_candidates.append(samples[max_idx])
            all_rewards.append(rewards[max_idx])
            r_scores.append((torch.tensor(prefs) * (rewards[max_idx])).sum())

        r_scores = np.array(r_scores)
        all_rewards = np.array(all_rewards)
        new_candidates = np.array(new_candidates)
    
        pareto_candidates, pareto_targets = pareto_frontier(new_candidates, all_rewards, maximize=True)
        
        mo_metrics = get_all_metrics(pareto_targets, self.eval_metrics, hv_ref=self._ref_point, r2_prefs=self.simplex, num_obj=self.obj_dim)

        return new_candidates, all_rewards, r_scores, mo_metrics
    
    def _get_condition_var(self, prefs=None, beta=None, train=True, bs=None):
        if prefs is None:
            if not train:
                prefs = self.simplex[0]
            else:
                prefs = np.random.dirichlet([self.pref_alpha]*self.obj_dim)
        if beta is None:
            if train:
                beta = float(np.random.randint(1, self.beta_max+1)) if self.beta_cond else self.sample_beta
            else:
                beta = self.sample_beta

        if self.pref_use_therm:
            prefs_enc = thermometer(torch.from_numpy(prefs), self.therm_n_bins, 0, 1) 
        else: 
            prefs_enc = torch.from_numpy(prefs)
        
        if self.beta_use_therm:
            beta_enc = thermometer(torch.from_numpy(np.array([beta])), self.therm_n_bins, 0, self.beta_max) 
        else:
            beta_enc = torch.from_numpy(np.array([beta]))
        if self.beta_cond:
            cond_var = torch.cat((prefs_enc.view(-1), beta_enc.view(-1))).float().to(self.device)
        else:
            cond_var = prefs_enc.view(-1).float().to(self.device)
        if bs:
            cond_var = torch.tile(cond_var.unsqueeze(0), (bs, 1))
        return cond_var, (prefs, beta)