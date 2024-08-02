import numpy as np
import torch

from setbench.algorithms.base import BaseAlgorithm
from setbench.tools.pareto_op import pareto_frontier
from setbench.metrics import get_all_metrics

from setbench.tools.string_op import random_proteins, random_aptamers
from setbench.algorithms.setrl import TaskHelper, SeqBoard

class RandSearch(BaseAlgorithm):
    def __init__(self, cfg, task, tokenizer, task_cfg, **kwargs):
        super(RandSearch, self).__init__(cfg, task, tokenizer, task_cfg)
        self.setup_vars(kwargs)

    def setup_vars(self, kwargs):
        cfg = self.cfg
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # Task stuff
        self.max_len = cfg.max_len
        self.min_len = cfg.min_len
        self.obj_dim = self.task.obj_dim
        # Alg stuff
        self.train_steps = cfg.train_steps

        self.max_size = cfg.max_size 
        self.eval_metrics = cfg.eval_metrics

        # Eval Stuff
        self._hv_ref = None
        self._ref_point = np.array([0] * self.obj_dim)

        import os
        os.makedirs(cfg.state_save_path + f'{self.task.task_name}/randsearch/', exist_ok=True)
        self.state_save_path = cfg.state_save_path + f'{self.task.task_name}/randsearch/{cfg.max_size}_{cfg.train_steps}_{kwargs["seed"]}.pkl.gz'
        print("state_save_path", self.state_save_path)

    def optimize(self, task, init_data=None):
        """
        optimize the task involving multiple objectives (all to be maximized) with 
        optional data to start with
        """

        total_budget = self.train_steps * 128
        budget_per_loop = np.ceil(total_budget / self.max_size)

        task_helper = TaskHelper(task, self._ref_point)
        self.seq_board = SeqBoard(task_helper, self.obj_dim, self.max_size, task_max_len=self.max_len)
        
        steps = 0
        while len(self.seq_board) < self.max_size:
            print("stage", len(self.seq_board), "/", self.max_size)
            if 'Aptamer' in self.tokenizer.__class__.__name__:
                samples = random_aptamers(int(budget_per_loop), self.min_len, self.max_len - 2)
            elif 'Residue' in self.tokenizer.__class__.__name__:
                samples = random_proteins(int(budget_per_loop), self.min_len, self.max_len - 2)
            r = self.process_reward(self.seq_board, samples)
            
            # top 1 metrics
            max_idx = r.argmax()
            self.seq_board.add(samples[max_idx: max_idx+1])
            steps += budget_per_loop
            print("best", r[max_idx])

        all_rewards = self.seq_board.cur_hash_vecs
        new_candidates = self.seq_board.cur_seqs
        pareto_candidates, pareto_targets = pareto_frontier(new_candidates, all_rewards, maximize=True)
        mo_metrics = get_all_metrics(pareto_targets, self.eval_metrics, hv_ref=self._ref_point, num_obj=self.obj_dim)

        self.update_state(dict(
            step=steps,
            hv=mo_metrics["hypervolume"],
            num_samples=len(samples),
        ))
        self.save_state()
        print("hv", mo_metrics["hypervolume"])
        return {
            'hypervol': mo_metrics["hypervolume"],
        }

    def process_reward(self, seq_board, seqs):
        return seq_board.get_rewards(seqs)
