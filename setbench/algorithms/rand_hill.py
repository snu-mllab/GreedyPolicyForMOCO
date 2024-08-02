import numpy as np
import torch

from setbench.algorithms.base import BaseAlgorithm
from setbench.tools.pareto_op import pareto_frontier
from setbench.metrics import get_all_metrics

from setbench.tools.string_op import random_proteins, random_aptamers
from setbench.algorithms.setrl import TaskHelper, SeqBoard

from setbench.tools.string_op import AMINO_ACIDS, APTAMER_BASES

class RandHill(BaseAlgorithm):
    def __init__(self, cfg, task, tokenizer, task_cfg, **kwargs):
        super(RandHill, self).__init__(cfg, task, tokenizer, task_cfg)
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
        os.makedirs(cfg.state_save_path + f'{self.task.task_name}/randhill/', exist_ok=True)
        self.state_save_path = cfg.state_save_path + f'{self.task.task_name}/randhill/{cfg.max_size}_{cfg.train_steps}_{kwargs["seed"]}.pkl.gz'
        print("state_save_path", self.state_save_path)

    def get_neighbor(self, seq):
        """
        sample a neighbor of the given sequence
        """
        if 'Aptamer' in self.tokenizer.__class__.__name__:
            alphabet = APTAMER_BASES
        elif 'Residue' in self.tokenizer.__class__.__name__:
            alphabet = AMINO_ACIDS
        nbd_seqs = []
        for nbd_idx in range(len(seq)):
            for char_idx in range(len(alphabet)):
                nbd_seq = seq[:nbd_idx] + alphabet[char_idx] + seq[nbd_idx+1:]
                nbd_seqs.append(nbd_seq)
        return nbd_seqs

    def optimize(self, task, init_data=None):
        """
        optimize the task involving multiple objectives (all to be maximized) with 
        optional data to start with
        """

        total_budget = self.train_steps * 128
        budget_per_loop = int(np.ceil(total_budget / self.max_size))

        task_helper = TaskHelper(task, self._ref_point)
        self.seq_board = SeqBoard(task_helper, self.obj_dim, self.max_size, task_max_len=self.max_len)
        
        steps = 0

        while len(self.seq_board) < self.max_size:
            print("stage", len(self.seq_board), "/", self.max_size)
            if 'Aptamer' in self.tokenizer.__class__.__name__:
                cur_sample = str(random_aptamers(1, self.max_len-2, self.max_len-2)[0])
            elif 'Residue' in self.tokenizer.__class__.__name__:
                cur_sample = str(random_proteins(1, self.max_len-2, self.max_len-2)[0])
            samples = np.array([cur_sample])
            rs = self.process_reward(self.seq_board, samples).tolist()
            
            cur_best = rs[0]
            while len(samples) < budget_per_loop:
                new_samples = np.array(self.get_neighbor(cur_sample))
                n_samples = min(len(new_samples), budget_per_loop - len(samples))
                new_samples = np.random.choice(new_samples, n_samples, replace=False)
                new_rs = self.process_reward(self.seq_board, new_samples).tolist()
                samples = np.concatenate([samples, new_samples])
                rs.extend(new_rs)

                best_ind = np.argmax(rs)
                cur_sample = str(samples[best_ind])

                assert len(rs) == len(samples)

                if rs[best_ind] > cur_best:
                    cur_best = rs[best_ind]
                else:
                    cur_sample = str(random_aptamers(1, self.max_len-2, self.max_len-2)[0])
                                    
            # top 1 metrics
            max_idx = np.argmax(rs)
            samples = np.array(samples)
            self.seq_board.add(samples[max_idx: max_idx+1])
            steps += budget_per_loop
            print("best", rs[max_idx])

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
