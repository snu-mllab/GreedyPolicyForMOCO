import time
import numpy as np
from sklearn import base
import torch
import random

from setbench.models.mlm import sample_tokens, evaluate_windows_simple
from setbench.models.lanmt import corrupt_tok_idxs
from setbench.tools.data_op import DataSplit, str_to_tokens, tokens_to_str
from setbench.tools.sample_op import weighted_resampling
from setbench.tools.pareto_op import pareto_frontier
class LaMBOLoop(object):
    def __init__(self, bb_task, tokenizer, encoder, num_gens, lr, num_opt_steps, patience, resampling_weight, encoder_obj,
                 optimize_latent, position_sampler, entropy_penalty, window_size, print_op=False, **kwargs):
        self.bb_task = bb_task
        self.tokenizer = tokenizer
        self.encoder = encoder

        self.batch_size = self.bb_task.batch_size
        self.num_gens = num_gens
        self.lr = lr
        self.num_opt_steps = num_opt_steps
        self.patience = patience
        self.resampling_weight = resampling_weight
        self.encoder_obj = encoder_obj
        self.optimize_latent = optimize_latent
        self.position_sampler = position_sampler
        self.entropy_penalty = entropy_penalty
        self.window_size = window_size
        self.print_op = print_op

    def optimize(self, surrogate_model, acq_fn, active_candidates, active_targets, log_prefix=''):
        self.surrogate_model = surrogate_model
        self.acq_fn = acq_fn
        self.active_candidates = active_candidates
        self.active_targets = active_targets
        if self.resampling_weight is None:
            weights = np.ones(self.active_targets.shape[0]) / self.active_targets.shape[0]
        else:
            _, weights, _ = weighted_resampling(self.active_targets, k=self.resampling_weight)

        base_cand_batches = []
        new_seq_batches = []
        new_seq_scores = []
        batch_entropy = []
        whole_history = []
        for gen_idx in range(self.num_gens):
            print(f"\n---- optimizing generation {gen_idx} / {self.num_gens} ----")
            base_candidates, best_seqs, best_score, best_entropy, history = self.loop(self.acq_fn, weights)

            base_cand_batches.append(base_candidates.copy())
            new_seq_batches.append(best_seqs.copy())
            new_seq_scores.append(best_score)
            batch_entropy.append(best_entropy)
            whole_history.extend(history)
    
        # Get best results
        best_idx = np.argmin(new_seq_scores)
        best_seqs = new_seq_batches[best_idx]
        metrics = {}
        metrics['best_acq_val_written'] = -new_seq_scores[best_idx]
        metrics['best_acq_val'] = self.acq_fn(best_seqs[None, :]).mean().item()
        whole_history = np.array(whole_history) # list of acq_val
        pareto_front = pareto_frontier(None, whole_history, maximize=True)[1]
        metrics['history_pareto'] = pareto_front
        metrics['best_seqs'] = best_seqs    
        metrics['base_candidates'] = base_cand_batches[best_idx]
        if self.print_op:
            print("best_acq_val_written", metrics['best_acq_val_written'])
            print("best_acq_val", metrics['best_acq_val'])
            print(pareto_front, pareto_front.shape)
        return metrics, (base_cand_batches, new_seq_batches, new_seq_scores, batch_entropy)

    def loop(self, acq_fn, weights):
        t0 = time.time()
        # select candidate sequences to mutate
        base_idxs = np.random.choice(np.arange(weights.shape[0]), self.batch_size, p=weights, replace=True)
        base_candidates = self.active_candidates[base_idxs]
        base_seqs = np.array([cand.mutant_residue_seq for cand in base_candidates])
        base_tok_idxs = str_to_tokens(base_seqs, self.encoder.tokenizer)
        base_mask = (base_tok_idxs != self.encoder.tokenizer.padding_idx)
        base_lens = base_mask.float().sum(-1).long()
        tgt_lens = None if self.bb_task.allow_len_change else base_lens

        with torch.no_grad():
            window_mask_idxs, _ = evaluate_windows_simple(
                base_seqs, self.encoder, self.window_size, replacement=True, encoder_obj=self.encoder_obj
            )

        # select token positions to mutate
        if self.position_sampler == 'uniform':
            mask_idxs = np.concatenate([
                np.concatenate(random.sample(w_idxs,self.bb_task.max_num_edits),axis=-1) for w_idxs in window_mask_idxs.values()
            ])
        else:
            raise ValueError

        with torch.no_grad():
            src_tok_idxs = base_tok_idxs.clone().to(self.surrogate_model.device)
            if self.encoder_obj == 'lanmt':
                src_tok_idxs = corrupt_tok_idxs(
                    src_tok_idxs, self.encoder.tokenizer, max_len_delta=None, select_idxs=mask_idxs
                )
                opt_features, src_mask = self.encoder.get_token_features(src_tok_idxs)
            elif self.encoder_obj == 'mlm':
                # this line assumes padding tokens are always added at the end
                np.put_along_axis(src_tok_idxs, mask_idxs, self.encoder.tokenizer.masking_idx, axis=1)
                src_tok_features, src_mask = self.encoder.get_token_features(src_tok_idxs)
                opt_features = np.take_along_axis(src_tok_features, mask_idxs[..., None], axis=1)
            else:
                raise ValueError

            # initialize latent token-choice decision variables
            opt_params = torch.empty(
                *opt_features.shape, requires_grad=self.optimize_latent, device=self.surrogate_model.device,
                dtype=self.surrogate_model.dtype
            )
            opt_params.copy_(opt_features)

        # optimize decision variables
        optimizer = torch.optim.Adam(params=[opt_params], lr=self.lr, betas=(0., 1e-2))
        lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.patience)
        best_score, best_step = None, 0

        best_seqs = base_seqs

        t1 = time.time()
        s0, s1, s2 = 0, 0, 0
        history = []
        for step_idx in range(self.num_opt_steps):
            t10 = time.time()
            opt_params.grad = None
            if self.encoder_obj == 'lanmt':
                lat_tok_features, pooled_features = self.encoder.pool_features(opt_params, src_mask)
                tgt_tok_logits, tgt_mask = self.encoder.logits_from_features(
                    opt_params, src_mask, lat_tok_features, tgt_lens
                )
                tgt_tok_idxs, logit_entropy = self.encoder.sample_tgt_tok_idxs(
                    tgt_tok_logits, tgt_mask, temp=1.
                )
            elif self.encoder_obj == 'mlm':
                current_features = src_tok_features.clone()
                np.put_along_axis(current_features, mask_idxs[..., None], opt_params, axis=1)
                lat_tok_features, pooled_features = self.encoder.pool_features(current_features, src_mask)
                tgt_tok_logits, tgt_mask = self.encoder.logits_from_features(
                    current_features, src_mask, lat_tok_features, tgt_lens
                )
                new_tok_idxs, logit_entropy = sample_tokens(
                    base_tok_idxs, tgt_tok_logits, self.encoder.tokenizer, replacement=False
                )
                new_tok_idxs = np.take_along_axis(new_tok_idxs, mask_idxs, axis=1)
                tgt_tok_idxs = src_tok_idxs.clone()
                np.put_along_axis(tgt_tok_idxs, mask_idxs, new_tok_idxs, axis=1)
                logit_entropy = np.take_along_axis(logit_entropy, mask_idxs, axis=1)
            else:
                raise ValueError
            
            lat_acq_vals = acq_fn(pooled_features.unsqueeze(0))
            loss = -lat_acq_vals.mean() + self.entropy_penalty * logit_entropy.mean()
            if self.optimize_latent:
                loss.backward()
                optimizer.step()
                lr_sched.step(loss)

            tgt_seqs = tokens_to_str(tgt_tok_idxs, self.encoder.tokenizer)
            with torch.no_grad():
                act_acq_vals = acq_fn(tgt_seqs[None, :]).mean().item()

            curr_score = -1.0 * act_acq_vals
            history.append(-curr_score)
            
            if best_score is None or curr_score <= best_score:
                best_step = step_idx + 1
                best_seqs_for_save = tgt_seqs.copy()
                best_score = curr_score
                best_entropy = logit_entropy.mean().item()
            if self.print_op:
                print(f"curr_score: {curr_score:.5f}, best_score: {best_score:.5f}")

        t2 = time.time()
        if self.print_op:
            print("Time for whole process: %.4f" % (t2 - t0))
            print("     Time for sampling: %.4f" % (t1 - t0))
            print("     Time for optimization: %.4f" % (t2 - t1))
        return base_candidates, best_seqs_for_save, best_score, best_entropy, history
    
