import torch
import logging
import numpy as np
import itertools
from dataclasses import dataclass
from tmrl.custom.custom_algorithms import SpinupSacAgent, REDQSACAgent
from tmrl.config import config_constants as cfg

@dataclass(eq=0)
class SAC_Agent(SpinupSacAgent):
    mixed_precision: bool = False # Use automatic mixed precision (AMP) for training. This is a PyTorch feature that allows you to use half-precision floats (FP16) for training, which can speed up training and reduce memory usage.

    def __post_init__(self):
        super().__post_init__()
        if self.mixed_precision:
            self.scaler = torch.amp.GradScaler()
            logging.info(f"Mixed precision training enabled")
    
    def train(self, batch):

        o, a, r, o2, d, _ = batch # Observations, actions, rewards, next observations, done flags, and info dicts

        # Flag to track if we're using mixed precision
        amp_enabled = self.mixed_precision and torch.cuda.is_available()

        with torch.amp.autocast("cuda", enabled=amp_enabled):
            # Compute the loss and gradients
            pi, logp_pi = self.model.actor(o)

            # loss_alpha:
            loss_alpha = None
            if self.learn_entropy_coef:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                alpha_t = torch.exp(self.log_alpha.detach())
                loss_alpha = -(self.log_alpha * (logp_pi + self.target_entropy).detach()).mean()
            else:
                alpha_t = self.alpha_t

        # Optimize entropy coefficient, also called entropy temperature or alpha in SAC literature
        if loss_alpha is not None:
            self.alpha_optimizer.zero_grad()
            if amp_enabled:
                self.scaler.scale(loss_alpha).backward()
                self.scaler.step(self.alpha_optimizer)
            else:
                loss_alpha.backward()
                self.alpha_optimizer.step()

        # Calculate Q losses with mixed precision if enabled
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            # Q-values for current states and actions
            q1 = self.model.q1(o, a)
            q2 = self.model.q2(o, a)
            
            # Bellman backup for Q functions - no gradient needed
            with torch.no_grad():
                # Target actions come from *current* policy
                a2, logp_a2 = self.model.actor(o2)
                
                # Target Q-values
                q1_pi_targ = self.model_target.q1(o2, a2)
                q2_pi_targ = self.model_target.q2(o2, a2)
                q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
                backup = r + self.gamma * (1 - d) * (q_pi_targ - alpha_t * logp_a2)
            
            # MSE loss against Bellman backup
            loss_q1 = ((q1 - backup)**2).mean()
            loss_q2 = ((q2 - backup)**2).mean()
            loss_q = (loss_q1 + loss_q2) / 2

        # Update critics with mixed precision if enabled
        self.q_optimizer.zero_grad()
        if amp_enabled:
            self.scaler.scale(loss_q).backward()
            self.scaler.step(self.q_optimizer)
        else:
            loss_q.backward()
            self.q_optimizer.step()

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        self.model.q1.requires_grad_(False)
        self.model.q2.requires_grad_(False)

        # Policy loss calculation with mixed precision if enabled
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            # Get Q-values for current policy
            q1_pi = self.model.q1(o, pi)
            q2_pi = self.model.q2(o, pi)
            q_pi = torch.min(q1_pi, q2_pi)
            
            # Entropy-regularized policy loss
            loss_pi = (alpha_t * logp_pi - q_pi).mean()

            if loss_pi < 0:
                logging.debug(f"Negative loss_pi: {loss_pi.item()} Alpha: {alpha_t.item()} Logp_pi: {logp_pi.mean().item()} Q_pi: {q_pi.mean().item()}")
                
        # Update actor with mixed precision if enabled
        self.pi_optimizer.zero_grad()
        if amp_enabled:
            self.scaler.scale(loss_pi).backward()
            self.scaler.step(self.pi_optimizer)
            self.scaler.update()
        else:
            loss_pi.backward()
            self.pi_optimizer.step()

         # Unfreeze Q-networks so you can optimize it at next DDPG step.
        self.model.q1.requires_grad_(True)
        self.model.q2.requires_grad_(True)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.model.parameters(), self.model_target.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

        # FIXME: remove debug info
        with torch.no_grad():

            if not cfg.DEBUG_MODE:
                ret_dict = dict(
                    loss_actor=loss_pi.detach().item(),
                    loss_critic=loss_q.detach().item(),
                )
            else:
                q1_o2_a2 = self.model.q1(o2, a2)
                q2_o2_a2 = self.model.q2(o2, a2)
                q1_targ_pi = self.model_target.q1(o, pi)
                q2_targ_pi = self.model_target.q2(o, pi)
                q1_targ_a = self.model_target.q1(o, a)
                q2_targ_a = self.model_target.q2(o, a)

                diff_q1pt_qpt = (q1_pi_targ - q_pi_targ).detach()
                diff_q2pt_qpt = (q2_pi_targ - q_pi_targ).detach()
                diff_q1_q1t_a2 = (q1_o2_a2 - q1_pi_targ).detach()
                diff_q2_q2t_a2 = (q2_o2_a2 - q2_pi_targ).detach()
                diff_q1_q1t_pi = (q1_pi - q1_targ_pi).detach()
                diff_q2_q2t_pi = (q2_pi - q2_targ_pi).detach()
                diff_q1_q1t_a = (q1 - q1_targ_a).detach()
                diff_q2_q2t_a = (q2 - q2_targ_a).detach()
                diff_q1_backup = (q1 - backup).detach()
                diff_q2_backup = (q2 - backup).detach()
                diff_q1_backup_r = (q1 - backup + r).detach()
                diff_q2_backup_r = (q2 - backup + r).detach()

                ret_dict = dict(
                    loss_actor=loss_pi.detach().item(),
                    loss_critic=loss_q.detach().item(),
                    # debug:
                    debug_log_pi=logp_pi.detach().mean().item(),
                    debug_log_pi_std=logp_pi.detach().std().item(),
                    debug_logp_a2=logp_a2.detach().mean().item(),
                    debug_logp_a2_std=logp_a2.detach().std().item(),
                    debug_q_a1=q_pi.detach().mean().item(),
                    debug_q_a1_std=q_pi.detach().std().item(),
                    debug_q_a1_targ=q_pi_targ.detach().mean().item(),
                    debug_q_a1_targ_std=q_pi_targ.detach().std().item(),
                    debug_backup=backup.detach().mean().item(),
                    debug_backup_std=backup.detach().std().item(),
                    debug_q1=q1.detach().mean().item(),
                    debug_q1_std=q1.detach().std().item(),
                    debug_q2=q2.detach().mean().item(),
                    debug_q2_std=q2.detach().std().item(),
                    debug_diff_q1=diff_q1_backup.mean().item(),
                    debug_diff_q1_std=diff_q1_backup.std().item(),
                    debug_diff_q2=diff_q2_backup.mean().item(),
                    debug_diff_q2_std=diff_q2_backup.std().item(),
                    debug_diff_r_q1=diff_q1_backup_r.mean().item(),
                    debug_diff_r_q1_std=diff_q1_backup_r.std().item(),
                    debug_diff_r_q2=diff_q2_backup_r.mean().item(),
                    debug_diff_r_q2_std=diff_q2_backup_r.std().item(),
                    debug_diff_q1pt_qpt=diff_q1pt_qpt.mean().item(),
                    debug_diff_q2pt_qpt=diff_q2pt_qpt.mean().item(),
                    debug_diff_q1_q1t_a2=diff_q1_q1t_a2.mean().item(),
                    debug_diff_q2_q2t_a2=diff_q2_q2t_a2.mean().item(),
                    debug_diff_q1_q1t_pi=diff_q1_q1t_pi.mean().item(),
                    debug_diff_q2_q2t_pi=diff_q2_q2t_pi.mean().item(),
                    debug_diff_q1_q1t_a=diff_q1_q1t_a.mean().item(),
                    debug_diff_q2_q2t_a=diff_q2_q2t_a.mean().item(),
                    debug_diff_q1pt_qpt_std=diff_q1pt_qpt.std().item(),
                    debug_diff_q2pt_qpt_std=diff_q2pt_qpt.std().item(),
                    debug_diff_q1_q1t_a2_std=diff_q1_q1t_a2.std().item(),
                    debug_diff_q2_q2t_a2_std=diff_q2_q2t_a2.std().item(),
                    debug_diff_q1_q1t_pi_std=diff_q1_q1t_pi.std().item(),
                    debug_diff_q2_q2t_pi_std=diff_q2_q2t_pi.std().item(),
                    debug_diff_q1_q1t_a_std=diff_q1_q1t_a.std().item(),
                    debug_diff_q2_q2t_a_std=diff_q2_q2t_a.std().item(),
                    debug_r=r.detach().mean().item(),
                    debug_r_std=r.detach().std().item(),
                    debug_d=d.detach().mean().item(),
                    debug_d_std=d.detach().std().item(),
                    debug_a_0=a[:, 0].detach().mean().item(),
                    debug_a_0_std=a[:, 0].detach().std().item(),
                    debug_a_1=a[:, 1].detach().mean().item(),
                    debug_a_1_std=a[:, 1].detach().std().item(),
                    debug_a_2=a[:, 2].detach().mean().item(),
                    debug_a_2_std=a[:, 2].detach().std().item(),
                    debug_a1_0=pi[:, 0].detach().mean().item(),
                    debug_a1_0_std=pi[:, 0].detach().std().item(),
                    debug_a1_1=pi[:, 1].detach().mean().item(),
                    debug_a1_1_std=pi[:, 1].detach().std().item(),
                    debug_a1_2=pi[:, 2].detach().mean().item(),
                    debug_a1_2_std=pi[:, 2].detach().std().item(),
                    debug_a2_0=a2[:, 0].detach().mean().item(),
                    debug_a2_0_std=a2[:, 0].detach().std().item(),
                    debug_a2_1=a2[:, 1].detach().mean().item(),
                    debug_a2_1_std=a2[:, 1].detach().std().item(),
                    debug_a2_2=a2[:, 2].detach().mean().item(),
                    debug_a2_2_std=a2[:, 2].detach().std().item(),
                )
        if self.learn_entropy_coef:
            ret_dict["loss_entropy_coef"] = loss_alpha.detach().item()
            ret_dict["entropy_coef"] = alpha_t.item()

        """ if amp_enabled:
            ret_dict["mixed_precision"] = True """

        return ret_dict
    

# REDQ-SAC =============================================================================================================

@dataclass(eq=0)
class REDQSAC_Agent(REDQSACAgent):
    mixed_precision: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.mixed_precision:
            self.scaler = torch.amp.GradScaler()
            logging.info(f"Mixed precision training enabled for REDQ-SAC")
        # Un solo optimizador para todos los críticos

        all_q_params = itertools.chain(*[q.parameters() for q in self.model.qs])
        self.q_optimizer = torch.optim.Adam(all_q_params, lr=self.lr_critic)
        # Asegura que self.alpha_t siempre esté definido
        if not self.learn_entropy_coef:
            self.alpha_t = torch.tensor(float(self.alpha)).to(self.device)
        else:
            self.alpha_t = torch.exp(self.log_alpha.detach())

    def train(self, batch):
        self.i_update += 1
        update_policy = (self.i_update % self.q_updates_per_policy_update == 0)

        o, a, r, o2, d, _ = batch
        amp_enabled = self.mixed_precision and torch.cuda.is_available()

        if update_policy:
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                pi, logp_pi = self.model.actor(o)
        # FIXME? log_prob = log_prob.reshape(-1, 1)

        loss_alpha = None
        if self.learn_entropy_coef and update_policy:
            alpha_t = torch.exp(self.log_alpha.detach())
            loss_alpha = -(self.log_alpha * (logp_pi + self.target_entropy).detach()).mean()
            self.alpha_t = alpha_t
        else:
            alpha_t = self.alpha_t

        if loss_alpha is not None:
            self.alpha_optimizer.zero_grad()
            if amp_enabled:
                self.scaler.scale(loss_alpha).backward()
                self.scaler.step(self.alpha_optimizer)
            else:
                loss_alpha.backward()
                self.alpha_optimizer.step()

        with torch.no_grad():
            a2, logp_a2 = self.model.actor(o2)
            sample_idxs = np.random.choice(self.n, self.m, replace=False)
            q_prediction_next_list = [self.model_target.qs[i](o2, a2) for i in sample_idxs]
            q_prediction_next_cat = torch.stack(q_prediction_next_list, -1)
            min_q, _ = torch.min(q_prediction_next_cat, dim=1, keepdim=True)
            backup = r.unsqueeze(dim=-1) + self.gamma * (1 - d.unsqueeze(dim=-1)) * (min_q - alpha_t * logp_a2.unsqueeze(dim=-1))

        q_prediction_list = [q(o, a) for q in self.model.qs]
        q_prediction_cat = torch.stack(q_prediction_list, -1)
        backup = backup.expand((-1, self.n)) if backup.shape[1] == 1 else backup

        with torch.amp.autocast("cuda", enabled=amp_enabled):
            loss_q = self.criterion(q_prediction_cat, backup)

        self.q_optimizer.zero_grad()
        if amp_enabled:
            self.scaler.scale(loss_q).backward()
            self.scaler.step(self.q_optimizer)
            self.scaler.update()
        else:
            loss_q.backward()
            self.q_optimizer.step()

        if update_policy:
            for q in self.model.qs:
                q.requires_grad_(False)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                qs_pi = [q(o, pi) for q in self.model.qs]
                qs_pi_cat = torch.stack(qs_pi, -1)
                ave_q = torch.mean(qs_pi_cat, dim=1, keepdim=True)
                loss_pi = (alpha_t * logp_pi.unsqueeze(dim=-1) - ave_q).mean()
            self.pi_optimizer.zero_grad()
            if amp_enabled:
                self.scaler.scale(loss_pi).backward()
                self.scaler.step(self.pi_optimizer)
                self.scaler.update()
            else:
                loss_pi.backward()
                self.pi_optimizer.step()
            for q in self.model.qs:
                q.requires_grad_(True)

        with torch.no_grad():
            for p, p_targ in zip(self.model.parameters(), self.model_target.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

        if update_policy:
            self.loss_pi = loss_pi.detach()
        ret_dict = dict(
            loss_actor=self.loss_pi.detach().item(),
            loss_critic=loss_q.detach().item(),
        )

        if self.learn_entropy_coef and update_policy:
            ret_dict["loss_entropy_coef"] = loss_alpha.detach().item()
            ret_dict["entropy_coef"] = alpha_t.item()

        return ret_dict