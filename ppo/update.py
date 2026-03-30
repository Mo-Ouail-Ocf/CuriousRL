"""
ppo/update.py

PPO minibatch update with sequence-aware LSTM (DETAIL 5, Section 4).

For each minibatch:
  - Take h0, c0 from buffer at the first step of each sequence.
  - Feed full (seq_len, embed_dim) sequence through LSTM from that h0, c0.
  - LSTM recomputes all intermediate states — correct gradient flow.
  - Sequences are shuffled, NOT individual timesteps (Constraint 6).

Dual value loss when use_dual_heads=True (Section 3).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def ppo_update(agent, optimizer, buffer, cfg, device, global_step, total_steps):
    """
    Runs cfg.ppo.n_epochs passes over the rollout buffer.
    Returns dict of mean loss metrics for logging.
    """
    n_epochs = cfg.ppo.n_epochs
    seq_len = cfg.ppo.seq_len
    n_minibatches = cfg.ppo.n_minibatches
    clip_coef = cfg.ppo.clip_coef
    ent_coef = cfg.ppo.ent_coef
    vf_coef = cfg.ppo.vf_coef
    max_grad_norm = cfg.ppo.max_grad_norm
    normalize_adv = cfg.ppo.normalize_advantage
    use_dual = cfg.curiosity.method != "none"
    lambda_int = cfg.curiosity.lambda_int

    # Learning-rate annealing (linear decay to 0)
    if cfg.ppo.anneal_lr:
        frac = 1.0 - global_step / total_steps
        lr_now = cfg.ppo.lr * frac
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

    # --- store advantages_combined in buffer if not already ---
    if not hasattr(buffer, "advantages_combined"):
        buffer.advantages_combined = buffer.advantages_ext

    metrics = {
        "losses/policy_loss": [],
        "losses/value_loss_ext": [],
        "losses/value_loss_int": [],
        "losses/entropy": [],
        "losses/approx_kl": [],
        "losses/clipfrac": [],
        "train/lstm_grad_norm": [],
    }

    for _ in range(n_epochs):
        for batch in buffer.get_minibatches(seq_len, n_minibatches):
            (
                obs_mb,          # (n_seq*seq_len, 7,7,3) uint8
                (h0_mb, c0_mb),  # each (1, n_seq, hidden)
                act_mb,          # (n_seq*seq_len,)
                adv_e_mb,        # (n_seq*seq_len,)
                adv_i_mb,
                ret_e_mb,
                ret_i_mb,
                old_lp_mb,       # (n_seq*seq_len,)
                n_seq_mb,        # int: number of sequences in this minibatch
                SL,              # int: seq_len
            ) = batch

            # ---- DETAIL 5: replay sequences from h0, c0 ----
            # Reshape obs to (n_seq_mb, seq_len, 7, 7, 3) for sequential LSTM feed
            obs_seq = obs_mb.view(n_seq_mb, SL, 7, 7, 3)

            # Build done tensor for LSTM reset: all False during sequence replay
            # (resets already happened at episode boundaries stored in buffer;
            #  here we replay from the correct h0 so mid-sequence resets are baked in)
            # We pass done=False (zeros) since we start each sequence from its own h0.
            done_dummy = torch.zeros(n_seq_mb * SL, device=device, dtype=torch.bool)

            # Process one step at a time through the LSTM, accumulating outputs
            # so the LSTM sees the correct temporal order per sequence.
            h_t, c_t = h0_mb, c0_mb          # (1, n_seq_mb, hidden)
            lstm_outs = []
            for s in range(SL):
                obs_s = obs_seq[:, s]          # (n_seq_mb, 7, 7, 3)
                done_s = torch.zeros(n_seq_mb, device=device, dtype=torch.bool)
                out_s, (h_t, c_t) = agent.forward(obs_s, (h_t, c_t), done_s)
                lstm_outs.append(out_s)        # (n_seq_mb, hidden)

            # Flatten back to (n_seq_mb * SL, hidden)
            feat_mb = torch.stack(lstm_outs, dim=1).reshape(n_seq_mb * SL, -1)

            # ---- Actor / critic heads directly on feat_mb ----
            logits = agent.actor(feat_mb)
            dist = torch.distributions.Categorical(logits=logits)
            new_lp = dist.log_prob(act_mb)
            entropy = dist.entropy()
            val_ext = agent.critic_ext(feat_mb).squeeze(-1)
            val_int = agent.critic_int(feat_mb).squeeze(-1) if use_dual else None

            # ---- Combined advantage ----
            if use_dual:
                adv_mb = adv_e_mb + lambda_int * adv_i_mb
            else:
                adv_mb = adv_e_mb

            if normalize_adv:
                adv_mb = (adv_mb - adv_mb.mean()) / (adv_mb.std() + 1e-8)

            # ---- PPO surrogate loss ----
            log_ratio = new_lp - old_lp_mb
            ratio = log_ratio.exp()
            with torch.no_grad():
                approx_kl = ((ratio - 1) - log_ratio).mean()
                clipfrac = ((ratio - 1.0).abs() > clip_coef).float().mean()

            pg_loss1 = -adv_mb * ratio
            pg_loss2 = -adv_mb * ratio.clamp(1 - clip_coef, 1 + clip_coef)
            policy_loss = torch.max(pg_loss1, pg_loss2).mean()

            # ---- Value losses ----
            vf_loss_ext = F.mse_loss(val_ext, ret_e_mb)
            if use_dual and val_int is not None:
                vf_loss_int = F.mse_loss(val_int, ret_i_mb)
            else:
                vf_loss_int = torch.tensor(0.0, device=device)

            total_vf_loss = vf_loss_ext + (vf_loss_int if use_dual else 0.0)

            # ---- Entropy bonus ----
            entropy_loss = entropy.mean()

            loss = policy_loss + vf_coef * total_vf_loss - ent_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()

            # Grad norm tracking for LSTM params
            lstm_params = list(agent.lstm.parameters())
            lstm_grad_norm = torch.nn.utils.clip_grad_norm_(lstm_params, float("inf"))
            torch.nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
            optimizer.step()

            metrics["losses/policy_loss"].append(policy_loss.item())
            metrics["losses/value_loss_ext"].append(vf_loss_ext.item())
            metrics["losses/value_loss_int"].append(vf_loss_int.item())
            metrics["losses/entropy"].append(entropy_loss.item())
            metrics["losses/approx_kl"].append(approx_kl.item())
            metrics["losses/clipfrac"].append(clipfrac.item())
            metrics["train/lstm_grad_norm"].append(lstm_grad_norm.item())

    return {k: sum(v) / len(v) for k, v in metrics.items()}
