"""
ppo/rollout.py

RolloutBuffer stores one full rollout (n_steps × n_envs) of experience.

Key design choices (Section 7):
  - terminated and truncated stored separately (for correct GAE masking).
  - lstm_h and lstm_c stored BEFORE each step (Detail 4, Section 4).
  - get_minibatches shuffles SEQUENCES not individual timesteps (Constraint 6).
  - Sequences are built per-env (no cross-env temporal bleeding).
"""

import numpy as np
import torch


class RolloutBuffer:

    def __init__(self, cfg, hidden_size: int, device: torch.device):
        self.cfg = cfg
        self.device = device
        n = cfg.ppo.n_steps
        e = cfg.env.n_envs
        h = hidden_size
        self.n_steps = n
        self.n_envs = e
        self.hidden = h
        self.ptr = 0

        # Observations — stored as uint8 to save memory
        self.obs = np.zeros((n, e, 7, 7, 3), dtype=np.uint8)

        # Actions / log-probs
        self.actions = np.zeros((n, e), dtype=np.int64)
        self.log_probs = np.zeros((n, e), dtype=np.float32)

        # Rewards
        self.rewards_ext = np.zeros((n, e), dtype=np.float32)
        self.rewards_int = np.zeros((n, e), dtype=np.float32)

        # Termination signals
        self.terminated = np.zeros((n, e), dtype=bool)
        self.truncated = np.zeros((n, e), dtype=bool)
        self.done = np.zeros((n, e), dtype=bool)

        # Value estimates
        self.values_ext = np.zeros((n, e), dtype=np.float32)
        self.values_int = np.zeros((n, e), dtype=np.float32)

        # LSTM state BEFORE each step — squeezed (no num_layers dim)
        self.lstm_h = np.zeros((n, e, h), dtype=np.float32)
        self.lstm_c = np.zeros((n, e, h), dtype=np.float32)

        # Computed by GAE
        self.advantages_ext = np.zeros((n, e), dtype=np.float32)
        self.advantages_int = np.zeros((n, e), dtype=np.float32)
        self.returns_ext = np.zeros((n, e), dtype=np.float32)
        self.returns_int = np.zeros((n, e), dtype=np.float32)

    def add(
        self,
        step: int,
        obs,
        actions,
        log_probs,
        rewards_ext,
        rewards_int,
        terminated,
        truncated,
        values_ext,
        values_int,
        lstm_h,     # (1, n_envs, hidden) or (n_envs, hidden) — squeezed before store
        lstm_c,
    ):
        self.obs[step] = obs
        self.actions[step] = actions
        self.log_probs[step] = log_probs
        self.rewards_ext[step] = rewards_ext
        self.rewards_int[step] = rewards_int
        self.terminated[step] = terminated
        self.truncated[step] = truncated
        self.done[step] = terminated | truncated

        if isinstance(values_ext, torch.Tensor):
            values_ext = values_ext.squeeze(-1).cpu().numpy()
        if isinstance(values_int, torch.Tensor):
            values_int = values_int.squeeze(-1).cpu().numpy()
        self.values_ext[step] = values_ext
        self.values_int[step] = values_int if values_int is not None else 0.0

        # Squeeze LSTM state for storage
        if isinstance(lstm_h, torch.Tensor):
            lstm_h = lstm_h.squeeze(0).detach().cpu().numpy()
        if isinstance(lstm_c, torch.Tensor):
            lstm_c = lstm_c.squeeze(0).detach().cpu().numpy()
        self.lstm_h[step] = lstm_h
        self.lstm_c[step] = lstm_c

    def get_minibatches(self, seq_len: int, n_minibatches: int):
        """
        Yields minibatches of sequences (shuffled at sequence level).

        Reshape order: (n_steps, n_envs, ...) → (n_envs, n_steps, ...) → flatten
        so that each env's timesteps are contiguous → no cross-env bleeding.

        DETAIL 5: yields h0, c0 (first step of each sequence) unsqueezed to (1, B, h).
        """
        N = self.n_steps
        E = self.n_envs
        H = self.hidden
        SL = seq_len
        n_seq = (N * E) // SL

        # Reorder dims to (n_envs, n_steps, ...) then flatten first two
        def reorder(x):
            # x: (N, E, ...) → (E, N, ...) → (E*N, ...)
            ax = np.moveaxis(x, 1, 0)           # (E, N, ...)
            shape = ax.shape
            return ax.reshape(shape[0] * shape[1], *shape[2:])

        obs_flat    = reorder(self.obs)          # (E*N, 7,7,3)
        act_flat    = reorder(self.actions)      # (E*N,)
        lp_flat     = reorder(self.log_probs)
        adv_ext_f   = reorder(self.advantages_ext)
        adv_int_f   = reorder(self.advantages_int)
        ret_ext_f   = reorder(self.returns_ext)
        ret_int_f   = reorder(self.returns_int)
        h_flat      = reorder(self.lstm_h)       # (E*N, H)
        c_flat      = reorder(self.lstm_c)

        # Build sequence views: (n_seq, SL, ...)
        obs_seqs   = obs_flat[:n_seq * SL].reshape(n_seq, SL, 7, 7, 3)
        act_seqs   = act_flat[:n_seq * SL].reshape(n_seq, SL)
        lp_seqs    = lp_flat[:n_seq * SL].reshape(n_seq, SL)
        adv_e_seqs = adv_ext_f[:n_seq * SL].reshape(n_seq, SL)
        adv_i_seqs = adv_int_f[:n_seq * SL].reshape(n_seq, SL)
        ret_e_seqs = ret_ext_f[:n_seq * SL].reshape(n_seq, SL)
        ret_i_seqs = ret_int_f[:n_seq * SL].reshape(n_seq, SL)
        # h0, c0: state at FIRST step of each sequence
        h0_seqs    = h_flat[:n_seq * SL:SL]     # (n_seq, H) — first of each seq
        c0_seqs    = c_flat[:n_seq * SL:SL]

        # Shuffle sequences (Constraint 6)
        seq_perm = np.random.permutation(n_seq)
        seqs_per_mb = n_seq // n_minibatches

        for mb_idx in range(n_minibatches):
            mb_seqs = seq_perm[mb_idx * seqs_per_mb: (mb_idx + 1) * seqs_per_mb]
            B = len(mb_seqs) * SL  # total timesteps in minibatch

            obs_mb   = torch.tensor(obs_seqs[mb_seqs].reshape(B, 7, 7, 3), device=self.device)
            act_mb   = torch.tensor(act_seqs[mb_seqs].reshape(B), device=self.device, dtype=torch.long)
            lp_mb    = torch.tensor(lp_seqs[mb_seqs].reshape(B), device=self.device)
            adv_e_mb = torch.tensor(adv_e_seqs[mb_seqs].reshape(B), device=self.device)
            adv_i_mb = torch.tensor(adv_i_seqs[mb_seqs].reshape(B), device=self.device)
            ret_e_mb = torch.tensor(ret_e_seqs[mb_seqs].reshape(B), device=self.device)
            ret_i_mb = torch.tensor(ret_i_seqs[mb_seqs].reshape(B), device=self.device)
            # h0, c0: (1, len(mb_seqs), H)
            h0_mb = torch.tensor(h0_seqs[mb_seqs], device=self.device).unsqueeze(0)
            c0_mb = torch.tensor(c0_seqs[mb_seqs], device=self.device).unsqueeze(0)

            yield (
                obs_mb, (h0_mb, c0_mb), act_mb,
                adv_e_mb, adv_i_mb,
                ret_e_mb, ret_i_mb,
                lp_mb,
                len(mb_seqs),  # n_sequences in minibatch (for LSTM seq replay)
                SL,
            )
