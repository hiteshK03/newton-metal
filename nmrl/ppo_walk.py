from nmrl import REPO
"""Train ANYmal C velocity-tracking locomotion FROM SCRATCH with PPO on a Mac.
Physics: Newton + MuJoCo-Warp on Apple Metal (warp-metal). PPO: PyTorch on MPS.

Usage: python train_anymal_ppo.py N_ENVS N_UPDATES [ckpt_path]
"""
import sys, time
import numpy as np
import torch, torch.nn as nn
from nmrl.anymal_walk_env import AnymalEnv, OBS_DIM, NJ, EP_LEN

N = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
UPDATES = int(sys.argv[2]) if len(sys.argv) > 2 else 600
CKPT = sys.argv[3] if len(sys.argv) > 3 else str(REPO / "checkpoints/anymal_walk.pt")
T = 24
GAMMA, LAM, CLIP, EPOCHS, MINIBATCHES = 0.99, 0.95, 0.2, 5, 4
DESIRED_KL = 0.01

torch.manual_seed(0); np.random.seed(0)
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def mlp(i, o, h=(256, 128, 128)):
    layers, d = [], i
    for k in h:
        layers += [nn.Linear(d, k), nn.ELU()]; d = k
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


class AC(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = mlp(OBS_DIM, NJ); self.critic = mlp(OBS_DIM, 1)
        self.log_std = nn.Parameter(torch.full((NJ,), float(np.log(0.8))))

    def dist(self, o):
        mu = self.actor(o)
        return mu, self.log_std.exp().expand_as(mu)


if __name__ == "__main__":
    env = AnymalEnv(N)
    net = AC().to(dev)
    lr = 5e-4
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    obs_np = env.get_obs_np()
    ep_ret = np.zeros(N); ep_len = np.zeros(N)
    fin_ret, fin_len, fin_fall = [], [], []
    t_start = time.perf_counter(); total = 0; best_score = -1e9
    print(f"N={N} T={T} batch={N*T} updates={UPDATES} device={dev}", flush=True)
    for upd in range(UPDATES):
        B = {k: torch.zeros((T, N) + s, device=dev) for k, s in
             dict(obs=(OBS_DIM,), act=(NJ,), mu=(NJ,), sig=(NJ,), logp=(), val=(), rew=(), done=()).items()}
        lin_acc = vx_acc = 0.0; t_roll = time.perf_counter()
        for t in range(T):
            o = torch.from_numpy(obs_np).to(dev)
            with torch.no_grad():
                mu, sig = net.dist(o)
                a = mu + sig * torch.randn_like(mu)
                logp = (-0.5 * ((a - mu) / sig) ** 2 - sig.log() - 0.9189385).sum(-1)
                v = net.critic(o).squeeze(-1)
            a_np = a.cpu().numpy().astype(np.float32)
            obs_np, rew, done, info = env.step(a_np)
            B["obs"][t] = o; B["act"][t] = a; B["mu"][t] = mu; B["sig"][t] = sig
            B["logp"][t] = logp; B["val"][t] = v
            B["rew"][t] = torch.from_numpy(rew.astype(np.float32)).to(dev)
            B["done"][t] = torch.from_numpy(done.astype(np.float32)).to(dev)
            ep_ret += rew; ep_len += 1
            lin_acc += info[:, 0].mean(); vx_acc += info[:, 2].mean()
            for i in np.flatnonzero(done > 0.5):
                fin_ret.append(ep_ret[i]); fin_len.append(ep_len[i]); fin_fall.append(ep_len[i] < EP_LEN)
            ep_ret[done > 0.5] = 0; ep_len[done > 0.5] = 0
            total += N
        t_roll = time.perf_counter() - t_roll
        with torch.no_grad():
            last_v = net.critic(torch.from_numpy(obs_np).to(dev)).squeeze(-1)
        adv = torch.zeros((T, N), device=dev); last = torch.zeros(N, device=dev)
        for t in reversed(range(T)):
            nv = last_v if t == T - 1 else B["val"][t + 1]
            nt = 1.0 - B["done"][t]
            delta = B["rew"][t] + GAMMA * nv * nt - B["val"][t]
            last = delta + GAMMA * LAM * nt * last
            adv[t] = last
        ret = adv + B["val"]
        f = lambda x: x.reshape(T * N, *x.shape[2:])
        bo, ba, bmu, bsig, blogp, badv, bret = (f(B["obs"]), f(B["act"]), f(B["mu"]), f(B["sig"]),
                                                f(B["logp"]), f(adv), f(ret))
        n = T * N; mb = n // MINIBATCHES
        kls = []
        for _ in range(EPOCHS):
            perm = torch.randperm(n, device=dev)
            for s in range(0, n, mb):
                ix = perm[s:s + mb]
                mu, sig = net.dist(bo[ix]); val = net.critic(bo[ix]).squeeze(-1)
                logp = (-0.5 * ((ba[ix] - mu) / sig) ** 2 - sig.log() - 0.9189385).sum(-1)
                with torch.no_grad():
                    kl = (torch.log(sig / bsig[ix]) + (bsig[ix] ** 2 + (bmu[ix] - mu) ** 2) / (2 * sig ** 2) - 0.5).sum(-1).mean().item()
                    kls.append(kl)
                a_ = badv[ix]; a_ = (a_ - a_.mean()) / (a_.std() + 1e-8)
                ratio = (logp - blogp[ix]).exp()
                pl = -torch.min(ratio * a_, ratio.clamp(1 - CLIP, 1 + CLIP) * a_).mean()
                vl = (val - bret[ix]).pow(2).mean()
                loss = pl + 1.0 * vl
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        kl = float(np.mean(kls))
        if kl > DESIRED_KL * 2: lr = max(1e-4, lr / 1.3)
        elif kl < DESIRED_KL / 2: lr = min(1e-3, lr * 1.3)
        for g in opt.param_groups: g["lr"] = lr
        el = time.perf_counter() - t_start
        if upd % 5 == 0 or upd == UPDATES - 1:
            k = 200
            print(f"upd {upd:4d} | rew/step={B['rew'].mean().item():6.3f} lin_track={lin_acc/T:5.3f} vx={vx_acc/T:6.3f} "
                  f"| ep_ret={np.mean(fin_ret[-k:]) if fin_ret else 0:7.1f} ep_len={np.mean(fin_len[-k:]) if fin_len else 0:5.0f} "
                  f"fall={np.mean(fin_fall[-k:]) if fin_fall else 0:4.2f} | std={net.log_std.exp().mean().item():.2f} lr={lr:.1e} kl={kl:.4f} "
                  f"| {total/el:,.0f} steps/s roll={t_roll:.1f}s | {total/1e6:.1f}M steps {el/60:.1f}min", flush=True)
        ck = {"net": net.state_dict(), "upd": upd, "steps": total}
        if upd % 25 == 0 or upd == UPDATES - 1:
            torch.save(ck, CKPT)
        if upd % 100 == 0 and upd > 0:
            torch.save(ck, CKPT.replace(".pt", f"_upd{upd}.pt"))
        if upd >= 100 and len(fin_ret) > 400:
            score = np.mean(fin_ret[-400:]) - 300 * np.mean(fin_fall[-400:])
            if score > best_score:
                best_score = score; torch.save(ck, CKPT.replace(".pt", "_best.pt"))
    print(f"TRAINING COMPLETE: {UPDATES} updates, {total/1e6:.1f}M env-steps, {(time.perf_counter()-t_start)/60:.1f} min", flush=True)
