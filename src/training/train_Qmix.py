import numpy as np
import torch
import torch.nn.functional as F
import collections
import random
import torch.optim as optim
import matplotlib.pyplot as plt
import argparse

class Env():
    def __init__(self, alpha, beta,gamma, B, N0, hi, pi, K, ser, Di, Ci, fi_m, fi_l):
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.B, self.N0, self.hi, self.pi, self.K, self.ser = B, N0, hi, pi, K, ser
        self.Di, self.Ci = Di, Ci
        self.fi_m, self.fi_l = fi_m, fi_l

        self.state = np.zeros((self.K, 2 * self.ser + 1))

        self.action_dim = 10
        self.action_map = [
            np.random.uniform(0, 1, (2 * self.ser + 1))
            for _ in range(self.action_dim)
        ]

        self.theta = np.random.uniform(0.8, 1, self.K)
        self.P_transmit = 0.5
        self.mu = 0.01

    def decode_action(self, actions):
        return np.array([self.action_map[a] for a in actions])

    def step(self, actions):

        actions = self.decode_action(actions)


        self.Di = np.random.uniform(300, 500, self.K)
        self.Ci = np.random.uniform(900, 1100, self.K)


        stra, f = np.split(actions, [self.ser + 1], axis=1)


        stra = stra / np.maximum(np.sum(stra, axis=1, keepdims=True), 1e-6)
        f = f / np.maximum(np.sum(f, axis=1, keepdims=True), 1e-6)


        a = self.pi * 0.001 * self.hi

        r_1 = f * self.B * 1e6 * np.log2(1 + (a / self.N0))


        omega = np.random.uniform(0.5, 1.0, self.K)
        r_1 = r_1 * omega[:, None]



        T1_ij = (
                stra[:, :self.ser]
                * self.Di[:, None]
                * 102400
                / (1 + r_1)
        )

        E1_ij = (
                T1_ij
                * (self.pi + self.P_transmit)
                * 1e-5
        )



        T2_ij = (
                stra[:, :self.ser]
                * self.Ci[:, None]
                * 100
                / (self.fi_m * 1000)
        )

        E2_ij = (
                stra[:, :self.ser]
                * self.Ci[:, None]
                * (self.fi_m ** 2)
                * 1e-5
        )



        T3 = (
                stra[:, self.ser]
                * self.Ci
                * 100
                / (self.fi_l * 1000 * self.theta)
        )

        E3 = (
                stra[:, self.ser]
                * self.Ci
                * (self.fi_l ** 2)
                * 1e-5
                / self.theta
        )



        T1 = np.max(T1_ij, axis=1)

        T2 = np.max(T2_ij, axis=1)

        T = np.maximum(T1 + T2, T3)

        E = np.sum(E1_ij + E2_ij, axis=1) + E3



        T_ij = T1_ij + T2_ij



        mask = stra[:, :self.ser] > 1e-6

        success_probs = np.exp(-self.mu * T_ij)

        combined_success_probs = np.ones(self.K)

        for i in range(self.K):

            if np.any(mask[i]):

                actual_success_probs = success_probs[i][mask[i]]

                if len(actual_success_probs) > 0:
                    combined_success_probs[i] = np.prod(actual_success_probs)

        P_fail = 1 - combined_success_probs



        reward = -(
                self.alpha * T
                + self.beta * E
                + self.gamma * P_fail
        )


        self.state = np.random.uniform(
            0, 1,
            (self.K, 2 * self.ser + 1)
        )

        done = [False] * self.K

        return self.state, reward[:, None], done, {}

    def reset(self):
        return np.random.uniform(0, 1, (self.K, 2 * self.ser + 1))


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, s, a, r, ns, d):
        self.buffer.append((s, a, r, ns, d))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (
            np.array(s),
            np.array(a),
            np.array(r),
            np.array(ns),
            np.array(d),
        )

    def size(self):
        return len(self.buffer)



class QNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class MixingNet(torch.nn.Module):
    def __init__(self, n_agents, state_dim):
        super().__init__()
        self.n_agents = n_agents

        self.hyper_w1 = torch.nn.Linear(state_dim, n_agents * 32)
        self.hyper_b1 = torch.nn.Linear(state_dim, 32)

        self.hyper_w2 = torch.nn.Linear(state_dim, 32)
        self.hyper_b2 = torch.nn.Linear(state_dim, 1)

    def forward(self, q, s):
        bs = q.size(0)

        w1 = torch.abs(self.hyper_w1(s)).view(bs, self.n_agents, 32)
        b1 = self.hyper_b1(s)

        hidden = torch.bmm(q.unsqueeze(1), w1).squeeze(1) + b1
        hidden = F.relu(hidden)

        w2 = torch.abs(self.hyper_w2(s)).view(bs, 32, 1)
        b2 = self.hyper_b2(s)

        return torch.bmm(hidden.unsqueeze(1), w2).squeeze(1) + b2



class QMIX:
    def __init__(self, env, state_dims, cfg, device):
        self.K = env.K
        self.device = device
        self.gamma = cfg.gamma
        self.batch_size = cfg.batch_size

        self.q_nets = [
            QNet(state_dims[i], cfg.hidden_dim, 10).to(device)
            for i in range(self.K)
        ]

        self.mixing = MixingNet(self.K, sum(state_dims)).to(device)

        self.optimizer = optim.Adam(
            list(self.mixing.parameters())
            + [p for q in self.q_nets for p in q.parameters()],
            lr=1e-6,
        )

    def take_action(self, s, eps=0.1):
        actions = []
        for i in range(self.K):
            if random.random() < eps:
                actions.append(np.random.randint(10))
            else:
                x = torch.tensor(s[i], dtype=torch.float32).unsqueeze(0).to(self.device)
                q = self.q_nets[i](x)
                actions.append(torch.argmax(q).item())
        return np.array(actions, dtype=np.int64)

    def update(self, buffer):
        if buffer.size() < self.batch_size:
            return

        s, a, r, ns, d = buffer.sample(self.batch_size)

        s = torch.tensor(s, dtype=torch.float32).to(self.device)
        ns = torch.tensor(ns, dtype=torch.float32).to(self.device)
        a = torch.tensor(a, dtype=torch.long).to(self.device)
        r = torch.tensor(r[:, 0], dtype=torch.float32).unsqueeze(1).to(self.device)
        d = torch.tensor(d[:, 0], dtype=torch.float32).unsqueeze(1).to(self.device)

        q_list = []
        target_q_list = []

        for i in range(self.K):
            q = self.q_nets[i](s[:, i, :])
            q = q.gather(1, a[:, i].unsqueeze(1))
            q_list.append(q)

            with torch.no_grad():
                target_q = self.q_nets[i](ns[:, i, :]).max(dim=1, keepdim=True)[0]
                target_q_list.append(target_q)

        q_all = torch.cat(q_list, dim=1)
        target_q_all = torch.cat(target_q_list, dim=1)

        q_total = self.mixing(q_all, s.view(self.batch_size, -1))
        target_total = self.mixing(target_q_all, ns.view(self.batch_size, -1))

        y = r + self.gamma * target_total.detach() * (1 - d)

        loss = F.mse_loss(q_total, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, path):
        torch.save([q.state_dict() for q in self.q_nets], path)

    def load(self, path):
        sd = torch.load(path, map_location=self.device)
        for q, s in zip(self.q_nets, sd):
            q.load_state_dict(s)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--capacity', default=10000, type=int)
    parser.add_argument('--hidden_dim', default=128, type=int)
    parser.add_argument('--max_episode', default=1000, type=int)
    parser.add_argument('--max_step', default=100, type=int)
    args = parser.parse_args()

    num, server = 5, 3
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    fi_m = np.random.uniform(3, 7, server)
    fi_l = np.random.uniform(0.8, 1.5, num)
    Di = np.random.uniform(300, 500, num)
    Ci = np.random.uniform(900, 1100, num)
    hi = pow(np.random.uniform(50, 200, (num, server)), -3)

    env = Env(alpha=0.6,beta=0.3,gamma=0.1,B=10,N0=pow(10, -174 / 10) * 0.001,
              hi=hi,pi=500,K=num,ser=server, Di=Di,Ci=Ci,fi_m=fi_m,fi_l=fi_l)

    state_dims = [2 * server + 1] * num
    agent = QMIX(env, state_dims, args, device)
    buffer = ReplayBuffer(args.capacity)

    rewards = []
    print("🚀 Start Training...")

    for ep in range(args.max_episode):
        s = env.reset()
        ep_r = 0

        for t in range(args.max_step):
            a = agent.take_action(s)
            ns, r, d, _ = env.step(a)

            buffer.add(s, a, r, ns, d)
            agent.update(buffer)

            s = ns
            ep_r += np.sum(r)

        rewards.append(ep_r)
        print(f"Episode {ep+1}, Total Reward: {ep_r:.2f}")

        agent.save("QMix-5.pth")

    plt.plot(rewards)
    plt.show()


if __name__ == "__main__":
    main()