import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn.functional as F
import numpy as np

from training import train_MADDPG as py
import testing.test_env as te

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


num = 5
server = 3

fi_m = te.fi_m[:server]
fi_l = te.fi_l[:num]

Di = 400 * np.ones(num)
Ci = 900 * np.ones(num)

hi = te.hi[:num, :server]

env = py.Env(
    alpha=0.6, beta=0.3, gamma=0.1,
    B=10,
    N0=pow(10, -174 / 10) * 0.001,
    hi=hi, pi=500,
    K=num, ser=server,
    Di=Di, Ci=Ci,
    fi_m=fi_m, fi_l=fi_l
)


state_dims = [(2 * env.ser + 1) * env.K for _ in range(env.K)]
action_dims = [2 * env.ser + 1 for _ in range(env.K)]

agents = py.MADDPG(env, state_dims, action_dims, state_dims[0] + action_dims[0], 64)
agents.load(path="../MADDPG-5.pth", map_location=device)


def generate_bad_states(env, num_samples=500, threshold=-100):

    bad_states = []

    for _ in range(num_samples):
        init = np.random.uniform(0, 1, (env.K, env.ser * 2 + 1))
        s, r, _, _ = env.step(init)

        if np.sum(r) < threshold:
            bad_states.append(s)

    return np.array(bad_states)

bad_states = generate_bad_states(env)


unlearn_steps = 500
lr = 1e-4
lambda_old = 0.5

print("\nStart KL Unlearning...\n")

for agent_id, agent in enumerate(agents.agents):

    optimizer = torch.optim.Adam(agent.actor.parameters(), lr=lr)

    for step in range(unlearn_steps):

        idx = np.random.randint(0, len(bad_states))
        state = torch.FloatTensor(bad_states[idx][agent_id]).unsqueeze(0).to(device)

        action = agent.actor(state)

        prob = F.softmax(action, dim=-1)

        uniform = torch.ones_like(prob) / prob.shape[-1]

        kl_unlearn = torch.sum(prob * torch.log((prob + 1e-8) / (uniform + 1e-8)))


        old_action = action.detach()
        old_prob = F.softmax(old_action, dim=-1)

        kl_keep = torch.sum(old_prob * torch.log((old_prob + 1e-8) / (prob + 1e-8)))

        loss = kl_unlearn + lambda_old * kl_keep

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"Agent {agent_id} | Step {step} | Loss {loss.item():.6f}")


save_path = "../KUC-MADDPG-5.pth"
for agt in agents.agents: torch.save(agt.actor.state_dict(), save_path)

print("\nKL Unlearning Finished")