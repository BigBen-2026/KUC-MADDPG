#Performance analysis of Qmix.
import numpy as np
import torch
import matplotlib.pyplot as plt
import random

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

from training import train_Qmix as py

random.seed(10)
np.random.seed(0)


def main(env):
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

    state_dims = [2 * env.ser + 1 for _ in range(env.K)]

    agents = py.QMIX(env, state_dims, py.argparse.Namespace(
        gamma=0.99,
        batch_size=32,
        capacity=10000,
        hidden_dim=128
    ), device)


    agents.load(
        path=r'C:\Users\ASUS\Desktop\KUC-MADDPG\src\training\Qmix-5.pth'
    )

    reward_history = []

    for i in range(20):
        states = env.reset()
        reward_t = []

        for t in range(50):

            action = agents.take_action(states, eps=0.0)

            next_state, reward_i, done, _ = env.step(action)

            if t % 10 == 0:
                print("Episode: \t{} Reward: \t{:0.2f}".format(i, np.sum(reward_i)))

            reward_t.append(np.sum(reward_i))
            states = next_state

        reward_history.append(np.average(reward_t))

    return np.mean(reward_history)



num = 5
server = 3

reward1 = np.zeros((5, 1))
reward2 = np.zeros((5, 1))
energy1 = np.zeros((5, 1))
time1 = np.zeros((5, 1))
time2 = np.zeros((5, 1))
energy2 = np.zeros((5, 1))
reward3 = np.zeros((5, 1))
time3 = np.zeros((5, 1))
energy3 = np.zeros((5, 1))

fi_m = np.random.uniform(3,7,server)
fi_l = np.random.uniform(0.8,1.5,num)
Di = 400*np.ones(num)
Ci = 900*np.ones(num)

C = [900, 950, 1000, 1050, 1100]
D = [300, 350, 400, 450, 500]

hi = pow(np.random.uniform(50,200,(num,server)),-3)

Bi = 10
B = [5, 10, 15, 20, 25]


for i1 in range(5):
    Di_test = D[i1] * np.ones(num)

    reward1[i1, 0] = main(py.Env(alpha=0.6, beta=0.3, gamma=0.1, B=Bi, N0=pow(10, -174 / 10) * 0.001,
                                  hi=hi, pi=500, K=num, ser=server,
                                  Di=Di_test, Ci=Ci,
                                  fi_m=fi_m, fi_l=fi_l))

    time1[i1, 0] =  (-1) * main(py.Env(alpha=1, beta=0, gamma=0, B=Bi, N0=pow(10, -174 / 10) * 0.001,
                                  hi=hi, pi=500, K=num, ser=server,
                                  Di=Di_test, Ci=Ci,
                                  fi_m=fi_m, fi_l=fi_l))

    energy1[i1, 0] = (-1) * main(py.Env(alpha=0, beta=1,gamma=0, B=Bi, N0=pow(10, -174 / 10) * 0.001,
                                  hi=hi, pi=500, K=num, ser=server,
                                  Di=Di_test, Ci=Ci,
                                  fi_m=fi_m, fi_l=fi_l))


for i2 in range(5):
    Ci_test = C[i2] * np.ones(num)

    reward2[i2, 0] =main(py.Env(alpha=0.6, beta=0.3, gamma=0.1, B=Bi, N0=pow(10, -174 / 10) * 0.001,
                                 hi=hi, pi=500, K=num, ser=server,
                                 Di=Di, Ci=Ci_test,
                                 fi_m=fi_m, fi_l=fi_l))

    time2[i2, 0] = (-1) * main(py.Env(alpha=1, beta=0, gamma=0, B=Bi, N0=pow(10, -174 / 10) * 0.001,
                                  hi=hi, pi=500, K=num, ser=server,
                                  Di=Di, Ci=Ci_test,
                                  fi_m=fi_m, fi_l=fi_l))

    energy2[i2, 0] =  (-1) * main(py.Env(alpha=0, beta=1, gamma=0, B=Bi, N0=pow(10, -174 / 10) * 0.001,
                                  hi=hi, pi=500, K=num, ser=server,
                                  Di=Di, Ci=Ci_test,
                                  fi_m=fi_m, fi_l=fi_l))


for i3 in range(5):
    reward3[i3, 0] = main(py.Env(alpha=0.6, beta=0.3, gamma=0.1, B=B[i3], N0=pow(10, -174 / 10) * 0.001,
                       hi=hi, pi=500, K=num, ser=server, Di=Di, Ci=Ci,
                       fi_m=fi_m, fi_l=fi_l))

    time3[i3, 0] = (-1) * main(py.Env(alpha=1, beta=0,gamma=0, B=B[i3], N0=pow(10, -174 / 10) * 0.001,
                        hi=hi, pi=500, K=num, ser=server, Di=Di, Ci=Ci,
                        fi_m=fi_m, fi_l=fi_l))

    energy3[i3, 0] = (-1) * main(py.Env(alpha=0, beta=1, gamma=0, B=B[i3], N0=pow(10, -174 / 10) * 0.001,
                        hi=hi, pi=500, K=num, ser=server, Di=Di, Ci=Ci,
                        fi_m=fi_m, fi_l=fi_l))
