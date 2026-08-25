
import random
from collections import deque

import numpy as np


class ReplayBuffer(object):
    def __init__(self, buffer_size, random_seed=12):
        """
        The right side of the deque contains the most recent experiences
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)

    def add(self, s, a, r, t, s2,U_s,omega):
        experience = (s, a, r, t, s2,U_s,omega)
    #    print(f"Adding experience: s={s}, a={a}, r={r}, t={t}, s2={s2},U_s={U_s},omega={omega}")
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)
      #  print(f"Buffer contents after adding: {list(self.buffer)}")

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)
        #print("Sampled raw batch:", batch)
        s_batch   = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch]).reshape(-1, 1)
        t_batch = np.array([_[3] for _ in batch]).reshape(-1, 1)
        s2_batch  = np.array([_[4] for _ in batch])
        U_s_batch = np.array([_[5] for _ in batch])
        omega_batch = np.array([_[6] for _ in batch])                  

        #print(f"Sampled batch: a_batch={a_batch}")

        return s_batch, a_batch, r_batch, t_batch, s2_batch,U_s_batch,omega_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0