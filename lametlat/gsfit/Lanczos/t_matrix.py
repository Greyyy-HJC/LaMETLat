"""
Convert the 2pt correlation function to the T-matrix.

Ref: https://arxiv.org/pdf/2406.20009
"""

# %%
import numpy as np
import gvar as gv
from lametlat.utils.resampling import bootstrap, bs_ls_avg


def check_alpha_2(pt2_norm):
    numerator = pt2_norm[3] - 2 * pt2_norm[2] * pt2_norm[1] + pt2_norm[1] ** 3
    denominator = pt2_norm[2] - pt2_norm[1] ** 2

    return numerator / denominator


class T_Matrix:
    def __init__(self, pt2_norm, m):
        self.m = m
        self.Lt = len(pt2_norm)
        self.pt2_norm = pt2_norm
        self.A_ls = np.zeros((self.m + 1, self.Lt))
        self.B_ls = np.zeros((self.m + 1, self.Lt))
        self.alpha_ls = np.zeros(self.m + 1)
        self.beta_ls = np.zeros(self.m + 1)

        for idx in range(1, self.Lt):
            self.A_ls[1, idx] = self.pt2_norm[idx]
            self.B_ls[1, idx] = 0

        self.beta_ls[1] = 0

    def update_A(self, j, k):  # * Need: A[j, k+2], B[j, k+1], alpha[j], beta[j+1]
        self.A_ls[j + 1, k] = (1 / self.beta_ls[j + 1] ** 2) * (
            self.A_ls[j, k + 2]
            + self.alpha_ls[j] ** 2 * self.A_ls[j, k]
            + self.beta_ls[j] ** 2 * self.A_ls[j - 1, k]
            - 2 * self.alpha_ls[j] * self.A_ls[j, k + 1]
            + 2 * self.alpha_ls[j] * self.beta_ls[j] * self.B_ls[j, k]
            - 2 * self.beta_ls[j] * self.B_ls[j, k + 1]
        )

    def update_B(self, j, k):  # * Need: A[j, k+1], B[j, k], alpha[j], beta[j+1]
        self.B_ls[j + 1, k] = (1 / self.beta_ls[j + 1]) * (
            self.A_ls[j, k + 1]
            - self.alpha_ls[j] * self.A_ls[j, k]
            - self.beta_ls[j] * self.B_ls[j, k]
        )

    def update_alpha(self, j):  # * Need: A[j, 1]
        self.alpha_ls[j] = self.A_ls[j, 1]

    def update_beta(self, j):  # * Need: A[j, 2], alpha[j], beta[j]
        # self.beta_ls[j+1] = np.sqrt(self.A_ls[j, 2] - self.alpha_ls[j]**2 - self.beta_ls[j]**2)
        self.beta_ls[j + 1] = np.sqrt(
            abs(self.A_ls[j, 2] - self.alpha_ls[j] ** 2 - self.beta_ls[j] ** 2)
        )  # TODO: abs needs to be checked

    def main(self):
        self.update_alpha(1)  # calculate alpha 1
        self.update_beta(1)  # calculate beta 2
        for idx1 in range(1, self.m + 1):
            self.update_A(1, idx1)
            self.update_B(1, idx1)

        for idx2 in range(2, self.m):
            self.update_alpha(idx2) # calculate alpha idx2
            self.update_beta(idx2) # calculate beta idx2 + 1
            for idx1 in range(1, self.m - 2 * idx2 + 3):
                self.update_A(idx2, idx1)
                self.update_B(idx2, idx1)

        return self.alpha_ls, self.beta_ls

if __name__ == "__main__":
    pt2 = gv.load("../../../examples/data/pion_2pt_example.dat")
    pt2_bs, _ = bootstrap(pt2, samp_times=100)
    pt2_samp = pt2_bs[0] # Take the first sample

    pt2_norm = pt2_samp / pt2_samp[0]  # normalize by the C(t=0)

    t_matrix = T_Matrix(pt2_norm, m=5)
    alpha_ls, beta_ls = t_matrix.main()
    print(alpha_ls[2])
    print(check_alpha_2(pt2_norm))

    print(alpha_ls)
    print(beta_ls)

# %%
