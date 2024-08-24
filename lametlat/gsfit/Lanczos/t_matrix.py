"""
Convert the 2pt correlation function to the T-matrix.

Ref: https://arxiv.org/pdf/2406.20009
"""

# %%
import numpy as np
import gvar as gv
from lametlat.utils.resampling import bootstrap


def check_alpha_2(pt2_norm):
    numerator = pt2_norm[3] - 2 * pt2_norm[2] * pt2_norm[1] + pt2_norm[1] ** 3
    denominator = pt2_norm[2] - pt2_norm[1] ** 2

    return numerator / denominator


def cut_spurious(t_matrix, tolerance=0.01):
    t_matrix_cut = t_matrix[1:, 1:]
    eigenvalues = np.linalg.eigvals(t_matrix)
    eigenvalues_cut = np.linalg.eigvals(t_matrix_cut)
    
    # Remove elements from eigenvalues that are also in eigenvalues_cut (within 1% tolerance)
    tolerance = 0.01  # 1% tolerance
    eigenvalues_to_keep = []
    
    for ev in eigenvalues:
        if not any(np.isclose(ev, ev_cut, rtol=tolerance) for ev_cut in eigenvalues_cut):
            eigenvalues_to_keep.append(ev)
            
    eigenvalues_to_keep = np.array(eigenvalues_to_keep)
    
    return eigenvalues_to_keep


class T_Matrix:
    def __init__(self, pt2_norm, m):
        self.m = m
        self.Lt = len(pt2_norm)
        self.pt2_norm = pt2_norm
        self.A_ls = np.zeros((self.m + 1, self.Lt), dtype=complex)
        self.B_ls = np.zeros((self.m + 1, self.Lt), dtype=complex)
        self.alpha_ls = np.zeros(self.m + 1, dtype=complex)
        self.beta_ls = np.zeros(self.m + 1, dtype=complex) #TODO: beta can be complex

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
        self.beta_ls[j + 1] = np.sqrt(self.A_ls[j, 2] - self.alpha_ls[j]**2 - self.beta_ls[j]**2)
        # self.beta_ls[j + 1] = np.sqrt(
        #     abs(self.A_ls[j, 2] - self.alpha_ls[j] ** 2 - self.beta_ls[j] ** 2)
        # )  # TODO: abs needs to be checked

    def main(self, ifcheck=False):
        self.update_alpha(1)  # calculate alpha 1
        self.update_beta(1)  # calculate beta 2
        for idx1 in range(1, self.m + 1):
            self.update_A(1, idx1) # calculate A[2, idx1]
            self.update_B(1, idx1) # calculate B[2, idx1]
            
        for idx2 in range(2, self.m):
            self.update_alpha(idx2) # calculate alpha idx2
            self.update_beta(idx2) # calculate beta idx2 + 1
            for idx1 in range(1, 2 * self.m - 2 * idx2 + 2): #* fixed the typo in article
                self.update_A(idx2, idx1) # calculate A[idx2+1, idx1]
                self.update_B(idx2, idx1) # calculate B[idx2+1, idx1]

        self.update_alpha(self.m) # calculate alpha m

        t_matrix = np.zeros((self.m, self.m), dtype=complex)
        for idx1 in range(self.m):
            for idx2 in range(self.m):
                if idx1 == idx2:
                    t_matrix[idx1, idx2] = self.alpha_ls[idx1 + 1].real
                elif abs(idx1 - idx2) == 1:
                    t_matrix[idx1, idx2] = self.beta_ls[max(idx1, idx2) + 1].real
                    
        # * check the alpha 2
        if ifcheck:
            print(">>> check the alpha 2 from the t-matrix: ", self.alpha_ls[2])
            print(">>> check the alpha 2 from the pt2: ", check_alpha_2(self.pt2_norm))

        return t_matrix

if __name__ == "__main__":
    from lametlat.utils.constants import GEV_FM
    a = 0.06
    
    
    pt2 = gv.load("../../../examples/data/pion_2pt_example.dat")
    pt2_bs, _ = bootstrap(pt2, samp_times=100)
    pt2_samp = pt2_bs[10] # Take the first sample

    pt2_norm = pt2_samp / pt2_samp[0]  # normalize by the C(t=0)

    t_matrix_class = T_Matrix(pt2_norm, m=6)
    t_matrix = t_matrix_class.main()

    print(t_matrix)
    
    # Calculate eigenvalues of the t_matrix
    eigenvalues = cut_spurious(t_matrix, tolerance=0.1)
    # Select eigenvalues with imaginary part smaller than threshold
    eigenvalues = eigenvalues[np.abs(eigenvalues.imag) < 1e-12].real
    # Calculate energy states
    energy_states = - GEV_FM / a * np.log( eigenvalues )
    energy_states = np.array([e for e in energy_states if not np.isnan(e) and e > 0])
    energy_states = np.sort(energy_states)
    
    print("Eigenvalues of the t_matrix:")
    print(eigenvalues)
    
    print("Energy states:")
    print(energy_states)

# %%
