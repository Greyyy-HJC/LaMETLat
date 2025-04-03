'''
Functions used to read from the raw data and return numpy array / gvar list of 2pt, 3pt, ratio, meff, etc. 
'''

import numpy as np

def calculate_meff(N_T, C_t_values, boundary="periodic"):
    """
    Calculate the effective mass meff for a range of time slices given the values of C(t).

    Parameters:
    N_T (int): The total number of time slices, which defines the periodicity.
    C_t_values (list of floats): The correlator values C(t) for each time slice.
    boundary (str, optional): Boundary condition. Can be "periodic" or "anti-periodic". Defaults to "periodic".

    Returns:
    list of floats: The effective mass meff for each time slice.
    """

    # Initialize the list to hold the meff values
    meff_values = []

    # Define the equation to solve for meff at each nt
    def meff_equation(meff, nt, C_nt, C_nt_plus_1, N_T):
        # The equation is derived from the ratio C(nt)/C(nt+1) = cosh(meff(nt - N_T/2)) / cosh(meff(nt+1 - N_T/2))
        if boundary == "periodic":
            return C_nt * np.cosh(meff * (nt + 1 - N_T/2)) - C_nt_plus_1 * np.cosh(meff * (nt - N_T/2))
        elif boundary == "anti-periodic":
            return C_nt * np.sinh(meff * (nt + 1 - N_T/2)) - C_nt_plus_1 * np.sinh(meff * (nt - N_T/2))

    # Loop over the time slices and solve for meff
    for nt in range(len(C_t_values) - 1):  # We exclude the last element because C(t+1) won't be available for it
        C_nt = C_t_values[nt]
        C_nt_plus_1 = C_t_values[nt + 1]

        # We need an initial guess for meff, it's a good idea to start with a small positive number
        initial_guess = 1.0

        # Solve the equation using fsolve
        from scipy.optimize import fsolve
        meff, = fsolve(meff_equation, initial_guess, args=(nt, C_nt, C_nt_plus_1, N_T))

        # Store the solution in the meff_values list
        meff_values.append(meff)

    return meff_values

# Example usage:
# N_T = 24  # Example total number of time slices
# C_t_values = [1.0, 0.9, 0.8, 0.7, ...]  # Example data for C(t)
# meff_values = calculate_meff(N_T, C_t_values)

# The above example is just illustrative; you would replace the C_t_values with your actual data.



def pt2_to_meff(pt2_array, boundary="periodic"):
    """
    Convert a given array of pt2 values to an array of meff values.

    Parameters:
    - pt2_array (ndarray): Array of pt2 values.
    - boundary (str, optional): Boundary condition. Can be "periodic" or "anti-periodic" or "none". Defaults to "periodic".

    Returns:
    - meff_array (ndarray): Array of meff values.

    """
    if boundary == "periodic":
        meff_array = np.arccosh( (pt2_array[2:] + pt2_array[:-2]) / (2 * pt2_array[1:-1]) ) #! This is not always correct.

    elif boundary == "anti-periodic":
        meff_array = np.arcsinh((pt2_array[2:] + pt2_array[:-2]) / (2 * pt2_array[1:-1])) #! This is not always correct.

    elif boundary == "none":
        meff_array = np.log(pt2_array[:-1] / pt2_array[1:])

    return meff_array


def pt2_to_meff_solve(pt2_array, boundary="periodic"):
    """
    Convert a given array of pt2 values to an array of meff values.

    Parameters:
    - pt2_array (ndarray): Array of pt2 values.
    - boundary (str, optional): Boundary condition. Can be "periodic" or "anti-periodic". Defaults to "periodic".

    Returns:
    - meff_array (ndarray): Array of meff values.

    """
    if boundary == "periodic":
        Nt = len(pt2_array)
        meff_array = calculate_meff(Nt, pt2_array, boundary="periodic")

    elif boundary == "anti-periodic":
        Nt = len(pt2_array)
        meff_array = calculate_meff(Nt, pt2_array, boundary="anti-periodic")

    elif boundary == "none":
        meff_array = np.log(pt2_array[:-1] / pt2_array[1:])

    return meff_array