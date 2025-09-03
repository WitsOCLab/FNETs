import torch
import numpy as np

import torch
   

def complex_pearson_torch(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """
    Computes the complex Pearson correlation by first calculating the 
    4x4 real-valued covariance matrix of the constituent real and imaginary parts.

    This is a demonstration of how to build the complex statistic from real-valued tools.

    Args:
        z1 (torch.Tensor): The first complex-valued tensor.
        z2 (torch.Tensor): The second complex-valued tensor.

    Returns:
        torch.Tensor: A scalar complex tensor representing the correlation coefficient.
    """
    if z1.shape != z2.shape:
        raise ValueError("Input tensors must have the same shape")
    if not z1.is_complex() or not z2.is_complex():
        raise ValueError("Input tensors must be complex")
        
    # Flatten the data
    z1_flat = z1.flatten()
    z2_flat = z2.flatten()
    
    # Stack the real and imaginary parts into a (4, N) tensor
    # Order: [Re(Z1), Im(Z1), Re(Z2), Im(Z2)]
    stacked_re_im = torch.stack([
        z1_flat.real,
        z1_flat.imag,
        z2_flat.real,
        z2_flat.imag
    ])
    
    # Compute the 4x4 covariance matrix
    cov_matrix = torch.cov(stacked_re_im)
    
    # Extract the necessary covariance terms from the matrix
    cov_re1_re2 = cov_matrix[0, 2]
    cov_im1_im2 = cov_matrix[1, 3]
    cov_im1_re2 = cov_matrix[1, 2]
    cov_re1_im2 = cov_matrix[0, 3]
    
    # Extract the variances (diagonal elements)
    var_re1 = cov_matrix[0, 0]
    var_im1 = cov_matrix[1, 1]
    var_re2 = cov_matrix[2, 2]
    var_im2 = cov_matrix[3, 3]
    
    # Assemble the complex covariance
    complex_cov = (cov_re1_re2 + cov_im1_im2) + 1j * (cov_im1_re2 - cov_re1_im2)
    
    # Calculate the variances of the complex variables
    # Var(Z) = Var(Re(Z)) + Var(Im(Z))
    # This is a known property for zero-mean circular complex variables,
    # and holds for centered variables in general.
    var_z1 = var_re1 + var_im1
    var_z2 = var_re2 + var_im2
    
    # Handle zero variance
    if var_z1 == 0 or var_z2 == 0:
        return torch.tensor(0.0 + 0.0j)
        
    # Assemble the complex correlation coefficient
    rho_c = complex_cov / torch.sqrt(var_z1 * var_z2)
    
    return rho_c


def complex_pearson_numpy(z1, z2):
    """
    Correctly computes the complex Pearson correlation coefficient using NumPy.
    This function implements the standard definition with conjugation.
    """
    z1_flat = z1.flatten()
    z2_flat = z2.flatten()
    
    mean1 = np.mean(z1_flat)
    mean2 = np.mean(z2_flat)
    
    centered1 = z1_flat - mean1
    centered2 = z2_flat - mean2
    
    # Correct covariance with conjugation
    covariance = np.mean(centered1 * np.conj(centered2))
    
    # Variances
    variance1 = np.mean(np.abs(centered1)**2)
    variance2 = np.mean(np.abs(centered2)**2)
    
    if variance1 == 0 or variance2 == 0:
        return np.complex128(0)
        
    return covariance / np.sqrt(variance1 * variance2)

def pearson_correlation(z1: torch.Tensor|np.ndarray, z2: torch.Tensor|np.ndarray):
    if isinstance(z1,torch.Tensor):
        if not isinstance(z2, torch.Tensor):
            z2 = torch.tensor(z2)
        return complex_pearson_torch(z1,z2)
    elif isinstance(z1,np.ndarray):
        if isinstance(z2, torch.Tensor):
            z2 = z2.detach().numpy()
        return complex_pearson_numpy(z1,z2)
    else:
        raise ValueError(f"z1 and z2 must be of correct type, not {type(z1)} and {type(z2)}")


if __name__ == "__main__":
    a = [4-1j, 4+1j]
    b = [4+1j, 4+0.2j]

    a = np.array(a)
    b = np.array(b)

    print(pearson_correlation(a,b))
    print(pearson_correlation(a,a))

    a = torch.tensor(a)
    b = torch.tensor(b)

    print(pearson_correlation(a,b))
    print(pearson_correlation(a,a))