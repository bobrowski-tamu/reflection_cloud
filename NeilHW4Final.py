import time
import numpy as np
import matplotlib.pyplot as plt

# Variables: 
TAU_C = 20.0
OMEGA = 0.95
G = 0.8
THETA0_DEG = 35.0
PHI0_DEG = 0.0
#Don't need but but for clarity 
F0 = 1361.0

mu0 = np.cos(np.radians(THETA0_DEG))
phi0 = np.radians(PHI0_DEG)

# Numerical grids
N_tau = 600
tau = np.linspace(0.0, TAU_C, N_tau)

N_mu = 240
MU_MIN = 0.01
u = np.linspace(0.0, 1.0, N_mu)
mu = MU_MIN + (1.0 - MU_MIN) * u**2

delta_tau = np.diff(tau)

dmu = np.empty_like(mu)
dmu[0] = 0.5 * (mu[1] - mu[0])
dmu[-1] = 0.5 * (mu[-1] - mu[-2])
dmu[1:-1] = 0.5 * (mu[2:] - mu[:-2])

# Azimuthal angles:
phi_out_deg = [30.0, 90.0, 120.0]

TRANSMISSION_MODE = "directional"

PHI_STEP_DEG = 10.0
phi_int_deg = np.arange(0.0, 360.0, PHI_STEP_DEG)
phi_int = np.radians(phi_int_deg)
dphi = np.radians(PHI_STEP_DEG)

p_out_targets = {}
for pdeg in phi_out_deg:
    phi_match_idx = np.where(np.isclose(phi_int_deg, pdeg))[0]
    if len(phi_match_idx) == 0:
        raise ValueError(
            f"phi_out_deg={pdeg} is not on the internal grid for PHI_STEP_DEG={PHI_STEP_DEG}."
        )
    p_out_targets[int(phi_match_idx[0])] = pdeg

# Scattering orders
N_order = 15

# Phase function and geometry
def phase_function(cos_theta):
    return (1.0 - G**2) / (4.0 * np.pi * (1.0 + G**2 - 2.0 * G * cos_theta) ** 1.5)
#Reflection Angle: 
def cos_theta_reflection(mu_out, phi_out_val, mu_in, phi_in):
    return (
        -mu_out * mu_in
        + np.sqrt(1.0 - mu_out**2) * np.sqrt(1.0 - mu_in**2) * np.cos(phi_out_val - phi_in)
    )
#Transmission Angle: 
def cos_theta_transmission(mu_out, phi_out_val, mu_in, phi_in):
    return (
        mu_out * mu_in
        + np.sqrt(1.0 - mu_out**2) * np.sqrt(1.0 - mu_in**2) * np.cos(phi_out_val - phi_in)
    )

# Source term, reflection and transmission angles: 
def J1_up(mu_val, phi_val, tau_val):
    cos_th = cos_theta_reflection(mu_val, phi_val, mu0, phi0)
    return OMEGA * F0 * phase_function(cos_th) * np.exp(-tau_val / mu0)

def J1_down(mu_val, phi_val, tau_val):
    cos_th = cos_theta_transmission(mu_val, phi_val, mu0, phi0)
    return OMEGA * F0 * phase_function(cos_th) * np.exp(-tau_val / mu0)

def upward_intensity_at_boundary(mu_val, J_tau):
    result = 0.0
    for k in range(len(tau) - 1):
        delta = tau[k + 1] - tau[k]
        exponent = -tau[k + 1] / mu_val
        if exponent > -700.0:
            step_exp = np.exp(-delta / mu_val)
            result += J_tau[k + 1] * np.exp(exponent) * (np.exp(delta / mu_val) - 1.0)
    return result

def downward_intensity_at_boundary(mu_val, J_tau):
    result = 0.0
    for k in range(1, len(tau)):
        a = np.exp(-(TAU_C - tau[k]) / mu_val)
        b = np.exp(-(TAU_C - tau[k - 1]) / mu_val)
        j_cell = 0.5 * (J_tau[k] + J_tau[k - 1])
        result += j_cell * (a - b)
    return result

def intensity_profile_upward(mu_val, J_tau):
    I = np.zeros_like(tau)
    I[-1] = 0.0
    for k in range(len(tau) - 2, -1, -1):
        delta = tau[k + 1] - tau[k]
        step_exp = np.exp(-delta / mu_val)
        I[k] = I[k + 1] * step_exp + J_tau[k + 1] * (1.0 - step_exp)
    return I


def intensity_profile_downward(mu_val, J_tau):
    I = np.zeros_like(tau)
    I[0] = 0.0
    for k in range(1, len(tau)):
        delta = tau[k] - tau[k - 1]
        step_exp = np.exp(-delta / mu_val)
        I[k] = I[k - 1] * step_exp + J_tau[k] * (1.0 - step_exp)
    return I


# Successive Order Scattering: 
start = time.time()

R_total = {pdeg: np.zeros_like(mu) for pdeg in phi_out_deg}
T_total = {pdeg: np.zeros_like(mu) for pdeg in phi_out_deg}

n_phi = len(phi_int)
n_mu = len(mu)
n_tau = len(tau)

mu_perp = np.sqrt(1.0 - mu**2)
phi_cos_diff = np.cos(phi_int[:, None] - phi_int[None, :])
mu_weights = dmu * dphi

phase_norm_total = np.zeros((n_phi, n_mu))
for p_out in range(n_phi):
    for i_out, mu_out in enumerate(mu):
        mu_out_perp = mu_perp[i_out]
        row_sum_r = 0.0
        row_sum_t = 0.0
        for p_in in range(n_phi):
            cos_dphi = phi_cos_diff[p_out, p_in]
            cos_r_vec = -mu_out * mu + mu_out_perp * mu_perp * cos_dphi
            cos_t_vec = mu_out * mu + mu_out_perp * mu_perp * cos_dphi
            P_r_vec = phase_function(cos_r_vec)
            P_t_vec = phase_function(cos_t_vec)
            row_sum_r += np.sum(P_r_vec * mu_weights)
            row_sum_t += np.sum(P_t_vec * mu_weights)
        row_sum_total = row_sum_r + row_sum_t
        phase_norm_total[p_out, i_out] = row_sum_total if row_sum_total > 0.0 else 1.0

print(
    "Discrete phase normalization range:",
    f"total_min={np.min(phase_norm_total):.6f}",
    f"total_max={np.max(phase_norm_total):.6f}",
)

I_up_prev = np.zeros((n_phi, n_mu, n_tau))
I_down_prev = np.zeros((n_phi, n_mu, n_tau))

# First order:
for p, phi_val in enumerate(phi_int):
    for i, mu_val in enumerate(mu):
        J_up_tau = J1_up(mu_val, phi_val, tau)
        J_down_tau = J1_down(mu_val, phi_val, tau)

        I_up_b = upward_intensity_at_boundary(mu_val, J_up_tau)
        I_down_b = downward_intensity_at_boundary(mu_val, J_down_tau)

        if p in p_out_targets:
            pdeg = p_out_targets[p]
            R_total[pdeg][i] += np.pi * I_up_b / (mu0 * F0)
            T_total[pdeg][i] += np.pi * I_down_b / (mu0 * F0)

        I_up_prev[p, i, :] = intensity_profile_upward(mu_val, J_up_tau)
        I_down_prev[p, i, :] = intensity_profile_downward(mu_val, J_down_tau)

# Higher orders
for order in range(2, N_order + 1):
    order_start = time.time()
    print(f"Computing order {order}...")
    I_up_new = np.zeros_like(I_up_prev)
    I_down_new = np.zeros_like(I_down_prev)

    for p_out, phi_out_val in enumerate(phi_int):
        for i_out, mu_out in enumerate(mu):
            J_up_tau = np.zeros_like(tau)
            J_down_tau = np.zeros_like(tau)
            mu_out_perp = mu_perp[i_out]
            norm_total = phase_norm_total[p_out, i_out]

            for p_in, phi_in_val in enumerate(phi_int):
                cos_dphi = phi_cos_diff[p_out, p_in]
                cos_r_vec = -mu_out * mu + mu_out_perp * mu_perp * cos_dphi
                cos_t_vec = mu_out * mu + mu_out_perp * mu_perp * cos_dphi
                P_r_vec = phase_function(cos_r_vec)
                P_t_vec = phase_function(cos_t_vec)

                wr = (P_r_vec / norm_total) * mu_weights
                wt = (P_t_vec / norm_total) * mu_weights

                J_up_tau += OMEGA * np.sum(wr[:, None] * I_down_prev[p_in, :, :], axis=0)
                J_down_tau += OMEGA * np.sum(wt[:, None] * I_down_prev[p_in, :, :], axis=0)
                J_up_tau += OMEGA * np.sum(wt[:, None] * I_up_prev[p_in, :, :], axis=0)
                J_down_tau += OMEGA * np.sum(wr[:, None] * I_up_prev[p_in, :, :], axis=0)

            I_up_b = upward_intensity_at_boundary(mu_out, J_up_tau)
            I_down_b = downward_intensity_at_boundary(mu_out, J_down_tau)

            if p_out in p_out_targets:
                pdeg = p_out_targets[p_out]
                R_total[pdeg][i_out] += np.pi * I_up_b / (mu0 * F0)
                T_total[pdeg][i_out] += np.pi * I_down_b / (mu0 * F0)

            I_up_new[p_out, i_out, :] = intensity_profile_upward(mu_out, J_up_tau)
            I_down_new[p_out, i_out, :] = intensity_profile_downward(mu_out, J_down_tau)

    I_up_prev = I_up_new
    I_down_prev = I_down_new
    print(f"  order {order} done in {time.time() - order_start:.2f} s")

elapsed = time.time() - start
print(f"Done in {elapsed:.2f} s")
for pdeg in phi_out_deg:
    print(f"phi={pdeg:.0f} deg: R_max={np.max(R_total[pdeg]):.6e}, T_max={np.max(T_total[pdeg]):.6e}")

#Plotting:
plt.figure(figsize=(8, 5))
for pdeg in phi_out_deg:
    plt.plot(mu, R_total[pdeg], label=rf"$\phi={pdeg:.0f}^\circ$")
plt.xlabel(r"$\mu=\cos\theta$")
plt.ylabel(r"$R(\mu,\phi,\mu_0,\phi_0)$")
plt.title("Bi-Directional Reflection Function")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.figure(figsize=(8, 5))
for pdeg in phi_out_deg:
    plt.plot(mu, T_total[pdeg], label=rf"$\phi={pdeg:.0f}^\circ$")
plt.ylabel(r"$T(\mu,\phi,\mu_0,\phi_0)$")
plt.title("Bi-Directional Transmission Function")
plt.xlabel(r"$\mu=\cos\theta$")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.show()