import numpy as np
import matplotlib.pyplot as plt

TAU_C = 20.0                  # cloud optical thickness
OMEGA = 0.95                  # single-scattering albedo
G = 0.8                       # Henyey-Greenstein asymmetry factor

THETA0_DEG = 35.0             # solar zenith angle
PHI0_DEG = 0.0                # solar azimuth angle
OUTPUT_PHI_DEG = [30.0, 90.0, 120.0]

# Because R and T are normalized by (mu0 * F0), F0 cancels.
F0 = 1.0

mu0 = np.cos(np.radians(THETA0_DEG))
phi0 = np.radians(PHI0_DEG)

N_mu = 100
N_phi = 36                    # 10-degree spacing: includes 30, 90, 120 exactly
N_tau = 120
N_order_max = 20
TOL = 1.0e-8

# Viewing-angle grid for mu = cos(theta)
mu = np.linspace(0.02, 1.0, N_mu)   # avoid mu = 0
dmu = mu[1] - mu[0]
w_mu = np.full(N_mu, dmu)
w_mu[0] *= 0.5
w_mu[-1] *= 0.5

# Internal azimuth quadrature grid
phi = np.linspace(0.0, 2.0*np.pi, N_phi, endpoint=False)
dphi = 2.0*np.pi / N_phi

# Optical-depth grid
tau = np.linspace(0.0, TAU_C, N_tau)
dtau = tau[1] - tau[0]

# Trapezoidal weights in tau
w_tau = np.full(N_tau, dtau)
w_tau[0] *= 0.5
w_tau[-1] *= 0.5

def phase_function(cos_theta):
    return (1.0 - G**2) / (1.0 + G**2 - 2.0*G*cos_theta)**1.5

# Angular grids
PHI, MU = np.meshgrid(phi, mu, indexing="ij")  # shape (N_phi, N_mu)

# Flattened angular arrays for matrix-form source updates
phi_flat = PHI.ravel()
mu_flat = MU.ravel()

# Input angular quadrature weights dmu dphi
w_ang = (np.broadcast_to(w_mu, (N_phi, N_mu)) * dphi).ravel()

sqrt_out = np.sqrt(1.0 - mu_flat[:, None]**2)
sqrt_in  = np.sqrt(1.0 - mu_flat[None, :]**2)
cos_dphi = np.cos(phi_flat[:, None] - phi_flat[None, :])

cos_same = (
    mu_flat[:, None] * mu_flat[None, :]
    + sqrt_out * sqrt_in * cos_dphi
)
cos_opp = (
    -mu_flat[:, None] * mu_flat[None, :]
    + sqrt_out * sqrt_in * cos_dphi
)

P_same = phase_function(cos_same)
P_opp = phase_function(cos_opp)

# Source-function kernels:
# J_up   = omega/(4pi) * [ P_same * I_up + P_opp * I_down ]
# J_down = omega/(4pi) * [ P_opp  * I_up + P_same * I_down ]
K_same = P_same * w_ang[None, :]
K_opp  = P_opp  * w_ang[None, :]

ALPHA = OMEGA / (4.0 * np.pi)

D = tau[None, :] - tau[:, None]   # D[k,j] = tau_j - tau_k

U_ops = np.zeros((N_mu, N_tau, N_tau))
L_ops = np.zeros((N_mu, N_tau, N_tau))

for i, mu_i in enumerate(mu):
    U = np.zeros_like(D)
    L = np.zeros_like(D)

    mask_up = D >= 0.0
    mask_dn = D <= 0.0

    U[mask_up] = np.exp(-D[mask_up] / mu_i) / mu_i
    L[mask_dn] = np.exp( D[mask_dn] / mu_i) / mu_i   # since D <= 0 here

    U_ops[i] = U * w_tau[None, :]
    L_ops[i] = L * w_tau[None, :]

def propagate_upward(J):
    """
    J shape: (N_phi, N_mu, N_tau)
    returns I_up with same shape
    """
    I = np.empty_like(J)
    for i in range(N_mu):
        I[:, i, :] = J[:, i, :] @ U_ops[i].T
    return I

def propagate_downward(J):
    """
    J shape: (N_phi, N_mu, N_tau)
    returns I_down with same shape
    """
    I = np.empty_like(J)
    for i in range(N_mu):
        I[:, i, :] = J[:, i, :] @ L_ops[i].T
    return I

beam_attenuation = np.exp(-tau / mu0)

cos_theta_R0 = (
    -mu_flat * mu0
    + np.sqrt(1.0 - mu_flat**2) * np.sqrt(1.0 - mu0**2) * np.cos(phi_flat - phi0)
)
cos_theta_T0 = (
    mu_flat * mu0
    + np.sqrt(1.0 - mu_flat**2) * np.sqrt(1.0 - mu0**2) * np.cos(phi_flat - phi0)
)

J1_up_flat = (ALPHA * F0 * phase_function(cos_theta_R0))[:, None] * beam_attenuation[None, :]
J1_dn_flat = (ALPHA * F0 * phase_function(cos_theta_T0))[:, None] * beam_attenuation[None, :]

J_up_prev = J1_up_flat.reshape(N_phi, N_mu, N_tau)
J_dn_prev = J1_dn_flat.reshape(N_phi, N_mu, N_tau)

I_up_prev = propagate_upward(J_up_prev)
I_dn_prev = propagate_downward(J_dn_prev)

# Accumulate reflection and transmission functions
R_all = np.pi * I_up_prev[:, :, 0] / (mu0 * F0)
T_all = np.pi * I_dn_prev[:, :, -1] / (mu0 * F0)

for order in range(2, N_order_max + 1):
    I_up_flat = I_up_prev.reshape(N_phi * N_mu, N_tau)
    I_dn_flat = I_dn_prev.reshape(N_phi * N_mu, N_tau)

    J_up_flat = ALPHA * (K_same @ I_up_flat + K_opp @ I_dn_flat)
    J_dn_flat = ALPHA * (K_opp  @ I_up_flat + K_same @ I_dn_flat)

    J_up_new = J_up_flat.reshape(N_phi, N_mu, N_tau)
    J_dn_new = J_dn_flat.reshape(N_phi, N_mu, N_tau)

    I_up_new = propagate_upward(J_up_new)
    I_dn_new = propagate_downward(J_dn_new)

    dR = np.pi * I_up_new[:, :, 0] / (mu0 * F0)
    dT = np.pi * I_dn_new[:, :, -1] / (mu0 * F0)

    R_all += dR
    T_all += dT

    max_increment = max(np.max(np.abs(dR)), np.max(np.abs(dT)))
    print(f"order = {order:2d}, max boundary increment = {max_increment:.3e}")

    I_up_prev = I_up_new
    I_dn_prev = I_dn_new

    if max_increment < TOL:
        print(f"Converged at order {order}.")
        break

dphi_deg = 360.0 / N_phi
phi_index = {}
for phi_deg in OUTPUT_PHI_DEG:
    idx_float = phi_deg / dphi_deg
    if abs(idx_float - round(idx_float)) > 1.0e-12:
        raise ValueError(
            "Chosen N_phi does not place the requested output azimuths on the grid. "
            "Use an N_phi compatible with 30, 90, and 120 degrees."
        )
    phi_index[phi_deg] = int(round(idx_float)) % N_phi

plt.figure(figsize=(8, 5))
for phi_deg in OUTPUT_PHI_DEG:
    idx = phi_index[phi_deg]
    plt.plot(mu, R_all[idx, :], label=fr'$\phi={phi_deg:.0f}^\circ$')

plt.xlabel(r'$\mu=\cos\theta$')
plt.ylabel(r'$R(\mu,\phi,\mu_0,\phi_0)$')
plt.title('Reflection Function')
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.figure(figsize=(8, 5))
for phi_deg in OUTPUT_PHI_DEG:
    idx = phi_index[phi_deg]
    plt.plot(mu, T_all[idx, :], label=fr'$\phi={phi_deg:.0f}^\circ$')

plt.xlabel(r'$\mu=\cos\theta$')
plt.ylabel(r'$T(\mu,\phi,\mu_0,\phi_0)$')
plt.title('Transmission Function')
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.show()