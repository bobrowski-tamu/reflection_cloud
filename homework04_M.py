import numpy as np
import matplotlib.pyplot as plt

TAU_C = 20.0
OMEGA = 0.95
G = 0.8

THETA0_DEG = 35.0
PHI0_DEG = 0.0

mu0 = np.cos(np.radians(THETA0_DEG))
phi0 = np.radians(PHI0_DEG)

#set steps for numerical calculation
N_mu = 100
N_tau = 100
N_order = 20

mu = np.linspace(0.02, 1.0, N_mu)      # avoid mu = 0
tau = np.linspace(0.0, TAU_C, N_tau)

dmu = mu[1] - mu[0]
dtau = tau[1] - tau[0]

phi_list_deg = [30.0, 90.0, 120.0]      #convert to radians
phi_list = np.radians(phi_list_deg)

# Phase function
def phase_function(cos_theta):
    return (1.0 - G**2) / (1.0 + G**2 - 2.0*G*cos_theta)**1.5

# Scattering angle cosines
def cos_theta_R(mu_val, phi_val):
    return -mu_val*mu0 + np.sqrt(1.0 - mu_val**2)*np.sqrt(1.0 - mu0**2)*np.cos(phi_val - phi0)

def cos_theta_T(mu_val, phi_val):
    return  mu_val*mu0 + np.sqrt(1.0 - mu_val**2)*np.sqrt(1.0 - mu0**2)*np.cos(phi_val - phi0)

# First-order source functions
def J1_up(mu_val, phi_val, tau_val):
    c = cos_theta_R(mu_val, phi_val)
    P = phase_function(c)
    return (OMEGA / (4.0*np.pi)) * np.exp(-tau_val / mu0) * P

def J1_down(mu_val, phi_val, tau_val):
    c = cos_theta_T(mu_val, phi_val)
    P = phase_function(c)
    return (OMEGA / (4.0*np.pi)) * np.exp(-tau_val / mu0) * P

# Formal solutions
def upward_intensity_from_source(mu_val, J_tau):
    # I_up(0) = (1/mu) int_0^TAU_C J(t) exp(-t/mu) dt
    return np.sum((1.0/mu_val) * J_tau * np.exp(-tau/mu_val)) * dtau

def downward_intensity_from_source(mu_val, J_tau):
    # I_down(TAU_C) = (1/mu) int_0^TAU_C J(t) exp(-(TAU_C-t)/mu) dt
    return np.sum((1.0/mu_val) * J_tau * np.exp(-(TAU_C - tau)/mu_val)) * dtau

# First-order intensities
R_total = {phi_deg: np.zeros_like(mu) for phi_deg in phi_list_deg}
T_total = {phi_deg: np.zeros_like(mu) for phi_deg in phi_list_deg}

# Store all orders as functions of tau, mu, phi-index
I_up_prev = np.zeros((len(phi_list), len(mu), len(tau)))
I_down_prev = np.zeros((len(phi_list), len(mu), len(tau)))

# First order
for p, phi_val in enumerate(phi_list):
    for i, mu_val in enumerate(mu):
        J_up_tau = J1_up(mu_val, phi_val, tau)
        J_down_tau = J1_down(mu_val, phi_val, tau)

        I_up_boundary = upward_intensity_from_source(mu_val, J_up_tau)
        I_down_boundary = downward_intensity_from_source(mu_val, J_down_tau)

        R_total[phi_list_deg[p]][i] += np.pi * I_up_boundary / mu0
        T_total[phi_list_deg[p]][i] += np.pi * I_down_boundary / mu0

        # save source-shaped profiles for building higher orders
        I_up_prev[p, i, :] = (1.0/mu_val) * np.array([
            np.sum(J_up_tau[k:] * np.exp(-(tau[k:] - tau[k])/mu_val)) * dtau
            for k in range(len(tau))
        ])
        I_down_prev[p, i, :] = (1.0/mu_val) * np.array([
            np.sum(J_down_tau[:k+1] * np.exp(-(tau[k] - tau[:k+1])/mu_val)) * dtau
            for k in range(len(tau))
        ])

# Higher orders: use 30, 90, 120 only and integrate over mu
for order in range(2, N_order + 1):
    I_up_new = np.zeros_like(I_up_prev)
    I_down_new = np.zeros_like(I_down_prev)

    for p_out, phi_out in enumerate(phi_list):
        for i_out, mu_out in enumerate(mu):

            J_up_tau = np.zeros_like(tau)
            J_down_tau = np.zeros_like(tau)

            for p_in, phi_in in enumerate(phi_list):
                for i_in, mu_in in enumerate(mu):

                    # reflection geometry
                    cosR = -mu_out*mu_in + np.sqrt(1-mu_out**2)*np.sqrt(1-mu_in**2)*np.cos(phi_out - phi_in)
                    PR = phase_function(cosR)

                    # transmission geometry
                    cosT = mu_out*mu_in + np.sqrt(1-mu_out**2)*np.sqrt(1-mu_in**2)*np.cos(phi_out - phi_in)
                    PT = phase_function(cosT)

                    J_up_tau += (OMEGA / (4.0*np.pi)) * PR * I_down_prev[p_in, i_in, :] * dmu
                    J_down_tau += (OMEGA / (4.0*np.pi)) * PT * I_down_prev[p_in, i_in, :] * dmu

                    J_up_tau += (OMEGA / (4.0*np.pi)) * PT * I_up_prev[p_in, i_in, :] * dmu
                    J_down_tau += (OMEGA / (4.0*np.pi)) * PR * I_up_prev[p_in, i_in, :] * dmu

            I_up_boundary = upward_intensity_from_source(mu_out, J_up_tau)
            I_down_boundary = downward_intensity_from_source(mu_out, J_down_tau)

            R_total[phi_list_deg[p_out]][i_out] += np.pi * I_up_boundary / mu0
            T_total[phi_list_deg[p_out]][i_out] += np.pi * I_down_boundary / mu0

            I_up_new[p_out, i_out, :] = (1.0/mu_out) * np.array([
                np.sum(J_up_tau[k:] * np.exp(-(tau[k:] - tau[k])/mu_out)) * dtau
                for k in range(len(tau))
            ])
            I_down_new[p_out, i_out, :] = (1.0/mu_out) * np.array([
                np.sum(J_down_tau[:k+1] * np.exp(-(tau[k] - tau[:k+1])/mu_out)) * dtau
                for k in range(len(tau))
            ])

    I_up_prev = I_up_new.copy()
    I_down_prev = I_down_new.copy()

# Plot Reflection
plt.figure(figsize=(8,5))
for phi_deg in phi_list_deg:
    plt.plot(mu, R_total[phi_deg], label=fr'$\phi={phi_deg:.0f}^\circ$')
plt.xlabel(r'$\mu=\cos\theta$')
plt.ylabel(r'$R(\mu,\phi,\mu_0,\phi_0)$')
plt.title(f'Reflection Function ')
plt.grid(True)
plt.legend()
plt.tight_layout()

# Plot Transmission
plt.figure(figsize=(8,5))
for phi_deg in phi_list_deg:
    plt.plot(mu, T_total[phi_deg], label=fr'$\phi={phi_deg:.0f}^\circ$')
plt.xlabel(r'$\mu=\cos\theta$')
plt.ylabel(r'$T(\mu,\phi,\mu_0,\phi_0)$')
plt.title('Transmission Function')
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.show()