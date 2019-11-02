# PSUACS
# soundStructInt
# Jasser Alshehri
# Starkey Hearing Technologies
# 10/5/2019


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp


def plate_phase_speed(h,v,E,rho,omega,K=5.6,eta = 0.):
    I_prime = plate_mom_inertia_wid(h)
    G = structural_shear_mod(E,v)
    D = flex_rig(E,h,v,eta)
    KhG = K*h*G
    D_KhG = D/KhG

    A = np.sqrt(((D_KhG-(I_prime/h))**2)*(omega**4) + 4*(D/(rho*h))*(omega**2))
    B = (omega**2)*(D_KhG + (I_prime/h))
    C = 2.*(1-(omega**2)*(I_prime*rho/KhG))

    c_B = np.sqrt((A-B)/C)
    c_B_Mindlin = ((D/(rho*h))*(omega**2))**(1./4.)
    c_B_thin = np.ones_like(omega)*shear_speed(E,rho,K,v)
    return c_B, c_B_Mindlin, c_B_thin

def shear_speed(E,rho,K,v):
    G = structural_shear_mod(E,v)
    c_S = np.sqrt(K * G / rho)
    return c_S

def structural_shear_mod(E,v):
    G = E / (2. * (1. + v))
    return G

def flex_rig(E,h,v,eta = 0.):
    I_prime = plate_mom_inertia_wid(h)
    D = E * (h ** 3) / (12. * (1. - (v ** 2)))
    D_comp = D*(1 + (-1j*eta))
    return D_comp

def plate_mom_inertia_wid(h):
    I_prime = (h**3)/12.
    return I_prime

def ss_flatplate_frq(D_comp,rho,h,m,a,n,b):
    omega_comp = np.sqrt(D_comp/(rho*h))*(((m*pi/a)**2)+(n*pi/b)**2)
    return omega_comp

def ss_thickplate_frq(c_B,omega,m,n,a,b):
    k_m = (m*pi/a)
    k_n = (n*pi/b)
    k_mn = np.ones_like(omega,dtype=complex)*np.sqrt((k_m**2)+(k_n**2))

    k_B = np.zeros_like(omega,dtype=complex)
    #print k_B

    for i in range(len(k_B)):
        k_B[i] = omega[i]/c_B[i]

    k_B_mn_diff = np.abs(k_B-k_mn)

    i = np.argmin(k_B_mn_diff)
    omega_mn_thick = k_B[i]*c_B[i]

    return omega_mn_thick

def tot_drivePoint_mobility(x_r,x_f,y_r,y_f,m_points,n_points,a,b,D,rho,h,omega,m_mn,plot=False):
    v_F_tot = np.zeros_like(omega,dtype=complex)
    for m in range(1,m_points):
        for n in range(1,n_points):
            A_mn = drivePoint_panelShape(x_r, x_f, y_r, y_f, m, n, a, b)
            omega_mn = ss_flatplate_frq(D, rho, h, m, n, a, b)
            v_F_tot += drivePoint_mobility(A_mn,omega_mn, m_mn, omega)
    return v_F_tot

def drivePoint_mobility(A_mn,omega_mn,m_mn,omega):
    v_F_mn = (-1j * omega / m_mn) * A_mn / ((omega_mn ** 2) - (omega ** 2))
    return v_F_mn

def inf_plateMob(D,rho,h):
    v_F = 1./(8.*np.sqrt(D*rho*h))
    return v_F

def Mode_Mesh(A_mn,a,b,m,n):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    X = np.linspace(0, a, int(1024*a))
    Y = np.linspace(0, b, int(1024*b))

    X, Y = np.meshgrid(X, Y)
    Z = A_mn*sin(m*pi*X/a)*sin(n*pi*Y/b)
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    ax.set_zlim(np.min(Z),np.max(Z))
    plt.title(str(m)+","+str(n))

    return 0

def drivePoint_panelShape(x_r,x_f,y_r,y_f,m,n,a,b):
    k_m = m*pi/a
    k_n = n*pi/b
    A = sin(k_m*x_r)
    B = sin(k_n*y_r)
    C = sin(k_m*x_f)
    D = sin(k_n*y_f)

    A_mn_DP = A*B*C*D
    return A_mn_DP

def ss_modalMass(rho,h,a,b):
    m_mn = rho*h*a*b/4.
    return m_mn