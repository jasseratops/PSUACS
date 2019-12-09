import numpy as np
from numpy import sin, cos, tan, pi, exp, log10
import sys
import matplotlib.pyplot as plt
import ACS519_StructInt.soundStructInt as ssint


def main(args):
    part1()
    return 0

def part1():
    c= 343.
    rho_0 = 1.29
    f = third_oct_band(20,10000,center=True)
    f = np.append(f,10**4)
    omega = 2*pi*f

    alpha_0 = absorption_coefficient(f)

    '''
    for i in range(len(f)):
        print str(i) + ": " + str(f[i])
        if not bool((i+1)%3): print "-"
    '''
    ### Panel dimensions
    l1_c = 1.97       # Cavity width
    l2_c = 0.0711     # Cavity Depth
    l3_c = 1.55       # Cavity Height

    ### Room dimensions
    l1_r = 5.
    l2_r = 5.
    l3_r = 5.

    ### Room loss factors
    eta_1 = 0.001
    eta_5 = 0.001

    ### Panel loss factors
    eta_2 = 0.005
    eta_3 = 0.005


    ### Material Properties for Aluminum (6061-T6)
    E_Al = 69.1E9  # Pa
    v_Al = 0.3  #
    rho_Al = 2673.  # kg/m^3

    ### Aluminum Panel dimensions
    l1_panel = l1_c         # Panel Width
    l2_panel = 3.18E-3      # Panel Thickness
    l3_panel = l3_c         # Panel Height

    M_panel = l1_panel*l2_panel*l3_panel*rho_Al

    c_Al, c_Al_Mindlin, c_Al_thin = ssint.plate_phase_speed(l2_panel, v_Al, E_Al, rho_Al, omega, K=5.6, eta=eta_2)
    omega_c = ssint.critical_frq_panel(c_B=c_Al,c_0=c,omega=omega)
    f_c = omega_c/(2*pi)
    print "Critical Frq: " + str(f_c)
    f = np.sort(np.append(f,f_c))
    omega = 2*pi*f
    alpha_0 = absorption_coefficient(f)

    ### cavity loss factor
    eta_3 = cav_loss_factor(f,c,l1_c,l2_c,l3_c,alpha_0)

    R_rad_2,rad_eff_2 = halfSpace_radRes(f,f_c,l1_c,l2_c,l3_c,rho_Al,c)

    eta_rad_2 = R_rad_2/(omega*M_panel)



    plt.figure()
    plt.loglog(f,eta_3)
    plt.xlim(10**2,10**5)
    plt.ylim(10**-3,10**-1)
    plt.grid(which="both")

    plt.figure()
    plt.loglog(f,rad_eff_2)
    plt.xlim(100.,10000.)
    plt.ylim(1.E-3,1.E1)
    plt.grid(which="both")

    plt.show()

    return 0

def part2():
    return 0


def third_oct_band(f_start,f_stop,center=True):
    f_ref = 1000.
    f_points = np.zeros(0)

    ### lower points
    f_c1 = f_ref
    while f_c1 > f_start:
        f_c2 = f_c1
        f_c2_l = f_c2*(2**(-1./6.))

        if center:f_p = f_c2
        else: f_p = f_c2_l

        if f_p > f_start:
            f_points = np.append(f_p,f_points)
        f_c1 = f_c2*(2**(-1./3.))

    ### higher points
    f_c1 = f_ref

    if center: f_points = f_points[0:-1]

    while f_c1 < f_stop:
        f_c2 = f_c1
        f_c2_h = f_c2*(2**(1./6.))

        if center:f_p = f_c2
        else: f_p = f_c2_h

        if f_p < f_stop:
            f_points = np.append(f_points,f_p)

        f_c1 = f_c2*(2**(1./3.))

    print f_points

    return f_points

def coupling_loss_factor(eta_ba,n_a,n_b):
    eta_ab = eta_ba*(n_b/n_a)
    return eta_ab

def cav_loss_factor(f,c,l1,l2,l3,alpha_0):
    omega= 2*pi*f
    S = 2*((l1*l2)+(l2*l3))
    V = l1*l2*l3
    eta_cav = np.zeros(len(omega))
    test_cond = c/(2.*l2)

    for i in range(len(eta_cav)):
        if f[i] < test_cond:
            eta_cav[i] = c*S*alpha_0[i]/(4.*omega[i]*V)
        else:
            eta_cav[i] = c*S*alpha_0[i]/(6.*omega[i]*V)

    return eta_cav

def absorption_coefficient(f):
    frqs = np.array([100,125,160,200,250,315,400,500,630,800,1000,1250,1600,2000])*1.0
    alpha_PC = np.array([.1,.1,.15,.2,.25,.3,.4,.6,.7,.8,.85,.9,.9,.95])
    alpha_0 = np.zeros(len(f))

    for i in range(len(alpha_0)):
        if f[i] < frqs[0]:
            alpha_0[i] = alpha_PC[0]
        elif f[i] > frqs[-1]:
            alpha_0[i] = alpha_PC[-1]
        else:
            x = np.argmin(np.abs(frqs-f[i]))
            alpha_0[i] = alpha_PC[x]

    return alpha_0

def room_modal_density(f,l1,l2,l3,c):
    omega = 2*pi*f
    V = l1*l2*l3
    A = 2*((l1*l2)+(l1*l3)+(l2*l3))
    P = 4*(l1+l2+l3)

    V_modes = V*(omega**2)/(2.*(pi**2)*(c**3))
    A_modes = A*omega/(8.*pi*(c**2))
    P_modes = P/(16.*pi*c)

    n_room = V_modes + A_modes + P_modes

    return n_room

def cavity_modal_density(f,l1,l2,l3,c,n_room):
    test_cond = c/(2*l2)
    omega = 2*pi*f
    n_cavity = np.zeros(len(omega))

    A = 2*((l1*l2)+(l1*l3)+(l2*l3))
    A_modes = A*omega/(8.*pi*(c**2))


    for i in range(len(omega)):
        if f < test_cond:
            n_cavity = A_modes
        else:
            n_cavity = n_room

    return n_cavity

def halfSpace_radRes(f,f_c,l1,l2,l3,rho,c):
    wl_c = c/f_c

    A = 2 * ((l1 * l2) + (l1 * l3) + (l2 * l3))
    P = 4 * (l1 + l2 + l3)
    rad_eff = np.zeros(len(f))
    g_1 = np.zeros(len(f))
    g_2 = np.zeros(len(f))

    fact = A*rho*c

    for i in range(len(rad_eff)):
        wl_a = c / f[i]
        a = (f[i] / f_c) ** (0.5)

        if f[i] < (0.5*f_c):
            g_1[i] = (4/(pi**4))*(1-(2*(a**2)))/(a*(1-(a**2))**(0.5))
        else:
            g_1[i] = 0.

        g_2[i] = ((2*pi)**(-2))*((1-(a**2))*np.log((1+a)/(1-a))+(2*a))/((1-(a**2))**(3./2.))

        if f[i] < f_c:
            rad_eff[i] = (wl_c*wl_a/A)*2*(f[i]/f_c)*g_1[i] + (P*wl_c/A)*g_2[i]
        elif f[i] == f_c:
            rad_eff[i] = ((l1/wl_c)**(0.5))+((l3/wl_c)**(0.5))
        else:
            rad_eff[i] = (1-(f_c/f[i]))**(-0.5)

    R_rad = rad_eff * fact

    return R_rad, rad_eff



if __name__ == "__main__":
    sys.exit(main(sys.argv))