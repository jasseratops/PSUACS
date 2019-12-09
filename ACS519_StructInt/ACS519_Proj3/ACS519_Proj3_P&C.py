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
    f = third_oct_band(20,10000)
    alpha_0 = absorption_coefficient(f)

    '''
    for i in range(len(f)):
        print str(i) + ": " + str(f[i])
        if not bool((i+1)%3): print "-"
    '''
    ### Panel dimensions
    l1_c = 1.97       # Panel width
    l2_c = 0.0711     # Cavity Depth
    l3_c = 1.55       # Panel Height

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

    ### cavity loss factor
    eta_3 = cav_loss_factor(f,c,l1_c,l2_c,l3_c,alpha_0)

    plt.figure()
    plt.loglog(f,eta_3)
    plt.xlim(10**2,10**5)
    plt.ylim(10**-3,10**-1)
    plt.grid(which="both")



    plt.show()

    return 0

def part2():
    return 0

def third_oct_band(f_start,f_stop):
    f_ref = 1000.
    f_points = np.zeros(0)

    ### lower points
    f_c1 = f_ref
    while f_c1 > f_start:
        f_c2 = f_c1
        f_c2_l = f_c2*(2**(-1./6.))
        if f_c2_l > f_start:
            f_points = np.append(f_c2_l,f_points)
        f_c1 = f_c2*(2**(-1./3.))

    ### higher points
    f_c1 = f_ref
    while f_c1 < f_stop:
        f_c2 = f_c1
        f_c2_h = f_c2*(2**(1./6.))
        if f_c2_h < f_stop:
            f_points = np.append(f_points,f_c2_h)
        f_c1 = f_c2*(2**(1./3.))

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



if __name__ == "__main__":
    sys.exit(main(sys.argv))