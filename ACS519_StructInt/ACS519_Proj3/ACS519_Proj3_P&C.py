import numpy as np
from numpy import sin, cos, tan, pi, exp, log10
import numpy.linalg as lin
import sys
import matplotlib.pyplot as plt
import ACS519_StructInt.soundStructInt as ssint



def main(args):
    # eta_p,E_p,v_p,rho_p,l2_p1,l2_p2,c_cav,rho_cav
    part1()
    return 0

def part1():
    eta = loss_factor()
    n = modal_density()
    tl = transmission_loss()

    c_0= 343.
    c_cav=c_0
    rho_0 = 1.29
    f = third_oct_band(20,10000,center=True)
    f = np.append(f,10**4)
    omega = 2*pi*f

    alpha_0 = absorption_coefficient(f)

    ### Cavity dimensions
    l1_c = 1.97       # Cavity width
    l2_c = 0.0711     # Cavity Depth
    l3_c = 1.55       # Cavity Height

    ### Room dimensions
    l1_r = 5.
    l2_r = 5.
    l3_r = 5.

    ### Room loss factors
    eta_1 = 0.001*np.ones_like(f)
    eta_5 = 0.001*np.ones_like(f)

    ### Panel loss factors
    eta_2 = 0.005*np.ones_like(f)
    eta_4 = 0.005*np.ones_like(f)


    ### Material Properties for Aluminum (6061-T6)
    E_p = 69.1E9  # Pa
    v_p = 0.3  #
    rho_p = 2673.  # kg/m^3

    ### Panel 1 dimensions
    l1_p1 = l1_c         # Panel Width
    l2_p1 = 3.18E-3      # Panel Thickness
    l3_p1 = l3_c         # Panel Height

    ### Panel 2 dimensions
    l1_p2 = l1_c         # Panel Width
    l2_p2 = 3.18E-3      # Panel Thickness
    l3_p2 = l3_c         # Panel Height

    M_p1 = l1_p1*l2_p1*l3_p1*rho_p
    M_p2 = l1_p2*l2_p2*l3_p2*rho_p

    c_p1, _, _ = ssint.plate_phase_speed(l2_p1, v_p, E_p, rho_p, omega, K=5.6, eta=eta_2)
    c_p2, _, _ = ssint.plate_phase_speed(l2_p2, v_p, E_p, rho_p, omega, K=5.6, eta=eta_4)

    omega_c_p1 = ssint.critical_frq_panel(c_B=c_p1,c_0=c_0,omega=omega)
    omega_c_p2 = ssint.critical_frq_panel(c_B=c_p2,c_0=c_0,omega=omega)

    f_c_p1 = omega_c_p1/(2.*pi)
    f_c_p2 = omega_c_p2/(2.*pi)

    print "Critical Frq (panel 1): " + str(f_c_p1)
    print "Critical Frq (panel 2): " + str(f_c_p2)

    alpha_0 = absorption_coefficient(f)

    ### cavity loss factor
    eta_3 = eta.cavity(f,c_cav,l1_c,l2_c,l3_c,alpha_0)

    ### Panel radiation efficiency
    R_rad_p1,rad_eff_p1 = halfSpace_radRes(f,f_c_p1,l1_p1,l3_p1,rho_0,c_0)
    R_rad_p2,rad_eff_p2 = halfSpace_radRes(f,f_c_p2,l1_p2,l3_p2,rho_0,c_0)


    ### Model Densities
    n1 = n.room(f,l1_r,l2_r,l3_r,c_0)
    n2 = n.panel(f,c_p1,l1_p1,l3_p1)
    n3 = n.cavity(f,c_0,l1_c,l3_c,l2_c)
    n3_fake = n.room(f,l1_c,l3_c,l2_c,c_cav,cavity=True)
    n4 = n.panel(f,c_p2,l1_p2,l3_p2)
    n5 = n.room(f,l1_r,l2_r,l3_r,c_0)

    TL_13 = tl.rand_inc_TL(f,rho_p,l2_p1,rho_0,c_0)
    TL_53 = tl.rand_inc_TL(f,rho_p,l2_p2,rho_0,c_0)

    A2 = l1_c*l3_c
    V1 = l1_r*l2_r*l3_r

    ### Loss factors
    eta_21 = eta.panel_to_room(f,R_rad_p1,M_p1,f_c_p1)
    eta_23 = eta.panel_to_cav(f, R_rad_p1,M_p1,f_c_p1)
    eta_45 = eta.panel_to_room(f,R_rad_p2,M_p2,f_c_p2)
    eta_43 = eta.panel_to_cav(f, R_rad_p2,M_p2,f_c_p2)
    eta_13 = eta.room_to_cav(f,TL_13,A2,V1,c_0)
    eta_53 = eta.room_to_cav(f,TL_53,A2,V1,c_0)

    ### converse loss fators
    eta_12 = eta.coupling_loss_factor(eta_21,n1,n2)
    eta_32 = eta.coupling_loss_factor(eta_23,n3,n2)
    eta_54 = eta.coupling_loss_factor(eta_45,n5,n4)
    eta_34 = eta.coupling_loss_factor(eta_43,n3,n4)
    eta_31 = eta.coupling_loss_factor(eta_13,n3,n1)
    eta_35 = eta.coupling_loss_factor(eta_53,n3,n5)

    ### filler loss factors
    fil = 0*np.ones_like(f)
    eta_41 = fil
    eta_42 = fil
    eta_51 = fil
    eta_52 = fil
    eta_14 = fil
    eta_24 = fil
    eta_15 = fil
    eta_25 = fil

    ## loss factor sum terms
    eta_11 = eta_1 + (eta_12 + eta_13 + eta_14 + eta_15)
    eta_22 = eta_2 + (eta_21 + eta_23 + eta_24 + eta_25)
    eta_33 = eta_3 + (eta_31 + eta_32 + eta_34 + eta_35)
    eta_44 = eta_4 + (eta_41 + eta_42 + eta_43 + eta_45)
    eta_55 = eta_5 + (eta_51 + eta_52 + eta_53 + eta_54)


    TL_15 = np.zeros_like(f)
    for i in range(len(f)):
        eta_mtx = 2*pi*f[i]* np.array([[eta_11[i], -eta_21[i], -eta_31[i], -eta_41[i], -eta_51[i]],
                                       [-eta_12[i], eta_22[i], -eta_32[i], -eta_42[i], -eta_52[i]],
                                       [-eta_13[i], -eta_23[i], eta_33[i], -eta_43[i], -eta_53[i]],
                                       [-eta_14[i], -eta_24[i], -eta_34[i], eta_44[i], -eta_54[i]],
                                       [-eta_15[i], -eta_25[i], -eta_35[i], -eta_45[i], eta_55[i]]],dtype=complex)

        P_mtx = [1.,0.,0.,0.,0.]

        E_mtx = lin.solve(eta_mtx,P_mtx)
        E_1 = E_mtx[0]
        E_5 = E_mtx[-1]

        TL_15[i] = 10.*log10(E_1/E_5)



    '''
    plt.figure()
    plt.loglog(f,eta_3)
    plt.xlim(10**2,10**5)
    plt.ylim(10**-3,10**-1)
    plt.grid(which="both")
    '''
    plt.figure()
    plt.loglog(f,rad_eff_p1)
    plt.xlim(100.,10000.)
    plt.ylim(1.E-3,1.E1)
    plt.grid(which="both")

    '''
    plt.figure()
    plt.loglog(f,n3,linestyle="-")
    plt.loglog(f,n3_fake)
    plt.xlim(10**2,10**5)
    plt.ylim(1.E-2,1.E0)
    plt.grid(which="both")
    '''
    '''
    plt.figure()
    plt.semilogx(f*rho_Al*l2_panel,TL_13)
    plt.xlim(10,10**6)
    plt.grid(which="both")
    '''


    plt.figure()
    plt.semilogx(f,TL_15)
    plt.xlim(100,10000)
    plt.ylim(0,75)
    plt.grid(which="both")
    '''
    plt.figure()
    plt.loglog(f,eta_12)
    plt.loglog(f,eta_21)
    '''
    plt.ylim(1E-7,1E-1)
    plt.xlim(100,10000)
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

    #print f_points

    return f_points

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

def halfSpace_radRes(f, f_c, l1, l3, rho_0, c_0):
    wl_c = c_0 / f_c

    A = l1 * l3
    P = 2 * (l1+l3)
    rad_eff = np.zeros(len(f))
    g_1 = np.zeros(len(f))
    g_2 = np.zeros(len(f))

    fact = A * rho_0 * c_0

    for i in range(len(rad_eff)):
        wl_a = c_0 / f[i]
        a = (f[i] / f_c)**0.5

        if f[i] < (0.5*f_c):
            g_1[i] = (4./(pi**4))*(1.-(2.*(a**2)))/(a*(1-((a**2))**(0.5)))
        else:
            g_1[i] = 0.

        g_2[i] = ((2*pi)**(-2)) * ((1.-(a**2))*np.log((1+a)/(1-a))+(2*a)) / ((1-(a**2))**(3./2.))


        if f[i] < f_c:
            rad_eff[i] = (wl_c*wl_a/A)*2*(f[i]/f_c)*g_1[i] + (P*wl_c/A)*g_2[i]
        elif f[i] == f_c:
            rad_eff[i] = ((l1/wl_c)**(0.5))+((l3/wl_c)**(0.5))
        else:
            rad_eff[i] = (1-(f_c/f[i]))**(-0.5)

    R_rad = rad_eff * fact

    return R_rad, rad_eff

class loss_factor:
    def __init__(self):
        return None

    def cavity(self, f, c_cav, l1_c, l2_c, l3_c, alpha_0):
        omega = 2 * pi * f
        S = 2 * ((l1_c * l2_c) + (l2_c * l3_c))
        V = l1_c * l2_c * l3_c
        eta_cav = np.zeros(len(omega))
        test_cond = c_cav / (2. * l2_c)

        for i in range(len(eta_cav)):
            if f[i] < test_cond:
                eta_cav[i] = c_cav * S * alpha_0[i] / (4. * omega[i] * V)
            else:
                eta_cav[i] = c_cav * S * alpha_0[i] / (6. * omega[i] * V)

        return eta_cav

    def panel_to_room(self,f,R_rad, M_panel, f_c):
        eta_panelRoom = np.zeros(len(f))
        omega = 2. * pi * f

        for i in range(len(f)):
            if f[i] < f_c:
                eta_panelRoom[i] = 2. * R_rad[i] / (omega[i] * M_panel)
            else:
                eta_panelRoom[i] = R_rad[i] / (omega[i] * M_panel)
        return eta_panelRoom

    def panel_to_cav(self,f, R_rad, M_panel, f_c):
        eta_panelCav = np.zeros(len(f))
        omega = 2. * pi * f

        for i in range(len(f)):
            if f[i] < f_c:
                eta_panelCav[i] = 4. * R_rad[i] / (omega[i] * M_panel)
            else:
                eta_panelCav[i] = R_rad[i] / (omega[i] * M_panel)

        return eta_panelCav

    def room_to_cav(self, f, TL, A_p, V_r, c_0):
        A2 = A_p
        V1 = V_r
        omega = 2.*pi*f
        eta_roomCav = 10.**((-TL+(10.*np.log10(A2*c_0/(4.*V1*omega))))/10.)
        return eta_roomCav

    def coupling_loss_factor(self,eta_ba,n_a,n_b):
        eta_ab = eta_ba*(n_b/n_a)
        return eta_ab

class modal_density:
    def __init__(self):
        return None

    def room(self, f, l1_r, l2_r, l3_r, c_0, cavity=False):
        l1 = l1_r
        l2 = l2_r
        l3 = l3_r

        omega = 2 * pi * f
        V = l1 * l2 * l3
        A = 2 * ((l1 * l2) + (l1 * l3) + (l2 * l3))
        P = 4 * (l1 + l2 + l3)

        V_modes = V*(omega ** 2)/(2.*(pi**2)*(c_0**3))
        A_modes = A * omega / (8. * pi * (c_0 ** 2))
        P_modes = P / (16. * pi * c_0)

        if cavity:
            A_modes = 0
            P_modes = 0

        n_room = V_modes + A_modes + P_modes

        return n_room

    def panel(self, f, c_B, l1_p, l3_p):
        omega = 2*pi*f
        A = l1_p * l3_p
        n_panel = A*omega/(4*pi*(c_B**2))

        return n_panel

    def cavity(self, f, c_cav, l1_c, l3_c, l2_c):
        test_cond = c_cav / (2 * l2_c)
        omega = 2*pi*f
        n_cavity = np.zeros(len(omega))

        n_room = self.room(f, l3_c, l1_c, l2_c, c_cav, cavity=True)

        A = l1_c * l3_c

        for i in range(len(omega)):
            if f[i] < test_cond:
                n_cavity[i] = (A*omega[i])/(2 * pi * (c_cav**2))
            else:
                n_cavity[i] = n_room[i]

        return n_cavity

class transmission_loss:
    def __init__(self):
        return None

    def zero_inc_TL(self, f, rho_p, l2_p, rho_0, c_0):
        rho_s = rho_p * l2_p
        rho = rho_0
        c = c_0
        omega = 2.*pi*f

        inner = 1. + ((omega * rho_s / (2 * rho * c))**2)
        TL_0 = 10.*np.log10(inner)

        return TL_0

    def rand_inc_TL(self, f, rho_p, l2_p, rho_0, c_0):
        TL_0 = self.zero_inc_TL(f, rho_p, l2_p, rho_0, c_0)

        TL_rand = TL_0 - 10*np.log10(0.23*TL_0)

        return TL_rand

if __name__ == "__main__":
    sys.exit(main(sys.argv))