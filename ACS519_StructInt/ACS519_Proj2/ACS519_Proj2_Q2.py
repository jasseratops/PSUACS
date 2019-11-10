# PSUACS
# ACS519_Proj2_Q2
# Jasser Alshehri
# Starkey Hearing Technologies
# 11/9/2019


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp
import ACS519_StructInt.soundStructInt as ssint
import xlwt
from xlwt import Workbook


def main(args):
    ### Excel setup for table
    wb = Workbook()
    sheet1 = wb.add_sheet("Sheet 1")
    sheet1.write(0,0,'m')
    sheet1.write(0,1,'n')
    sheet1.write(0,2,"Frequency Al")
    sheet1.write(0,3,"Frequency sw")
    sheet1.write(0,4,"Rad Eff Al")
    sheet1.write(0,5,"Rad Eff sw")

    ### Plot Setup
    f = np.linspace(1,4.E3,1024)
    omega = f*2*pi
    m_array = np.arange(5)+1
    n_array = np.arange(5)+1
    a = 0.5
    b = 0.4

    ### Conversion factors
    pcf_to_kgm3 = 16.02
    in_to_m = 0.0254
    ksi_to_Pa = 6.895E6

    ### Panel Dimensions.
    h_Al = 0.25 * in_to_m
    t_fs = 0.02 * in_to_m
    h_core = 0.5 * in_to_m

    ### Material Properties for Aluminum (6061-T6)
    E_Al = 68.9E9  # Pa
    v_Al = 0.33  #
    rho_Al = 2.7E3  # kg/m^3
    eta_Al = 0.004
    ### Material Properties of Aluminum facesheets
    E_fs = E_Al
    v_fs = v_Al
    rho_fs = rho_Al

    ### Material Properties of Hexcel Core
    ### HRH-10-1/8-9.0
    G_ribbon = 17.5 * ksi_to_Pa
    G_warp = 11.0 * ksi_to_Pa
    G_core = np.sqrt(G_ribbon*G_warp)
    rho_core = 9.0 * pcf_to_kgm3

    ### Panel Properties
    u_s = 2*(rho_fs*t_fs) + rho_core*h_core
    D = flex_rig_sandwich(E_fs,t_fs,h_core,v_fs)
    N = shear_rig_sandwich(G_core,t_fs,h_core)
    c_b_sw = panel_wavespeed_sandwich(N,u_s,D,omega)

    c_Al,c_Al_Mindlin,c_Al_thin = ssint.plate_phase_speed(h_Al,v_Al,E_Al,rho_Al,omega,K=5.6,eta=eta_Al)

    f_Al_array = np.zeros(len(m_array)*len(n_array))
    f_sw_array = np.zeros(len(m_array)*len(n_array))
    rad_mn_Al_array = np.zeros(len(m_array)*len(n_array))
    rad_mn_sw_array = np.zeros(len(m_array)*len(n_array))

    c_0 = 343.
    k = omega/c_0
    f_cr_Al = critical_frq_panel(c_Al,c_0,omega)/(2*pi)
    f_cr_sw = critical_frq_sandwich(D,N,u_s,c_0)/(2*pi)

    print "f_cr_Al: " + str(f_cr_Al)
    print "f_cr_sw: " + str(f_cr_sw)
    print "m,n|fAL|fSw|re_Al|re_Sw"
    print "-"*23

    for i_m in range(len(m_array)):
        m = m_array[i_m]
        for i_n in range(len(n_array)):
            n = n_array[i_n]
            i = i_m*len(n_array)+i_n

            ### Aluminum calculations
            w_Al = ssint.ss_thickplate_frq(c_Al,omega,m,n,a,b)
            f_Al = w_Al.real/(2*pi)
            rad_mn_Al = rad_eff_Wallace_gauss(a,b,m,n,N=64,k=w_Al/c_0)[0].real

            f_Al_array[i] = f_Al
            rad_mn_Al_array[i] = rad_mn_Al

            ### Sandwich panel calculations
            w_sw = ssint.ss_thickplate_frq(c_b_sw,omega,m,n,a,b)
            f_sw = w_sw.real/(2*pi)
            rad_mn_sw = rad_eff_Wallace_gauss(a,b,m,n,N=64,k=w_sw/c_0)[0].real

            f_sw_array[i] = f_sw
            rad_mn_sw_array[i] = rad_mn_sw

            ### Excel write:
            sheet1.write(i+1, 0,m)
            sheet1.write(i+1, 1,n)
            sheet1.write(i+1, 2,f_Al)
            sheet1.write(i+1, 3,f_sw)
            sheet1.write(i+1, 4,str(rad_mn_Al))
            sheet1.write(i+1, 5,str(rad_mn_sw))
            ### Table print
            print str(m)+","+str(n)+","\
                  +str(int(round(f_Al)))+","+str(int(round(f_sw)))+","\
                  +str('%.3f'%rad_mn_Al)+","+str('%.3f'%rad_mn_sw)

    #wb.save("ACS519_Proj2_Q2.xls")
    plt.figure()
    plt.title("Radiation Efficiency, Panel vs. Sandwich")
    plt.scatter(f_Al_array,rad_mn_Al_array,
                marker="s",edgecolors="black",facecolors="black",label="Aluminum Panel")
    plt.scatter(f_sw_array,rad_mn_sw_array,
                marker="s",edgecolors="black",facecolors="none",label="Sandwich Panel")

    plt.axvline(f_cr_Al,color="black",label=r"Aluminum Panel $f_{cr}$")
    plt.axvline(f_cr_sw,color="black",linestyle=":",label=r"Sandwich Panel $f_{cr}$")
    plt.grid(True, which="both", axis="both")
    plt.xlim(0,4.E3)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Radiation Efficiency")
    plt.legend()
    plt.show()

    return 0

def flex_rig_sandwich(E_fs,t_fs,h_core,v_fs):
    D = E_fs*t_fs*((h_core+t_fs)**2)/(2*((1-(v_fs**2))))
    return D

def shear_rig_sandwich(G_core,t_fs,h_core):
    N = G_core*h_core*((1+(t_fs/h_core))**2)
    return N

def panel_wavespeed_sandwich(N,u_s,D,omega):
    num = 2*N
    denom_a = u_s
    denom_b = np.sqrt((u_s**2)+((4*u_s*(N**2))/((omega**2)*D)))
    denom = denom_a+denom_b

    c_b_2 = num/denom
    c_b = np.sqrt(c_b_2)
    return c_b

def critical_frq_panel(c_B,c_0,omega):
    i = np.argmin(np.abs(c_B-c_0))
    omega_cr = omega[i]
    return omega_cr

def critical_frq_sandwich(D,N,u_s,c_0):
    omega_cr_2 = ((c_0**4)*u_s/D)*(1./((1.-((c_0**2)*u_s/N))))
    omega_cr = np.sqrt(omega_cr_2)
    return omega_cr

def rad_eff_Wallace_gauss(a,b,m,n,N,k):
    S_mn = np.zeros_like(k)

    x_i_array, w_i_array = gauleg(0.,pi/2.,N)
    x_j_array, w_j_array = gauleg(0.,pi/2.,N)
    '''
    plt.figure()
    plt.plot(x_i_array,w_i_array)
    plt.plot(x_j_array,w_j_array)
    '''
    A = 64.*(k**2)*a*b/((pi**6)*(m*n)**2)

    k_m = m*pi/a
    k_n = n*pi/b
    k_mn = np.sqrt((k_m**2)+(k_n**2))   # modal wavenumber

    gamma = k/k_mn      # wavenumber ratio

    m_odd = bool(m%2)
    n_odd = bool(n%2)


    #print N
    for i in range(N):
        x_i = x_i_array[i]
        w_i = w_i_array[i]

        for j in range(N):
            #print "(" + str(i) + "," + str(j) + "):"
            x_j = x_j_array[j]
            w_j = w_j_array[j]

            alpha = k*a*sin(x_i)*cos(x_j)
            beta = k*b*sin(x_i)*sin(x_j)
            '''
            print alpha/(m*pi)
            print beta/(n*pi)
            '''
            if m_odd:
                num_a = cos(alpha/2.)
            else:
                num_a = sin(alpha/2.)

            if n_odd:
                num_b = cos(beta/2.)
            else:
                num_b = sin(beta/2.)

            num = num_a*num_b
            denom = (((alpha/(m*pi))**2)-1)*(((beta/(n*pi))**2)-1)
            g_x = A*((num/denom)**2)

            S_mn += w_i*w_j*g_x*sin(x_i)


    return S_mn, gamma

def gauleg(X1,X2,N):
    eps = 3e-14
    M = (N+1)/2

    XM = 0.5*(X2+X1)
    XL = 0.5*(X2-X1)

    X = np.zeros(N)
    W = np.zeros(N)

    for i in range(M):
        i+=1
        Z = cos(pi*(i-0.25)/(N+0.5))
        Z1 = -100000

        while np.abs(Z-Z1) > eps:
            P1 = 1
            P2 = 0
            for j in range(N):
                j+=1
                P3 = P2
                P2 = P1
                P1 = ((2*j-1)*Z*P2-(j-1)*P3)/j
            PP = N*(Z*P1-P2)/(Z*Z-1)
            Z1 = Z
            Z = Z1-P1/PP
        i-=1
        X[i] = XM-XL*Z
        X[-i-1] = XM+XL*Z
        W[i]= 2*XL/((1-Z**2)*PP**2)
        W[-i-1] = W[i]

    return X,W


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))