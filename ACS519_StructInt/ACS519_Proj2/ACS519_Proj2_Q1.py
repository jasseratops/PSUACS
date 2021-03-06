# PSUACS
# ACS519_Proj1
# Jasser Alshehri
# Starkey Hearing Technologies
# 11/2/2019


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp
import time


def main(args):
    mn_fig2 = [[1,1],
          [1,3],
          [3,3],
          [1,2],
          [2,3],
          [2,2]]
    ab_fig2 = np.ones_like(mn_fig2)

    mn_fig3 = [[1,12],
          [1,11],
          [2,12],
          [12,12],
          [11,12],
          [11,11]]
    ab_fig3 = np.ones_like(mn_fig3)

    mn_fig6 = [[1,1],
               [1,3],
               [1,5],
               [1,4],
               [1,2]]
    ab_fig6 = [[1,1],
               [1,3],
               [1,5],
               [1,4],
               [1,2]]

    mn_notes = [[1,1],
                [2,1],
                [2,2],
                [1,11],
                [11,11]]
    ab_notes = np.ones_like(mn_notes)

    mn_lowIntDens2_2 = [[2, 2]]
    ab_lowIntDens2_2 = np.ones_like(mn_lowIntDens2_2)

    mn_lowIntDens11_11 = [[11, 11]]
    ab_lowIntDens11_11 = np.ones_like(mn_lowIntDens11_11)


    RadEff_Plot(ab_fig2,mn_fig2,N=64,ab_leg=False,low_ka=False,title_str="Wallace, Fig 2")
    RadEff_Plot(ab_fig3,mn_fig3,N=64,ab_leg=False,low_ka=False,title_str="Wallace, Fig 3")
    RadEff_Plot(ab_fig6,mn_fig6,N=64,ab_leg=True,low_ka=False,title_str="Wallace, Fig 6")

    RadEff_Plot(ab_fig2,mn_fig2,N=16,ab_leg=False,low_ka=False,title_str="Wallace, Fig 2, low int dens")
    RadEff_Plot(ab_fig3,mn_fig3,N=16,ab_leg=False,low_ka=False,title_str="Wallace, Fig 3, low int dens")
    RadEff_Plot(ab_fig6,mn_fig6,N=16,ab_leg=True,low_ka=False,title_str="Wallace, Fig 6, low int dens")

    RadEff_Plot(ab_fig2,mn_fig2,N=64,ab_leg=False,low_ka=True,title_str="Wallace, Fig 2, low ka")
    RadEff_Plot(ab_fig3,mn_fig3,N=64,ab_leg=False,low_ka=True,title_str="Wallace, Fig 3, low ka")

    RadEff_Plot_lowIntDens(ab_lowIntDens2_2,mn_lowIntDens2_2,"Radiation Efficiency for (2,2) Mode")
    RadEff_Plot_lowIntDens(ab_lowIntDens11_11,mn_lowIntDens11_11,"Radiation Efficiency for (11,11) Mode")

    '''
    RadEff_Plot(ab_notes,mn_notes,N=64,ab_leg=False,low_ka=True,title_str="low_ka notes")
    RadEff_Plot_comp([[1,1]],[[11,11]],N=64,ab_leg=False,low_ka=False,title_str="Gauss vs. Riemann")
    '''

    plt.show()

    return 0


def RadEff_Plot(ab_array,mn_array,N,ab_leg,low_ka,title_str):
    c = 343.
    f = np.linspace(0.01,10.E3,2048)
    omega = 2.*pi*f
    k = omega/c

    mn = mn_array
    ab = ab_array

    plt.figure(figsize=(5,7))
    plt.subplots_adjust(left=0.17)

    for i in range(len(mn)):
        m = mn[i][0]
        n = mn[i][1]
        a = ab[i][0]
        b = ab[i][1]

        S_mn_gauss, gamma = rad_eff_Wallace_gauss(a,b,m,n,N,k)
        ab_str = ""

        if ab_leg:
            ab_str = " b/a = " + str(b/a)

        p = plt.loglog(gamma,S_mn_gauss,label=str(m)+","+str(n)+ab_str)

        if low_ka:
            S_mn_lowka,_ = rad_eff_low_ka(a,b,m,n,k)
            plt.loglog(gamma,S_mn_lowka,linestyle=":", color = p[0].get_color())

    plt.title(title_str)
    plt.xlim(0.03,3)
    plt.ylim(1E-5,4)
    plt.xlabel(r"$\gamma$")
    plt.ylabel("Radiation Efficiency")
    plt.grid(True,which="both",axis="both")
    plt.legend()
    #plt.savefig("ACS519_Proj2_Q2 " + title_str+".png")

    return 0

def RadEff_Plot_comp(ab_array,mn_array,N,ab_leg,low_ka,title_str):
    c = 343.
    f = np.linspace(0.01,10.E3,2048)
    omega = 2.*pi*f
    k = omega/c

    mn = mn_array
    ab = ab_array

    plt.figure(figsize=(5,7))
    plt.subplots_adjust(left=0.17)

    for i in range(len(mn)):
        m = mn[i][0]
        n = mn[i][1]
        a = ab[i][0]
        b = ab[i][1]

        S_mn_gauss, gamma = rad_eff_Wallace_gauss(a,b,m,n,N,k)
        S_mn_riemann,_ = rad_eff_Wallace(a,b,m,n,N,k)
        ab_str = ""

        if ab_leg:
            ab_str = " b/a = " + str(b/a)

        p = plt.loglog(gamma,S_mn_gauss,label=str(m)+","+str(n)+" Gauss")
        p = plt.loglog(gamma,S_mn_riemann,label=str(m)+","+str(n)+" Riemann")


    plt.title(title_str)
    plt.xlim(0.003,3)
    plt.ylim(1E-8,4)
    plt.xlabel(r"$\gamma$")
    plt.ylabel("Radiation Efficiency")
    plt.grid(True,which="both",axis="both")
    plt.legend()

    return 0

def RadEff_Plot_lowIntDens(ab_array,mn_array,title_str):
    c = 343.
    f = np.linspace(0.01,10.E3,2048)
    omega = 2.*pi*f
    k = omega/c

    mn = mn_array
    ab = ab_array

    plt.figure(figsize=(5,7))
    plt.subplots_adjust(left=0.17)

    for i in range(len(mn)):
        m = mn[i][0]
        n = mn[i][1]
        a = ab[i][0]
        b = ab[i][1]

        S_mn_gauss_16, gamma = rad_eff_Wallace_gauss(a,b,m,n,16,k)
        S_mn_gauss_32, _ = rad_eff_Wallace_gauss(a,b,m,n,32,k)
        S_mn_gauss_64, _ = rad_eff_Wallace_gauss(a,b,m,n,64,k)

        ab_str = ""

        p = plt.loglog(gamma,S_mn_gauss_16,label="N=16")
        p = plt.loglog(gamma,S_mn_gauss_64,label="N=32")
        p = plt.loglog(gamma,S_mn_gauss_32,label="N=64")

    plt.title(title_str)
    plt.xlim(0.03,3)
    plt.ylim(1E-5,4)
    plt.xlabel(r"$\gamma$")
    plt.ylabel("Radiation Efficiency")
    plt.grid(True,which="both",axis="both")
    plt.legend()
    #plt.savefig("ACS519_Proj2_Q2 " + title_str+".png")

    return 0


def rad_eff_Wallace(a,b,m,n,N,k):
    start = time.time()
    A = 64.*(k**2)*a*b/((pi**6)*(m*n)**2)

    theta = np.linspace(0,pi/2.,N)
    phi = np.linspace(0,pi/2.,N)
    k_m = m*pi/a
    k_n = n*pi/b
    k_mn = np.sqrt((k_m**2)+(k_n**2))   # modal wavenumber

    gamma = k/k_mn      # wavenumber ratio

    m_odd = bool(m%2)
    n_odd = bool(n%2)

    dTheta = theta[1]-theta[0]
    dPhi = phi[1]-phi[0]

    S_mn = np.zeros_like(k)

    for i in range(len(theta)):
        weight = sin(theta[i]) * dTheta * dPhi
        #print weight
        for l in range(len(phi)):
            alpha = k*a*sin(theta[i]+(dTheta/2.))*cos(phi[l]+(dPhi/2.))
            beta  = k*b*sin(theta[i]+(dTheta/2.))*sin(phi[l]+(dPhi/2.))

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
            z = ((num/denom)**2)*weight
            #print z
            S_mn += z

    S_mn *= A

    end = time.time()
    print "Riemann: " + str(end-start)

    return S_mn, gamma

def rad_eff_Wallace_gauss(a,b,m,n,N,k):
    start = time.time()
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

    end = time.time()

    #print "Gauss: " + str(end-start)

    return S_mn, gamma

def rad_eff_low_ka(a,b,m,n,k):
    k_m = m*pi/a
    k_n = n*pi/b
    k_mn = np.sqrt((k_m**2)+(k_n**2))   # modal wavenumber
    gamma = k/k_mn      # wavenumber ratio

    arf = ((a * n) / (m * b)) + ((m*b)/(a*n))   # aspect ratio factor
    S_mn = np.zeros_like(k)      # initialize radiation efficiency vector

    m_odd = bool(m%2)
    n_odd = bool(n%2)

    if m_odd and n_odd:
        A = (32./(m*n*(pi**3)))*arf*(gamma**2)
        B = arf*m*n*pi*(gamma**2)/12.
        m_term = (1.-(8./((m*pi)**2)))*(a/b)
        n_term = (1.-(8./((n*pi)**2)))*(b/a)

    elif m_odd != n_odd:
        B = arf * m * n * pi * (gamma ** 2)/20.

        if m_odd:
            A = (8./(3.*pi)) * (arf**2) * (gamma ** 4) * b/a
            m_term = (1. - (8. / ((m * pi) ** 2))) * (a / b)
            n_term = (1. - (24./ ((n * pi) ** 2))) * (b / a)

        else:
            A = (8./(3.*pi)) * (arf**2) * (gamma ** 4) * a/b
            m_term = (1. - (24./ ((m * pi) ** 2))) * (a / b)
            n_term = (1. - (8. / ((n * pi) ** 2))) * (b / a)

    else:
        A = (2.*m*n*pi/15.)*(arf**3)*(gamma**6)
        B = arf*m*n*pi*(gamma**2)*(5./64.)
        m_term = (1. - (24. / ((m * pi) ** 2))) * (a / b)
        n_term = (1. - (24. / ((n * pi) ** 2))) * (b / a)


    S_mn = A*(1.-(m_term + n_term)*B)

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