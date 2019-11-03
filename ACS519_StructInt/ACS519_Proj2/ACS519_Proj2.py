# PSUACS
# ACS519_Proj1
# Jasser Alshehri
# Starkey Hearing Technologies
# 11/2/2019


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp


def main(args):
    c = 343.
    f = np.linspace(1,1.E3,1024)
    omega = 2.*pi*f
    k = omega/c
    #print k

    a = 1.
    b = 1.

    m=1
    n=1

    N = 128
    phi = np.linspace(0,pi/2.,N)
    theta = np.linspace(0,pi/2.,N)


    #phi,theta = np.meshgrid(phi, theta)

    S_mn, gamma = rad_eff_Wallace_gauss(a,b,m,n,N,k)

    #S_mn, gamma = rad_eff_low_ka(a,b,m,n,k)

    plt.figure()
    plt.loglog(gamma,S_mn)
    plt.xlim(0,3)
    #plt.ylim(S_mn[1],2)
    plt.show()


    return 0

def rad_eff_Wallace(a,b,m,n,theta,phi,k):
    A = 64.*(k**2)*a*b/((pi**6)*(m*n)**2)

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
            denom = (((alpha/m*pi)**2)-1)*(((beta/n*pi)**2)-1)
            z = ((num/denom)**2)*weight
            #print z
            S_mn += z

    S_mn *= A

    return S_mn, gamma

def rad_eff_Wallace_gauss(a,b,m,n,N,k):
    S_mn = np.zeros_like(k)

    x_i,w_i = gauleg(0.,pi/2.,N)
    x_j,w_j = gauleg (0.,pi/2.,N)

    A = 64.*(k**2)*a*b/((pi**6)*(m*n)**2)

    k_m = m*pi/a
    k_n = n*pi/b
    k_mn = np.sqrt((k_m**2)+(k_n**2))   # modal wavenumber

    gamma = k/k_mn      # wavenumber ratio

    m_odd = bool(m%2)
    n_odd = bool(n%2)

    for i in range(N):
        for j in range(N):
            alpha = k*a*sin(x_i)*cos(x_j)
            beta  = k*b*sin(x_i)*sin(x_j)

            if m_odd:
                num_a = cos(alpha/2.)
            else:
                num_a = sin(alpha/2.)

            if n_odd:
                num_b = cos(beta/2.)
            else:
                num_b = sin(beta/2.)

            num = num_a*num_b
            denom = (((alpha/m*pi)**2)-1)*(((beta/n*pi)**2)-1)
            g_x = ((num/denom)**2)
            S_mn += w_i*w_j*g_x*sin(x_i)

        S_mn *= A

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
        A = (32./(m*n*pi**3))*arf*(gamma**2)
        B = arf*(m*n*pi/12.)*(gamma**2)
        m_term = (1.-(8./((m*pi)**2)))*(a/b)
        n_term = (1.-(8./((n*pi)**2)))*(b/a)

    elif m_odd != n_odd:
        A = 8./(3.*pi)*(arf**2)(gamma**4)*b/a
        B = m*n*pi*(gamma**2)/20.

        if m_odd:
            m_term = (1.-(8./((m*pi)**2)))*(a/b)
            n_term = (1.-(24./((n*pi)**2)))*(b/a)
        else:
            m_term = (1.-(24./((m*pi)**2)))*(a/b)
            n_term = (1.-(8./((n*pi)**2)))*(b/a)

    else:
        A = (2.*m*n*pi/15.)*(arf**3)*(gamma**6)
        B = (5.*m*n*pi/64.)*(gamma**2)
        m_term = (1. - (24. / ((m * pi) ** 2))) * (a / b)
        n_term = (1. - (24. / ((n * pi) ** 2))) * (b / a)


    S_mn = A**(1.-(m_term + n_term)*B)

    return S_mn, gamma

def gauleg(X1,X2,N):
    print N
    eps = 3e-14
    M = (N+1)/2
    XM = 0.5*(X2+X1)
    XL = 0.5*(X2-X1)

    X = np.zeros(M)
    W = np.zeros(M)

    for i in range(M):
        i+=1
        Z = cos(pi*(i-0.25)/(N+0.5))
        Z1 = -100000

        while np.abs(Z-Z1) > eps:
            P1 = 1
            P2 = 0
            for j in range(N):
                j+1
                P3 = P2
                P2 = P1
                P1 = ((2*j-1)*Z*P2-(j-1)*P3)/j
            PP = N*(Z*P1-P2)/(Z*Z-1)
            Z1 = Z
            Z = Z1-P1/PP
        print i
        i-=1
        X[i] = XM-XL*Z
        X[N+1-i] = XM+XL*Z
        W[i]= 2*XL/((1-Z**2)*PP**2)
        W[N+1-i] = W[i]

    return X,W

if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))