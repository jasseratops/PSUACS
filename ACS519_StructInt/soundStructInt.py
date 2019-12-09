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

def critical_frq_panel(c_B,c_0,omega):
    i = np.argmin(np.abs(c_B-c_0))
    omega_cr = omega[i]
    return omega_cr

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