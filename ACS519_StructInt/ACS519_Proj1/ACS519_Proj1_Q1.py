# PSUACS
# ACS519_Proj1_Q1
# Jasser Alshehri
# Starkey Hearing Technologies
# 10/5/2019


from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from numpy import pi, sin, cos, tan, exp
import ACS519_StructInt.soundStructInt as ssint

def main(args):
    E = 68.9E9      # Pa
    v = 0.33        #
    rho = 2.7E3     # kg/m^3
    eta = 0.004
    h = 6.35E-3     # m
    a = 22.5E-2     # m
    b = 12.4E-2     # m
    A_mn = 1.

    m_length = 10
    n_length = 10

    m_array = np.zeros(10,dtype=int)
    n_array = np.zeros(10,dtype=int)
    omega_mn_array = np.zeros(10,dtype=complex)

    table = np.zeros((m_length*n_length,3),dtype=complex)

    D_comp = ssint.flex_rig(E,h,v,eta)

    for m in range(1,m_length+1):
        for n in range(1,n_length+1):
            omega_comp = ss_flatplate_frq(D_comp,rho,h,m,n,a,b)
            ind = ((m-1)*10) + (n-1)
            table[ind]= omega_comp,m,n

    low10 = np.sort(table[:,0])[:10]       # first 10 frqs

    print "|m|n|omega |"
    for x in range(len(low10)):
        compare = low10[x]
        for i in range(len(table[:,0])):
            if table[i,0] == compare:
                omega_mn_array[x] = table[i,0]
                m_array[x] = int(table[i,1].real)
                n_array[x] = int(table[i,2].real)
                print "|" +str(m_array[x]) + "|" + str(n_array[x]) + "|" + str(omega_mn_array[x])
                #Mode_Mesh(A_mn, a, b, m, n)

    #############
    # Drive Point Mobility
    #############

    dp = [[5.625E-2,3.1E-2],
         [11.25E-2,6.2E-2],
         [8.0E-2,8.0E-2]]

    frq = np.linspace(1.,10.E3,1024)
    omega = frq*2*pi
    m_mn = ss_modalMass(rho,h,a,b)

    v_F_inf = inf_plateMob(D_comp,rho,h)*np.ones_like(omega)

    for i in range(len(dp)):

        plt.figure()
        x = dp[i][0]
        y = dp[i][1]

        v_F_tot = tot_drivePoint_mobility(x,x,y,y,128,128,a,b,D_comp,rho,h,omega,m_mn)
        '''
        v_F_tot = np.zeros_like(omega, dtype=complex)
        for m in range(1,128):
            for n in range(1,128):
                v_F_mn = drivePoint_mobility(x, x, y, y, m, n, a, b, D_comp, rho, h, omega, m_mn)
                v_F_tot += v_F_mn

        '''
        v_F_avg = np.mean(v_F_tot.real)*np.ones_like(v_F_tot.real)

        plt.subplot(311)
        plt.title(r"Re{v/F}: (" + str(x*100) + "cm," + str(y*100) + "cm)")
        plt.semilogy(omega,v_F_tot.real,label="v_F_tot",color="black")
        plt.semilogy(omega,v_F_inf.real,label="v_F_inf")
        plt.xlim(omega[0],omega[-1])

        plt.legend()
        plt.subplot(312)
        plt.title(r"Im{v/F}: (" + str(x*100) + "cm," + str(y*100) + "cm)")
        plt.plot(omega, v_F_tot.imag,label="v_F_tot")
        plt.plot(omega, v_F_inf.imag,label="v_F_inf")
        plt.xlim(omega[0],omega[-1])
        plt.legend()

        plt.subplot(313)
        plt.title(r"|v/F|: (" + str(x*100) + "cm," + str(y*100) + "cm)")
        #plt.title("Drive Point: (" + str(x*100) + "cm," + str(y*100) + "cm)")
        plt.semilogy(omega,np.abs(v_F_tot),label="v_F_tot",color="black")
        plt.semilogy(omega,np.abs(v_F_inf),label="v_F_inf")
        plt.semilogy(omega,np.abs(v_F_avg),label="v_F_avg",linestyle=":")
        plt.xlim(omega[0],omega[-1])
        plt.legend()

    ###
    # Surface-Averaged Mobility
    ###

    dx = 0.1E-2
    dy = dx
    dAk = dx*dy
    x_k_array = np.arange(0, a + dx, dx)
    y_k_array = np.arange(0, b + dy, dy)

    A = a*b
    N = 200

    for i in range(len(dp)):
        vK_Ff_tot = np.zeros_like(omega, dtype=complex)
        x = dp[i][0]
        y = dp[i][1]

        for K in range(1,N):
            vK_Ff_tot += tot_drivePoint_mobility(x_k_array[K],x,y_k_array[K],y,128,128,a,b,D_comp,rho,h,omega,m_mn)

        vK_Ff_savg = np.abs(vK_Ff_tot)*dAk/(a*b)




    plt.show()

    return 0

def tot_drivePoint_mobility(x_r,x_f,y_r,y_f,m_points,n_points,a,b,D,rho,h,omega,m_mn):
    v_F_tot = np.zeros_like(omega,dtype=complex)
    for m in range(1,m_points):
        for n in range(1,n_points):
            v_F_tot += drivePoint_mobility(x_r, x_f, y_r, y_f, m, n, a, b, D, rho, h, omega, m_mn)
    return v_F_tot

def drivePoint_mobility(x_r,x_f,y_r,y_f,m,n,a,b,D,rho,h,omega,m_mn):
    A_mn = drivePoint_panelShape(x_r, x_f, y_r, y_f, m, n, a, b)
    omega_mn = ss_flatplate_frq(D, rho, h, m, n, a, b)
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

def ss_flatplate_frq(D_comp,rho,h,m,n,a,b):
    omega_comp = np.sqrt(D_comp/(rho*h))*(((m*pi/a)**2)+(n*pi/b)**2)
    return omega_comp

def drivePoint_panelShape(x_r,x_f,y_r,y_f,m,n,a,b):
    k_m = m*pi/a
    k_n = n*pi/b
    A_mn_DP = sin(k_m*x_r)*sin(k_n*y_r)*sin(k_m*x_f)*sin(k_n*y_f)

    return A_mn_DP

def ss_modalMass(rho,h,a,b):
    m_mn = rho*h*a*b/4.
    return m_mn

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))