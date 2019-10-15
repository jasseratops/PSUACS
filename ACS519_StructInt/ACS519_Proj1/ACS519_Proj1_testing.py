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
    E = 200.E9      # Pa
    v = 0.303      #
    rho = 7.75E6    # kg/m^3
    eta = 0.01
    h = 0.005       # m
    a = 1.0         # m
    b = 4.3         # m
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


    # mobility
    dp = [[0.5,0.5]]

    frq = np.linspace(1.,10.E2,1024)
    omega = frq*2*pi
    m_mn = ss_modalMass(rho,h,a,b)


    v_F_inf = inf_plateMob(D_comp,rho,h)*np.ones_like(omega)



    for i in range(len(dp)):
        v_F_tot = np.zeros_like(omega, dtype=complex)
        plt.figure()
        x = dp[i][0]
        y = dp[i][1]
        for m in range(1,300):
            for n in range(1,300):
                A_mn = drivePoint_panelShape(x,x,y,y,m,n,a,b)
                omega_mn = ss_flatplate_frq(D_comp,rho,h,m,n,a,b)
                v_F_mn = (-1j*omega/m_mn)*A_mn/((omega_mn**2)-(omega**2))
                v_F_tot += v_F_mn
                '''
                plt.subplot(211)
                plt.semilogy(omega, v_F_mn.real, label="("+str(m)+","+str(n)+")")
                plt.subplot(212)
                plt.plot(omega, v_F_mn.imag, label="("+str(m)+","+str(n)+")")
                '''
        v_F_avg = np.mean(v_F_tot.real)*np.ones_like(v_F_tot.real)

        plt.subplot(211)
        plt.title("Drive Point: (" + str(x) + "," + str(y) + ")")
        plt.semilogy(omega,v_F_tot.real,label="v_F_tot",color="black")
        plt.semilogy(omega,v_F_inf.real,label="v_F_inf")
        plt.semilogy(omega,v_F_avg,label="v_F_avg")
        plt.xlim(omega[0],omega[-1])
        plt.legend()
        plt.subplot(212)
        plt.plot(omega, v_F_tot.imag,label="v_F_tot")
        plt.plot(omega, v_F_inf.imag,label="v_F_inf")
        plt.xlim(omega[0], omega[-1])
        plt.legend()

        plt.figure()
        plt.title("Drive Point: (" + str(x) + "," + str(y) + ")")
        plt.semilogy(omega,np.abs(v_F_tot),label="v_F_tot",color="black")
        plt.semilogy(omega,np.abs(v_F_inf),label="v_F_inf")
        plt.semilogy(omega,np.abs(v_F_avg),label="v_F_avg")
        plt.legend()

    plt.show()

    return 0

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