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

### Given Dimensions
h = 6.35E-3  # m
a = 22.5E-2  # m
b = 12.4E-2  # m

### Material Properties for Aluminum (6061-T6)
E = 68.9E9  # Pa
v = 0.33  #
rho = 2.7E3  # kg/m^3
eta = 0.004


frq = np.linspace(1., 10.E3, 1024*2)      # freq vector
omega = frq * 2 * pi            # angular freq vector

MN = 6       # number of modes for convergence

def main(args):
    ### Drive points for Mobility
    dp = [[5.625E-2, 3.1E-2],
          [11.25E-2, 6.2E-2],
          [8.0E-2, 8.0E-2]]

    D_comp = ssint.flex_rig(E, h, v, eta)  # Calculate Complex Flexural Rigidity

    #Q1Q2_Res_Freqs(D_comp,plot=True)
    #Q3_Drive_Point_Mobility(dp,D_comp)
    Q4_Surface_Averaged_Mobility(dp,D_comp)
    #Q5_Variable_Loss_Factor(dp)
    plt.show()
    return 0

def Q1Q2_Res_Freqs(D_comp,plot=False):
    A_mn=1.             # mode shape amplitude
    m_length = 10       # number of m modes to try
    n_length = 10       # number of n modes to try

    m_array = np.zeros(10,dtype=int)                # array of m's for the 10 lowest modes
    n_array = np.zeros(10,dtype=int)                # array of n's for the 10 lowest modes
    omega_mn_array = np.zeros(10,dtype=complex)     # array of omegas for the 10 lowest modes
    table = np.zeros((m_length*n_length,3),dtype=complex)   # array of m's, n's, and omega's

    # Calculate resonant frequencies for all values of m,n

    for m in range(1,m_length+1):
        for n in range(1,n_length+1):
            omega_comp = ss_flatplate_frq(D_comp,rho,h,m,n,a,b)
            ind = ((m-1)*10) + (n-1)
            table[ind]= omega_comp,m,n

    # Sort the table (increasing) by values for omega, and take the first 10 values.
    low10 = np.sort(table[:,0])[:10]

    # calculate array of plate phase speed for thick plate resonance frequency calculations.
    c_B = ssint.plate_phase_speed(h,v,E,rho,omega,eta=eta)[0]

    # Print m's,
    if plot: print "|m|n|k_m|k_n|omega_mn|omega_mn_thick"
    for x in range(len(low10)):
        compare = low10[x]
        for i in range(len(table[:,0])):
            if table[i,0] == compare:
                omega_mn_array[x] = table[i,0].real
                m_array[x] = int(table[i,1].real)
                n_array[x] = int(table[i,2].real)
                # calculate thick-plate resonance
                omega_mn_thick = ss_thickplate_frq(c_B,omega,m_array[x],n_array[x],a,b).real
                k_m = m_array[x]*pi/a
                k_n = n_array[x]*pi/a
                if plot:
                    print "|" + str(m_array[x]) + "|" + str(n_array[x]) + "|" +str(k_m) + "|" + str(k_n) + "|"+ str(omega_mn_array[x].real) + "|" + str(
                        omega_mn_thick)
                    Mode_Mesh(A_mn, a, b, m_array[x], n_array[x])

                    ''' Save figures
                    path="mode" + str(m_array[x]) + str(n_array[x]) +".png"
                    print path
                    plt.savefig(path)
                    '''
    return omega_mn_array

def Q3_Drive_Point_Mobility(dp,D_comp):
    m_mn = ss_modalMass(rho,h,a,b)              # Calculate modal mass
    v_F_inf = inf_plateMob(D_comp,rho,h)*np.ones_like(omega)    # Calculate mobility for an infinite plate

    for i in range(len(dp)):
        plt.figure(figsize=(12,10))

        x = dp[i][0]                # Assign x and y coordinates from drive point list
        y = dp[i][1]

        # Calculate total drive point mobility and its average
        v_F_tot = tot_drivePoint_mobility(x,x,y,y,MN,MN,a,b,D_comp,rho,h,omega,m_mn)

        v_F_avg = np.mean(v_F_tot.real)*np.ones_like(v_F_tot.real)

        plt.subplot(311)
        plt.title(r"Re{v/F}: (" + str(x*100) + "cm," + str(y*100) + "cm)")
        plt.semilogy(omega,v_F_tot.real,label="v_F_tot",color="black")
        plt.semilogy(omega,v_F_inf.real,label="v_F_inf",color="red")
        plt.xlim(omega[0],omega[-1])
        plt.xlabel("Angular Frequency [rad/s]")
        plt.ylabel("Mobility [s/kg]")
        plt.legend()

        plt.legend()
        plt.subplot(312)
        plt.title(r"Im{v/F}: (" + str(x*100) + "cm," + str(y*100) + "cm)")
        plt.plot(omega, v_F_tot.imag,label="v_F_tot",color="black")
        #plt.plot(omega, v_F_inf.imag,label="v_F_inf")
        plt.xlim(omega[0],omega[-1])
        plt.xlabel("Angular Frequency [rad/s]")
        plt.ylabel("Mobility [s/kg]")
        plt.legend()

        plt.subplot(313)
        plt.title(r"|v/F|: (" + str(x*100) + "cm," + str(y*100) + "cm)")
        #plt.title("Drive Point: (" + str(x*100) + "cm," + str(y*100) + "cm)")
        plt.semilogy(omega,np.abs(v_F_tot),label="v_F_tot",color="black")
        plt.semilogy(omega,np.abs(v_F_inf),label="v_F_inf",color="red")
        plt.semilogy(omega,np.abs(v_F_avg),label="v_F_avg",linestyle=":",color="green",linewidth=3.0)
        plt.xlim(omega[0],omega[-1])
        plt.xlabel("Angular Frequency [rad/s]")
        plt.ylabel("Mobility Magnitude [s/kg]")
        plt.legend()
        plt.subplots_adjust(hspace=0.4)
        #save fig
        path = "dp_"+str(x*100)+"cm-"+str(y*100)+"cm.png"
        #plt.savefig(path)


    return 0

def Q4_Surface_Averaged_Mobility(dp,D_comp):
    m_mn = ss_modalMass(rho,h,a,b)              # Calculate modal mass
    firstModeOmega = Q1Q2_Res_Freqs(D_comp).real      # Calculate the frequency for the first 10 modes

    omega_peak1=firstModeOmega[0]               # Determine the frequencies for the modes we are interested in
    omega_peak2=firstModeOmega[2]               # Determined via trial and error
    omega_peak3=firstModeOmega[7]

    # Finding the argument for the value in the omega vector closest
    # to the omega peak values calculated above
    arg_peak1 = np.argmin(np.abs(omega-omega_peak1))
    arg_peak2 = np.argmin(np.abs(omega-omega_peak2))
    arg_peak3 = np.argmin(np.abs(omega-omega_peak3))

    omega_peak1=omega[arg_peak1]
    omega_peak2=omega[arg_peak2]
    omega_peak3=omega[arg_peak3]

    # Perform calculation for each drive point.
    for i in range(len(dp)):
        k_range = range(1, 10)  # Set  range of k, which is the number of integration points

        x_f = dp[i][0]                                      # Set x_f & y_f as the drive point coordinates
        y_f = dp[i][1]

        mode1_array = np.zeros_like(k_range,dtype=float)    # Init arrays for Mobility magnitudes at peaks 1,2,and 3
        mode2_array = np.zeros_like(k_range,dtype=float)
        mode3_array = np.zeros_like(k_range,dtype=float)

        plt.figure(figsize=(12,7))

        plt.subplot(211)
        plt.title(r"|v/F|: (" + str(x_f*100) + "cm," + str(y_f*100) + "cm)")

        for k in k_range:
            vK_Ff_tot = np.zeros_like(omega, dtype=complex)  # Initialize total drive point mobility array
            x_k_array = np.linspace((a/k), a, k)             # Generate array of x-points
            y_k_array = np.linspace((b/k), b, k)             # Generate array of y-points
            dx = a/k
            dy = b/k
            dAk = dx * dy                                    # Area of single integration increment

            for x_k in x_k_array:
                for y_k in y_k_array:
                    x_r = x_k -(dx/2.)       # obtain value in the MIDDLE of the increment
                    y_r = y_k -(dy/2.)       # obtain value in the MIDDLE of the increment
                    vK_Ff_tot += tot_drivePoint_mobility(x_r,x_f,y_r,y_f,MN,MN,a,b,D_comp,rho,h,omega,m_mn)

            vK_Ff_savg = np.abs(vK_Ff_tot)*dAk/((a*b))      # Calculate surface-averaged mobility

            mode1_array[k - 1] = np.abs(vK_Ff_savg[arg_peak1])  # Enter mobility mag. for peaks
            mode2_array[k - 1] = np.abs(vK_Ff_savg[arg_peak2])  #
            mode3_array[k - 1] = np.abs(vK_Ff_savg[arg_peak3])  #

            plt.semilogy(omega, vK_Ff_savg, label="k=" + str(k))
            plt.xlim(omega[0], omega[-1])
            plt.legend(loc=1)

        plt.axvline(omega_peak1,color="black")
        plt.axvline(omega_peak2,color="black")
        plt.axvline(omega_peak3,color="black")
        plt.xlabel("Angular Frequency [rad/s]")
        plt.ylabel("Mobility Magnitude (s/kg)")

        k_range = np.array(k_range)**2

        plt.subplot(212)
        plt.title("Mobility Magnitude vs k")
        plt.plot(k_range,mode1_array,label="mode1")
        plt.plot(k_range,mode2_array,label="mode2")
        plt.plot(k_range,mode3_array,label="mode3")
        plt.xlim(k_range[0],k_range[-1])
        plt.xlabel("k (number of integration points)")
        plt.ylabel("Mobility Magnitude (s/kg)")
        plt.legend(loc=1)
        plt.subplots_adjust(hspace=0.4)
        path = "savg-dp_"+str(x_f)+"-"+str(y_f)+".png"
        #plt.savefig(path)

    return 0

def Q5_Variable_Loss_Factor(dp):
    print dp
    print [dp[2]]

    #for i in range(len(dp[2]))
    D_comp = ssint.flex_rig(E,h,v,eta=0.04)
    Q3_Drive_Point_Mobility([dp[2]],D_comp)
    Q4_Surface_Averaged_Mobility([dp[2]],D_comp)


    D_comp = ssint.flex_rig(E,h,v,eta=0.4)
    Q3_Drive_Point_Mobility([dp[2]],D_comp)
    Q4_Surface_Averaged_Mobility([dp[2]],D_comp)



    return 0

#################
#### Methods ####
#################

def find_first_max(arr):
    for i in range(1,len(arr)):
        if (arr[i] > arr[i-1]) and (arr[i] > arr[i+1]):
            return i
            break

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

def ss_flatplate_frq(D_comp,rho,h,m,n,a,b):
    omega_comp = np.sqrt(D_comp/(rho*h))*(((m*pi/a)**2)+(n*pi/b)**2)
    return omega_comp

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

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))