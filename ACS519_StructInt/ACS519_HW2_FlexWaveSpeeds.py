import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, tan, exp

def main(args):
    runner(h=1.0, E=30.E6, rho=7.33E-4, K=(5. / 6.), v=0.3, plotTitle="Original")
    runner(h=2*1.0, E=30.E6, rho=7.33E-4, K=(5. / 6.), v=0.3, plotTitle="Double Plate Thickness")
    runner(h=1.0, E=2*30.E6, rho=7.33E-4, K=(5. / 6.), v=0.3, plotTitle="Double Young's Modulus")
    runner(h=1.0, E=30.E6, rho=2*7.33E-4, K=(5. / 6.), v=0.3, plotTitle="Double Density")
    runner(h=1.0, E=30.E6, rho=7.33E-4, K=(5. / 6.), v=0.499, plotTitle="Poisson Ratio = 0.499")

    plt.show()         # uncomment to display plots


def runner(h,E,rho,K,v,plotTitle=""):

    f = np.linspace(0.,100.E3,2**16)
    omega = 2*pi*f

    c_S = shear_speed(E,rho,K,v)
    c_B, c_B_Mindlin, c_B_thin = plate_phase_speed(h,v,E,rho,omega,K)
    k_B = omega/c_B
    print "-"*10
    print plotTitle
    print "Shear Wave Speed: " + str(c_S)
    for i in range(len(c_B)):
        if c_B[i] >= c_S*0.85:
            x = f[i]
            x_Norm = k_B[i]*h
            print "c_B=85% of c_S @ f = " +str(x)
            print "c_B=85% of c_S @ k_B*h = " +str(x_Norm)
            break

    plt.figure(figsize=(10,7))
    plt.subplot(211)
    plt.plot(f,c_B,label = r"${c_B}$")
    plt.plot(f,c_B_Mindlin,label = r"${c_{B_{low}}}$")
    plt.plot(f,c_B_thin,label = r"${c_{B_{high}}}$")
    plt.axvline(x, color="red")
    plt.grid()
    plt.title(plotTitle)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Phase Speed [inches/seconds]")
    plt.xlim(f[0],f[-1])
    plt.legend()

    plt.subplot(212)
    plt.plot(k_B*h,c_B/c_S)
    plt.axvline(x_Norm, color="red")
    plt.title(plotTitle + ", Normalized")
    plt.xlabel("Normalized Wavenumber")
    plt.ylabel("Normalized Phase Speed")
    plt.xlim(0,11)
    plt.grid()
    plt.subplots_adjust(hspace=0.4)
    plt.ylim(0,1)

    plt.savefig(plotTitle+".png")  # uncomment to save plots

    return 0

def plate_phase_speed(h,v,E,rho,omega,K=5./6.):
    I_prime = plate_mom_inertia_wid(h)
    G = structural_shear_mod(E,v)
    D = flex_rig(E,h,v)
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

def flex_rig(E,h,v):
    I_prime = plate_mom_inertia_wid(h)
    D = E * (h ** 3) / (12. * (1. - (v ** 2)))
    return D

def plate_mom_inertia_wid(h):
    I_prime = (h**3)/12.
    return I_prime

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))