import matplotlib.pyplot as plt
import math

def df(f,t):
    return math.pow(f,2)-3*f+math.exp(-t)

def plot_euler_soln( dt, ax ):

    t0=0
    t1=3

    f0=0 #not actually defined in the question!

    ts=[t0]
    fs=[f0]

    while ts[-1]<t1:

        f=fs[-1]+df(fs[-1],ts[-1])*dt

        ts.append(ts[-1]+dt)
        fs.append(f)

    ax.plot( ts,fs, label="Euler solution with dt=%.2f" % dt )
    ax.set_xlabel("t (seconds)")
    ax.set_ylabel("f(t)")

fig, ax = plt.subplots()
ax.set_title( 'Euler Solutions' )
plot_euler_soln( 0.5, ax )
plot_euler_soln( 0.1, ax )
plot_euler_soln( 0.01, ax )
ax.legend()
plt.show()
