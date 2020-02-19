from pylab import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import seaborn as sns
sns.set()

# increase the font size, so text is legible in presentations
plt.rcParams['font.size'] = 40

# Parameters

# Mass for each stage
m1 = 2290000+496200+123000
m2 = 496200+123000
m3 = 123000


# Fuel for each stage
f1 = 2160000
f2 = 456100
f3 = 109500

# Thrust for each stage
thrust1 = 35100 * 10**3
thrust2 = 5151 * 10**3
thrust3 = 1033.1 * 10**3


# Time for each stage
t1 = 168  # Total time for first phase   168s
t2 = 360    # Total time for the second phase  360s
t3 = 165 # Total time for the third phase    165s
t4 = 30 # Turning phase, 30s = 3 degrees per second
t5 = 20000 #  Orbit phase   t = 200 000
tmax = t1 + t2 + t3 + t4 + t5 # Total time
dt = 1 # Timestep

# Constants
Cd = 0.42 # Drag coefficent
angle = np.pi/(2*t4/dt) # Used to turn the rocket 90 degrees
radius = 5.5 # Radius of the rocket t

# Earth data
earth_mass = 5.97*10**24
G = 6.67*10**(-11) # Graviational constant
earth_radius = 6371 * 10**3


t = 0.	             # start time
x = 0                # initial position
y = earth_radius     # initial position
vx = 0            # initial velocity
vy = 0           # initial velocity
accx = 0   # acceleration in x
accy = 0   # accelertion in y

time = []            # list to store time
posx = []            # list to store x
posy = []            # list to store y
posr = []            # list to store distance from the earth
vel = []             # list to store velocity
acceleration = [] # list to store accleration


stepsperframe = 25  # Increases the speed of the animation
numframes     = int((tmax)/(stepsperframe*dt))


#sets up the figure, the axis, and the plot elements we want to animate
fig = plt.figure()
ax  = plt.subplot(xlim=(-10000*10**3, 10000*10**3), ylim=(-10000*10**3,10000*10**3))
plt.axhline(y=0)    # draw a default hline at y=1 that spans the xrange
plt.axvline(x=0)    # draw a default vline at x=1 that spans the yrange
plt.tight_layout()  # adapt the plot area tot the text with larger fonts
rocket1, = ax.plot([],[], 'ro', markersize=10)
rocket2, = ax.plot([],[], 'yo', markersize=10)
rocket3, = ax.plot([],[], 'go', markersize=10)
earth, = ax.plot([], [], 'bo', markersize=150)

# Perform a single integration step
def integrate():
    global t, x, y, vx, vy, accx,accy, m1, m2, m3, f1, f2, f3, wx, wy, angle
    r = sqrt(x**2 + y**2) # distance to the center of the earth
    v = sqrt(vx**2 + vy**2) # Total velocity
    acc = sqrt(accx**2 + accy**2) # Total acceleration

    if t < t1: # First phase
        accy = thrust1/m1 - G*earth_mass*y/r**3 - drag(y-earth_radius,vy)/m1
        m1 = m1 - f1/int(t1/dt) # Burns fuel
        #Euler
        # x += dt*vx
        # y += dt*vy
        # vx += dt*accx
        # vy += dt*accy

        #Euler-Cromer
        vx += dt*accx
        vy += dt*accy
        x += dt*vx
        y += dt*vy

    elif t<t1+t2: # Second phase
        accy = thrust2/m2 - G*earth_mass*y/r**3
        m2 = m2 - f2/int(t2/dt)
        #Euler
        # x += dt*vx
        # y += dt*vy
        # vx += dt*accx
        # vy += dt*accy

        #Euler-Cromer
        vx += dt*accx
        vy += dt*accy
        x += dt*vx
        y += dt*vy

    elif t<t1+t2+t3: # Third phase
        accy = thrust3/m3 - G*earth_mass*y/r**3
        m3 = m3 - f3/int(t3/dt)
        #Euler
        # x += dt*vx
        # y += dt*vy
        # vx += dt*accx
        # vy += dt*accy

        #Euler-Cromer
        vx += dt*accx
        vy += dt*accy
        x += dt*vx
        y += dt*vy

    elif t<t1+t2+t3+t4: # Turning phase
        accx = -G*earth_mass*x/r**3
        accy = -G*earth_mass*y/r**3
        #Euler
        # x += dt*vx
        # y += dt*vy
        # vx = np.sin(angle)*v + dt*accx
        # vy = np.cos(angle)*v + dt*accy

        #Euler-Cromer
        vx = np.sin(angle)*v + dt*accx
        vy = np.cos(angle)*v + dt*accy
        x += dt*vx
        y += dt*vy

        angle += np.pi/(2*t4/dt)

    else:  # Orbit phase
        accx = -G*earth_mass*x/r**3
        accy = -G*earth_mass*y/r**3

        #Euler
        # x += dt*vx
        # y += dt*vy
        # vx += dt*accx
        # vy += dt*accy

        #Euler-Cromer
        vx += dt*accx
        vy += dt*accy
        x += dt*vx
        y += dt*vy

    t += dt

    time.append(t)
    posx.append(x)
    posy.append(y)
    posr.append(r)
    vel.append(v)
    acceleration.append(acc)

# Initialization function
def init():
    rocket1.set_data([], [])
    return rocket1,

# Animation function
def animate(framenr):

    for it in range(stepsperframe):
        if sqrt(x**2+y**2) < earth_radius:  # Stops if we hit the eart
            break
        else:
            integrate()
    xx = x
    yy = y
    if t<t1: # Used to change color of the rocket
        rocket1.set_data(xx, yy)
        earth.set_data(0,0)
        return rocket1, earth,
    elif t<t1+t2: # Phase 2
        rocket2.set_data(xx, yy)
        earth.set_data(0,0)
        return rocket2, earth,
    else: # Phase 3
        rocket3.set_data(xx, yy)
        earth.set_data(0,0)
        return rocket3, earth,

# Help functions that calculates drag force, only used in first phase of launch.
def drag(h,v):
    return 0.55*1.228*np.exp(-0.0001168*h)*v**2*Cd*radius**2*np.pi

# Call the animator, blit=True means only re-draw parts that have changed
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=numframes, interval=25, blit=True, repeat=False)
plt.show()  # show the animation


# Plots position,velocity and acceleration versus time
fig, ax = plt.subplots(3)
fig.subplots_adjust(hspace=0.5)
posr_km = [i * 10**(-3) - 6371 for i in posr] # Scales the position vector to km

# Plots lines to separate the different phases
ax[0].plot([168,168],[0,2000],'k--',label='t=168s')
ax[0].plot([528,528],[0,2000],'k--',label='t=360s')
ax[1].plot([168,168],[0,7000],'k--',label='t=168s')
ax[1].plot([528,528],[0,7000],'k--',label='t=360s')
ax[2].plot([168,168],[0,70],'k--',label='t=168s')
ax[2].plot([528,528],[0,70],'k--',label='t=360s')


ax[0].plot(time,posr_km,color='r')
ax[1].plot(time,vel,color='b')
ax[2].plot(time,acceleration,color='g')
ax[0].set_title('Orbit phase, Euler-Cromer method',size=30)
ax[0].set_xlabel('Time [s]',size=15)
ax[0].set_ylabel('Position [km]',size=15)
ax[1].set_xlabel('Time [s]',size=15)
ax[1].set_ylabel('Velocity [m/s]',size=15)
ax[2].set_xlabel('Time [s]',size=15)
ax[2].set_ylabel('Acceleration [m/s^2]',size=15)
plt.show()

# Plots Orbits
plt.plot(posx,posy)
plt.plot(posx[0],posy[0],'ro',label='starting point')
plt.plot(posx[-1],posy[-1],'go',label='end point')
plt.title('Orbit, Euler-Cromer',size=30)
plt.legend(loc='upper right')
plt.xlabel('position in x',size=15)
plt.ylabel('position in y',size=15)
plt.show()

# Prints position,velociy and acceleration after simulation
print('Height above sea level =',posr[-1]-earth_radius)
print('Velocity =',vel[-1])
print('Acceleration =',acceleration[-1])
