import numpy as np
import matplotlib.pyplot as plt
from my_relu import relu

x=np.linspace(-3,3,1000)
x_1=np.linspace(-6,6,1000)

y=relu(x)
y_1=relu(x_1)

pf = np.polyfit(x, y, 2)
p = np.poly1d(pf)

z = p(x)
z_1 = p(x_1)

print pf

def plot(x,y,z):
    plt.plot(x,y,label="$relu$",color="red",linewidth=2)
    plt.plot(x,z,"b--",label="$ployfit$")
    plt.xlim(-6,6)
    plt.ylim(-4,4)
    plt.legend()
    plt.show()
 
plot(x_1,y_1,z_1)

