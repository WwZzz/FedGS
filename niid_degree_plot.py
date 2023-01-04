import matplotlib.pyplot as plt
import numpy as np
skew = [0.1,0.7,0.8,0.9]
alpha = [(-4 * np.log(s + 10e-8)) ** 4 for s in skew]
methods = {
    'uniform':[0.2534,0.2654,0.2922,0.3259],
    'mdsample':[0.2511,0.2625,0.2884,0.3281],
    'powerofchoice':[0.3260,0.3293,0.3566,0.3810],
    'fedprox':[0.2498,0.2627, 0.2901,0.3288],
    'fedgs':[0.2431,0.2520,0.2813,0.3184],
}
legends = list(methods.keys())
colors = ['r','g','b','purple','orange']
x = [1,2,3,4]
for id,algo in enumerate(methods):
    y = methods[algo]
    plt.plot(x,y,'--', linewidth=1.5,c = colors[id],label=legends[id])
    plt.plot(x, y, 'o', linewidth=3, c=colors[id])
plt.xticks(x,['Dirichlet(inf)','Dirichlet(4.14)','Dirichlet(0.63)','Dirichlet(0.03)',])
plt.legend()
plt.xlabel('The Degree of Data Heterogeneity')
plt.ylabel('Testing Loss')
plt.title('FashionMNIST on 100 Clients / LessDataFirst-0.9')
plt.show()