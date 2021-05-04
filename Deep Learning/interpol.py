import numpy as np
import pandas as pd




filename = 'Interpol.csv'
data = pd.read_csv(filename,  sep=';', header=0)

y = data.pop("Gain_UCM")

y.dropna(inplace=True)
y = y.values
x= [703, 718, 733, 758, 773,788,800,807,820,822,832,847,862,880,895,897,915,925,940,942,960]
xvals = [690,698,704,710,728,734,740,746,753,763,776,785,792,807,822,832,847,862,880,898,915,925,943,960]

yinterp = np.interp(xvals, x, y)


yinterp.tofile("result.csv", sep=";")


'''
x = np.linspace(0, 2*np.pi, 10)
print(x)
y = np.sin(x)
print(y)
xvals = np.linspace(0, 2*np.pi, 50)
print(xvals)

yinterp = np.interp(xvals, x, y)

print(yinterp)
'''