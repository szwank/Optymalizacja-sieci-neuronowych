from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt

# x = np.array([1.5, 2, 2.6, 3.5, 4.5, 5.5, 7, 8, 9])
# y = np.array([1, 0.3, 0.4, 0.17, 0.13, 0.07, 0, 0, 0])
# f2 = interpolate.UnivariateSpline(x, y, s=100)
#
# xnew = np.linspace(1.5, 7, num=41, endpoint=True)
#
# plt.plot(x, y, 'o', xnew, f2(xnew), '-')
# plt.legend(['data', 'linear', 'cubic'], loc='best')
# plt.show()


def calculate_percent_of_argument_increase(the_array: np.ndarray):

    the_array = np.subtract(the_array, min(the_array))
    print(the_array)
    sorted_array = the_array
    sorted_array.sort()
    increase = []
    for element in the_array:
        increase.append(element / sorted_array[-2])
    print(increase)


def calculate_the_values_in_the_range(min: float, max: float, increase_value_percents: list):
    values_in_range = []
    for percentage_increase in increase_value_percents:
        value = min + (max - min) * percentage_increase
        values_in_range.append(value)
    return values_in_range

# x = np.array([1.5, 1.65, 2.65, 3.46, 4.5, 5.5, 7, 8])
percentage_increases = [0.0, 0.027272727272727258, 0.20909090909090908, 0.3563636363636364, 0.5454545454545454, 0.7272727272727273, 1.0, 1.1818181818181819]
minimum = 1
maximum = 10
x = calculate_the_values_in_the_range(minimum, maximum, percentage_increases)
y = np.array([1, 0.8, 0.31, 0.15, 0.07, 0.02, 0, 0])
tck = interpolate.splrep(x, y, s=0.001)
xnew = np.linspace(minimum, maximum, num=41, endpoint=True)
ynew = interpolate.splev(xnew, tck, der=0)

plt.figure()
plt.plot(xnew, ynew, '-', x, y, 'o')
# plt.legend(['Linear', 'Cubic Spline', 'True'])
# plt.axis([-0.05, 6.33, -1.05, 1.05])
# plt.title('Cubic-spline interpolation')
plt.show()