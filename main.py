import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf
import pywt

# Define the radius of the arc
radius = 1.0

# Calculate the start and end angles for the arc
start_angle = np.arctan(-1)
end_angle = np.arctan(1)

# Generate random angles within the angle range
np.random.seed(42)
num_points = 500

# Convert the polar coordinates (radius, angle) to Cartesian coordinates (x, y)
x_data = np.random.rand(num_points)
y_data = np.random.rand(num_points)

# Create the RBF interpolator using a thin-plate spline kernel
rbf_interpolator = Rbf(x_data, y_data, function='thin_plate')

# Create a range of x values for visualization
x_values = np.linspace(0, 1, num=num_points)

# Interpolate the y values for the x values
y_interpolated = rbf_interpolator(x_values)

# Perform wavelet decomposition using Coiflet wavelets (Coif1)
wavelet = 'coif3'
coeffs = pywt.wavedec(y_interpolated, wavelet)

# Apply thresholding to the detail coefficients to remove noise
threshold = 0.1
coeffs[1:] = [
  pywt.threshold(detail_coeff, value=threshold, mode='soft')
  for detail_coeff in coeffs[1:]
]

# Reconstruct the denoised RBF data
y_denoised = pywt.waverec(coeffs, wavelet)

# Plot the original points, RBF interpolation, and denoised RBF
plt.scatter(x_data, y_data, c='r', s=10, edgecolors='k', label='Original data')
plt.plot(x_values, y_interpolated, 'b-', label='RBF Interpolation')
plt.plot(x_values, y_denoised, 'g-', label='Denoised RBF')
plt.title(
  'Thin-Plate Spline RBF Interpolation and Denoising using Coiflet Wavelets')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
