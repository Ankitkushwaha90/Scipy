Here is a comprehensive set of SciPy examples to serve as a tutorial, covering common use cases like linear algebra, optimization, statistics, and signal processing.

### 1. Basic SciPy Import and Setup
SciPy builds on NumPy, so you’ll need both libraries.

```python
import numpy as np
from scipy import linalg, optimize, stats, signal

# Check SciPy version
import scipy
print("SciPy version:", scipy.__version__)
```
### 2. Linear Algebra
- a. Matrix Operations

SciPy provides advanced matrix operations.

```python
from scipy import linalg
import numpy as np

# Create two matrices
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Matrix multiplication
C = np.dot(A, B)
print("Matrix multiplication:\n", C)

# Inverse of a matrix
A_inv = linalg.inv(A)
print("Inverse of A:\n", A_inv)

# Matrix determinant
det_A = linalg.det(A)
print("Determinant of A:", det_A)
```
- b. Solving Linear Equations

```python
from scipy import linalg
import numpy as np

# Coefficients matrix A and constant vector B
A = np.array([[3, 2], [1, 4]])
B = np.array([6, 8])

# Solve Ax = B
x = linalg.solve(A, B)
print("Solution for x:\n", x)
```
### 3. Optimization
SciPy has powerful optimization algorithms.

- a. Minimizing a Function

```python
from scipy import optimize

# Define a simple quadratic function
def f(x):
    return x**2 + 2*x + 1

# Minimize the function
result = optimize.minimize(f, x0=0)  # Initial guess x0=0
print("Minimized value at:", result.x)
```
- b. Solving Non-Linear Equations

```python
from scipy import optimize

# Define a non-linear equation
def equation(x):
    return x**3 - 2*x - 5

# Solve the equation f(x) = 0
solution = optimize.root(equation, x0=0)
print("Root of the equation:", solution.x)
```
### 4. Integration
SciPy can perform both definite and indefinite integrals.

- a. Definite Integration

```python
from scipy import integrate
import numpy as np

# Define a function to integrate
def f(x):
    return np.sin(x)

# Integrate f(x) from 0 to π
result, error = integrate.quad(f, 0, np.pi)
print("Integral result:", result)
print("Estimated error:", error)
```
- b. Double Integration

```python
from scipy import integrate
import numpy as np

# Define a function of two variables
def f(x, y):
    return np.exp(-x**2 - y**2)

# Integrate f(x, y) over a 2D region
result, error = integrate.dblquad(f, 0, 1, lambda x: 0, lambda x: 1)
print("Double integral result:", result)
```
### 5. Interpolation
SciPy provides interpolation methods for both 1D and 2D data.

- a. 1D Interpolation

```python
from scipy import interpolate
import numpy as np

# Known data points
x = np.array([0, 1, 2, 3, 4])
y = np.array([0, 1, 0, 1, 0])

# Create an interpolating function
f = interpolate.interp1d(x, y, kind='cubic')

# Interpolate at new points
x_new = np.linspace(0, 4, 100)
y_new = f(x_new)

# Print a few interpolated points
print("Interpolated values:", y_new[:5])
```
- b. 2D Interpolation

```python
from scipy import interpolate
import numpy as np

# Known data points
x = np.arange(0, 5)
y = np.arange(0, 5)
z = np.array([[0, 1, 2, 3, 4],
              [1, 2, 3, 4, 5],
              [2, 3, 4, 5, 6],
              [3, 4, 5, 6, 7],
              [4, 5, 6, 7, 8]])

# Create a 2D interpolating function
f = interpolate.interp2d(x, y, z, kind='cubic')

# Interpolate at new points
x_new = np.linspace(0, 4, 10)
y_new = np.linspace(0, 4, 10)
z_new = f(x_new, y_new)
print("Interpolated 2D values:\n", z_new)
```
### 6. Statistics
SciPy provides a variety of statistical distributions and tests.

- a. Probability Distributions

```python
from scipy import stats

# Generate random data from a normal distribution
data = np.random.normal(loc=0, scale=1, size=1000)

# Fit the data to a normal distribution
mu, std = stats.norm.fit(data)
print("Mean:", mu, "Standard Deviation:", std)

# PDF of a normal distribution
x = np.linspace(-3, 3, 100)
pdf = stats.norm.pdf(x, mu, std)
print("PDF of the normal distribution:", pdf[:5])
```
- b. Hypothesis Testing

```python
from scipy import stats
import numpy as np

# Generate two datasets
data1 = np.random.normal(loc=0, scale=1, size=100)
data2 = np.random.normal(loc=0.5, scale=1, size=100)

# Perform a two-sample t-test
t_stat, p_value = stats.ttest_ind(data1, data2)
print("T-statistic:", t_stat)
print("P-value:", p_value)
```
### 7. Signal Processing
SciPy includes various signal processing functions.

- a. Fourier Transform

```python
from scipy import fftpack
import numpy as np

# Create a sample signal
t = np.linspace(0, 1, 500)
signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 10 * t)

# Compute the Fourier Transform
signal_fft = fftpack.fft(signal)

# Compute frequencies
frequencies = fftpack.fftfreq(len(signal), d=t[1] - t[0])

# Print the first few frequency components
print("Frequencies:", frequencies[:5])
print("FFT values:", signal_fft[:5])
```
- b. Filtering a Signal

```python
from scipy import signal
import numpy as np

# Create a noisy signal
t = np.linspace(0, 1, 500)
signal_data = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 20 * t) + 0.5 * np.random.normal(size=t.shape)

# Design a low-pass filter
sos = signal.butter(10, 15, 'low', fs=500, output='sos')

# Apply the filter
filtered_signal = signal.sosfilt(sos, signal_data)

# Print a few filtered values
print("Filtered signal:", filtered_signal[:5])
```
### 8. Sparse Matrices
SciPy provides efficient storage and operations on sparse matrices.

```python
from scipy.sparse import csr_matrix

# Create a dense matrix
dense_matrix = np.array([[0, 0, 1], [2, 0, 0], [0, 3, 4]])

# Convert to a CSR sparse matrix
sparse_matrix = csr_matrix(dense_matrix)
print("CSR sparse matrix:\n", sparse_matrix)

# Perform operations on the sparse matrix
print("Non-zero elements:", sparse_matrix.data)
print("Sum of all elements:", sparse_matrix.sum())
```
### 9. Image Processing (Optional: Use scipy.ndimage)
SciPy provides basic image manipulation functions using ndimage.

```python
from scipy import ndimage
import numpy as np

# Create a sample 2D array (image)
image = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])

# Apply a Gaussian filter
filtered_image = ndimage.gaussian_filter(image, sigma=1)
print("Filtered image:\n", filtered_image)

# Rotate the image by 45 degrees
rotated_image = ndimage.rotate(image, 45, reshape=False)
print("Rotated image:\n", rotated_image)
```
This collection of SciPy code examples provides an overview of core features, including linear algebra, optimization, integration, interpolation, statistics, signal processing, and more. These examples will help you get started with SciPy in practical data science and scientific computing scenarios.
