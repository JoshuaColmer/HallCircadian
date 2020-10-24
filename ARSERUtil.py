'''
The file cantain filtering subroutine savit_golay, ols class, they were got from scipy.org cookbook
The two subroutines are called by class Arser
Date: Thu Oct 15 10:21:49 CST 2009
'''
from __future__ import division
import numpy
from scipy import c_, ones, dot, stats, diff
from scipy.linalg import inv, solve, det
from numpy import log, pi, sqrt, square, diagonal, array, angle
from numpy.random import randn, seed
import time


def savitzky_golay(data, kernel=11, order=4):
    """
        applies a Savitzky-Golay filter
        input parameters:
        - data => data as a 1D numpy array
        - kernel => a positiv integer > 2*order giving the kernel size
        - order => order of the polynomal
        returns smoothed data as a numpy array
        invoke like:
        smoothed = savitzky_golay(<rough>, [kernel = value], [order = value]
    """
    try:
        kernel = abs(int(kernel))
        order = abs(int(order))
    except ValueError as msg:
        raise ValueError("kernel and order have to be of type int (floats will be converted).")
    if kernel % 2 != 1 or kernel < 1:
        raise TypeError("kernel size must be a positive odd number, was: %d" % kernel)
    if kernel < order + 2:
        raise TypeError("kernel is to small for the polynomals\nshould be > order + 2")

    # a second order polynomal has 3 coefficients
    order_range = range(order + 1)
    half_window = (kernel - 1) // 2
    b = numpy.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
    # since we don't want the derivative, else choose [1] or [2], respectively
    m = numpy.linalg.pinv(b).A[0]
    window_size = len(m)
    half_window = (window_size - 1) // 2

    # precompute the offset values for better performance
    offsets = range(-half_window, half_window + 1)
    offset_data = zip(offsets, m)

    smooth_data = list()

    # temporary data, extended with a mirror image to the left and right
    firstval = data[0]
    lastval = data[len(data) - 1]

    # left extension: f(x0-x) = f(x0)-(f(x)-f(x0)) = 2f(x0)-f(x)
    # right extension: f(xl+x) = f(xl)+(f(xl)-f(xl-x)) = 2f(xl)-f(xl-x)
    leftpad = numpy.zeros(half_window) + 2 * firstval
    rightpad = numpy.zeros(half_window) + 2 * lastval
    leftchunk = data[1:1 + half_window]
    leftpad = leftpad - leftchunk[::-1]
    rightchunk = data[len(data) - half_window - 1:len(data) - 1]
    rightpad = rightpad - rightchunk[::-1]
    data = numpy.concatenate((leftpad, data))
    data = numpy.concatenate((data, rightpad))
    for i in range(half_window, len(data) - half_window):
        value = 0.0
        for offset, weight in offset_data:
            value += weight * data[i + offset]
        smooth_data.append(value)
    return numpy.array(smooth_data)


class ols:
    """
    Author: Vincent Nijs (+ ?)
    Email: v-nijs at kellogg.northwestern.edu
    Last Modified: Mon Jan 15 17:56:17 CST 2007

    Dependencies: See import statement at the top of this file
    Doc: Class for multi-variate regression using OLS
    For usage examples of other class methods see the class tests at the bottom of this file. To see the class in action
    simply run this file using 'python ols.py'. This will generate some simulated data and run various analyses. If you have rpy installed
    the same model will also be estimated by R for confirmation.
    Input:
        y = dependent variable
        y_varnm = string with the variable label for y
        x = independent variables, note that a constant is added by default
        x_varnm = string or list of variable labels for the independent variables

    Output:
        There are no values returned by the class. Summary provides printed output.
        All other measures can be accessed as follows:
        Step 1: Create an OLS instance by passing data to the class
            m = ols(y,x,y_varnm = 'y',x_varnm = ['x1','x2','x3','x4'])
        Step 2: Get specific metrics
            To print the coefficients:
                >>> print m.b
            To print the coefficients p-values:
                >>> print m.p

    """

    def __init__(self, y, x, y_varnm='y', x_varnm=''):
        """
        Initializing the ols class.
        """
        self.y = y
        self.x = c_[ones(x.shape[0]), x]
        self.y_varnm = y_varnm
        if not isinstance(x_varnm, list):
            self.x_varnm = ['const'] + list(x_varnm)
        else:
            self.x_varnm = ['const'] + x_varnm

        # Estimate model using OLS
        self.estimate()

    def estimate(self):

        # estimating coefficients, and basic stats
        self.inv_xx = inv(dot(self.x.T, self.x))
        xy = dot(self.x.T, self.y)
        self.b = dot(self.inv_xx, xy)  # estimate coefficients

        self.nobs = self.y.shape[0]  # number of observations
        self.ncoef = self.x.shape[1]  # number of coef.
        self.df_e = self.nobs - self.ncoef  # degrees of freedom, error
        self.df_r = self.ncoef - 1  # degrees of freedom, regression

        self.e = self.y - dot(self.x, self.b)  # residuals
        self.sse = dot(self.e, self.e) / self.df_e  # SSE
        self.se = sqrt(diagonal(self.sse * self.inv_xx))  # coef. standard errors
        self.t = self.b / self.se  # coef. t-statistics
        self.p = (1 - stats.t.cdf(abs(self.t), self.df_e)) * 2  # coef. p-values

        self.R2 = 1 - self.e.var() / self.y.var()  # model R-squared
        self.R2adj = 1 - (1 - self.R2) * ((self.nobs - 1) / (self.nobs - self.ncoef))  # adjusted R-square

        self.F = (self.R2 / self.df_r) / ((1 - self.R2) / self.df_e)  # model F-statistic
        self.Fpv = 1 - stats.f.cdf(self.F, self.df_r, self.df_e)  # F-statistic p-value

    def dw(self):
        """
        Calculates the Durbin-Waston statistic
        """
        de = diff(self.e, 1)
        dw = dot(de, de) / dot(self.e, self.e);

        return dw

    def omni(self):
        """
        Omnibus test for normality
        """
        return stats.normaltest(self.e)

    def JB(self):
        """
        Calculate residual skewness, kurtosis, and do the JB test for normality
        """

        # Calculate residual skewness and kurtosis
        skew = stats.skew(self.e)
        kurtosis = 3 + stats.kurtosis(self.e)

        # Calculate the Jarque-Bera test for normality
        JB = (self.nobs / 6) * (square(skew) + (1 / 4) * square(kurtosis - 3))
        JBpv = 1 - stats.chi2.cdf(JB, 2);

        return JB, JBpv, skew, kurtosis

    def ll(self):
        """
        Calculate model log-likelihood and two information criteria
        """

        # Model log-likelihood, AIC, and BIC criterion values
        ll = -(self.nobs * 1 / 2) * (1 + log(2 * pi)) - (self.nobs / 2) * log(dot(self.e, self.e) / self.nobs)
        aic = -2 * ll / self.nobs + (2 * self.ncoef / self.nobs)
        bic = -2 * ll / self.nobs + (self.ncoef * log(self.nobs)) / self.nobs

        return ll, aic, bic

    def summary(self):
        """
        Printing model output to screen
        """

        # local time & date
        t = time.localtime()

        # extra stats
        ll, aic, bic = self.ll()
        JB, JBpv, skew, kurtosis = self.JB()
        omni, omnipv = self.omni()

if __name__ == '__main__':

    ##########################
    ### testing the ols class
    ##########################

    # creating simulated data and variable labels
    seed(1)
    data = randn(100, 5)  # the data array

    # intercept is added, by default
    m = ols(data[:, 0], data[:, 1:], y_varnm='y', x_varnm=['x1', 'x2', 'x3', 'x4'])
    m.summary()

    # if you have rpy installed, use it to test the results
    have_rpy = False
    try:
        print
        "\n"
        print
        "=" * 30
        print
        "Validating OLS results in R"
        print
        "=" * 30

        import rpy

        have_rpy = True
    except ImportError:
        print
        "\n"
        print
        "=" * 30
        print
        "Validating OLS-class results in R"
        print
        "=" * 30
        print
        "rpy is not installed"
        print
        "=" * 30

    if have_rpy:
        y = data[:, 0]
        x1 = data[:, 1]
        x2 = data[:, 2]
        x3 = data[:, 3]
        x4 = data[:, 4]
        rpy.set_default_mode(rpy.NO_CONVERSION)
        linear_model = rpy.r.lm(rpy.r("y ~ x1 + x2 + x3 + x4"), data=rpy.r.data_frame(x1=x1, x2=x2, x3=x3, x4=x4, y=y))
        rpy.set_default_mode(rpy.BASIC_CONVERSION)
        print
        linear_model.as_py()['coefficients']
        summary = rpy.r.summary(linear_model)
        print
        summary