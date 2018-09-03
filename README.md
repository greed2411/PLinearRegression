# PLinearRegression
Scikit-Learn's linear regression extended with p-values.

> "... the null hypothesis is never proved or established, but is possibly disproved, in the course of experimentation. Every experiment may be said to exist only to give the facts a chance of disproving the null hypothesis." -  R. A. Fisher 

People from [R](https://www.r-project.org/) background are familiar with [hypothesis testing](https://en.wikipedia.org/wiki/Statistical_hypothesis_testing) and [p-values](https://en.wikipedia.org/wiki/P-value) whereas ones from Python's [scikit-learn](http://scikit-learn.org/) background haven't heard of them. That's why I just made this script, which is an extension of scikit-learn's [LinearRegression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html), it can also be extended to Lasso and Ridge regressions respectively.

critiques and comments are always welcome.


A quick example:

```python
from p_linear_regression import PLinearRegression
plr = PLinearRegression()
plr.fit(X_train, y_train)
y_pred = plr.predict(X_test)

print(plr.summary)
```

Output for the diabetes dataset from sklearn looks like this:

```
   coefficients  standard Errors  t statistic      p values
0     37.900314        68.934688     0.549800  5.828141e-01
1   -241.966248        68.468840    -3.533962  4.654885e-04
2    542.425753        76.826436     7.060405  9.272139e-12
3    347.708305        71.252628     4.879937  1.628348e-06
4   -931.461261       450.477090    -2.067722  3.941565e-02
5    518.044055       363.566096     1.424896  1.550964e-01
6    163.403535       232.663793     0.702316  4.829584e-01
7    275.310038       185.125003     1.487158  1.378919e-01
8    736.189098       192.157851     3.831168  1.516492e-04
9     48.671125        73.305207     0.663952  5.071672e-01
```

There's also a [demo.ipynb](https://github.com/greed2411/PLinearRegression/blob/master/demo.ipynb) along with this repository for demonstration purposes.
