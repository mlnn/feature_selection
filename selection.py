# -*- coding: utf-8 -*-
import numpy as np
import sklearn.linear_model as linear_model
from sklearn.linear_model import Ridge
import sklearn.feature_selection as feature_selection
from sklearn.ensemble import RandomForestRegressor

np.random.seed(0)
size = 750
X = np.random.uniform(0, 1, (size, 14))

print(X[:,1])

Y = (10 * np.sin(np.pi*X[:, 0]*X[:, 1]) + 20*(X[:, 2] - .5)**2 + 10 * X[:, 3] + 5*X[:, 4]**5 + np.random.normal(0, 1))
X[:,10:] = X[:,:4] + np.random.normal(0, .025, (size,4))

lin = linear_model.LinearRegression()
lin.fit(X, Y)
ridge = Ridge()  # alpha=0.1
ridge.fit(X, Y)
lasso = linear_model.Lasso()  # alpha=0.1
lasso.fit(X, Y)
randLasso = linear_model.RandomizedLasso()
randLasso.fit(X, Y)
rfe = feature_selection.RFE(estimator=linear_model.LinearRegression())
rfe.fit(X=X, y=Y)

rfr = RandomForestRegressor()
rfr.fit(X, Y)
freg = feature_selection.f_regression(X, Y)


ans_lin = abs(lin.coef_)
mx = [max(ans_lin)] * 14
ans_lin = ans_lin / mx
ans_ridge = abs(ridge.coef_)
mx = [max(ans_ridge)] * 14
ans_ridge = ans_ridge / mx
ans_lasso = abs(lasso.coef_)
# ������� �� 0
ans_randLasso = abs(randLasso.scores_)
mx = [max(ans_randLasso)] * 14
ans_randLasso = ans_randLasso / mx
ans_rfe = rfe.ranking_  # �������?
mx = [max(ans_rfe) + 1] * 14
ans_rfe = (mx - ans_rfe) / mx
ans_rfr = abs(rfr.feature_importances_)
mx = [max(ans_rfr)] * 14
ans_rfr = ans_rfr / mx
ans_freg = abs(freg[1])
mx = [max(ans_freg)] * 14
ans_freg = ans_freg / mx


ans = [0] * 14
for i in range(14):
    ans[i] = (abs(ans_lin[i]) + abs(ans_ridge[i]) + abs(ans_lasso[i]) + abs(ans_randLasso[i]) + abs(ans_rfe[i]) + abs(ans_rfr[i]) + abs(ans_freg[i])) / 7
print("One")
print (ans)

ridge2 = Ridge(alpha=7)
ridge2.fit(X, Y)
print("Two")
print(ridge2.coef_)

lasso2 = linear_model.Lasso(alpha=.05)
lasso2.fit(X, Y)
print("Three")
print(lasso2.coef_)


