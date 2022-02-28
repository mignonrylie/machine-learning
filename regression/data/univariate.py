#synthetic-1 and synthetic-2 are single-input datasets, i.e. univariate regression
#for univariate regression, h(x) = h_t(x) = t_0 + t_1*x
#choose parameters t such that the difference between h_t(x) and y is minimized
#the loss function (measure of distance) will be mean squared error:
#J(t_1) = 1/2M * sum from i to M of (h_t(x^(i)) - y^(i))^2
#where x^(i), y^(i) is the i-th example, and M is the number of data points
#to update our parameters, we use gradient descent:
#t_j = t_j - a d/dt_j J(t)
#where t_j is the given parameter, a (0<a<1) is the learning rate, and d/dt_j is the partial derivative of J(t) w.r.t. the given parameter

#start with random values for t_0, t_1, and a
t0 = 20 #bias term, b in y=mx+b
t1 = 0.25 #weight, m in y=mx+b
a = 0.1 #learning rate

#update t0 and t1 as follows:
#t_j = t_j - a d/dt_j J(t)
#==
#t_j = t_j - a d/dt_j (1/2M * sum from i to M of (h_t(x^(i)) - y^(i))^2)
#==
#t_j = t_j - a d/dt_j (1/2M * sum from i to M of (t_0 + t_1*x^(i) - y^(i))^2)


#t0 = t0 - a * 1/M sum from i to M of (t0 )