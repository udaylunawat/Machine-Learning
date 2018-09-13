*  The bias-variance trade-off is the point where we are just adding noise by adding model complexity (flexibility).
*  The training error goes down as it has to, but the test error is starting to go up.
*  The model after the bias trade-off begins to overfit.


![alt text](https://github.com/udaylunawat/100DaysofMLCode/blob/master/Machine%20Learning%20Notes%20and%20notebooks/Cross%20Validation%20and%20bias%20variance%20tradeoff/Images/a.JPG)

*  A common temptation for beginners is to continually add complexity to a model until it fits the training set very well. But if it fits the training set much too well it will fail to predict for new points, i.e it will overfit.
*  This will cause large errors on new data such as the test set.
*  Balance out bias and variance of the model to the point the test data and train data have reached some sort of minimum and are grouping together.
