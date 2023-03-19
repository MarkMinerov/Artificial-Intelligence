# Logistic Regression

**Logistic regression** is an example of supervised learning. It is used to calculate or predict the probability of a binary (yes/no) event occurring. An example of logistic regression could be applying machine learning to determine if a person is likely to be infected with COVID-19 or not. Since we have two possible outcomes to this question - yes they are infected, or no they are not infected - this is called **binary classification**.

## How is logistic regression different from linear regression?

In **linear regression**, the outcome is continuous and can be any possible value. However in the case of logistic regression, the predicted outcome is discrete and restricted to a limited number of values.

For example, say we are trying to apply machine learning to the sale of a house. If we are trying to predict the sale price based on the size, year built, and number of stories we would use **linear regression**, as **linear regression** can predict a sale price of any possible value. If we are using those same factors to predict if the house sells or not, we would logistic regression as the possible outcomes here are restricted to yes or no.

Hence, **linear regression** is an example of a regression model and logistic regression is an example of a classification model.

## The three types of logistic regression

- Binary logistic regression - When we have two possible outcomes, like our original example of whether a person is likely to be infected with COVID-19 or not.
- Multinomial logistic regression - When we have multiple outcomes, say if we build out our original example to predict whether someone may have the flu, an allergy, a cold, or COVID-19.
- Ordinal logistic regression - When the outcome is ordered, like if we build out our original example to also help determine the severity of a COVID-19 infection, sorting it into mild, moderate, and severe cases.
