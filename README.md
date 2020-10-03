# reddit_OD_estimations

File name | description

cluster_and_linreg.py  | perform k means clustering on users, build linear regression models, and calculate p values

heat_plotter.py | given topic probabilities and attributes about a group of authors, comstruct colorfield plots

computer_author_attrs.py | construct the user network from comment data for use in the above scripts.

# How to run these files
1. get comment data from: https://bigquery.cloud.google.com/savedquery/190018058270:f56ba675837a4754a11df78511d1b69a

2. using your favorite NLP toolset, construct a topic model and calculate the average probability for each topic for each user

3. use these files to run compute_author_attrs.py and get some network data

4. (optional) use network data and topic probabilities to visualize colorfields with heat_plotter.py

5. use examples from cluster_and_linreg.py to generate your own models and calculate p values

6. Use autoencoder.py for logistic regression analysis
