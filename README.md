# sydneysolar

This project is a web based dashboard application designed to predict an optimal design speed for a solar racing vehicle. 

Available to view at : https://sydneysolar.herokuapp.com (Runs best on chrome, note heroku is a free web server, so if the webpage has been inactive for more than 30 minutes it will take a few minutes to get started again, please be patient). 

Notes on using the machine learning tools:
- Near the bottom of the page are the machine learning tools for predicting solar exposure 
- Step 1: Select a regression model from the left dropdown 
- Step 2: Select inputs from the right dropdown. The inputs you choose are based on the correlation results in the above plots (either trusting the spearman rank results or intuitively observing the scatter and time series plots)
- Step 3: Click run, most models have already been trained so this should be enough. The centre button will change colour when the process is finished to let you know it's ready for another input.
- Step 3.5: If it changes colour but nothing on the plots change, then the model you selected has never been trained before, so you need to click train to add it to the library of trained models, then click run to use it as per step 3
- Step 4: Observe convergence plots. The two bottom plots illustrate convergence. The time series view lets you view the "face value" prediction quality 

General Notes:
- Everything is clickable, zooming in/out on the plots, double clicking to reset the view, turning legend items on and off, selecting the dropdowns etc
- Refreshing the page resets the page so you can start again if something goes wrong 

