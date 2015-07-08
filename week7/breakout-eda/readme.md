##Part 1: Exploratory Data Analysis (EDA)

Exploratory data analysis is a first crucial step to building predictive models from your data. EDA allows you
to confirm or invalidate some of the assumptions you are making about your data and understand relationships between your variables.
 
<br>
 
In this scenario, you are a data scientist at [Bay Area Bike Share](http://www.bayareabikeshare.com/). Your task
is to provide insights on bike user activity and behavior to the products team. 


1. Load the file `data/201402_trip_data.csv` into a dataframe. Provide the argument `parse_dates=['start_date', 'end_date']`
   with `pandas.read_csv()` to read the columns in as datetime objects. 
   
   Make 4 extra columns from the `start_date` column (We will use these in later questions):
   - `month` would contain only the month component
   - `dayofweek` would indicate what day of the week the date is
   - `date` would contain only the date component 
   - `hour` would only contain the hour component
   - [Hint to deal with datetime objects in pandas](http://stackoverflow.com/questions/25129144/pandas-return-hour-from-datetime-column-directly)

2. Group the bike rides by `month` and count the number of users per month. Plot the number of users for each month. 
   What do you observe? Provide a likely explanation to your observation. Real life data can often be messy/incomplete
   and cursory EDA is often able to reveal that.
   
3. Plot the daily user count from September to December. Mark the `mean` and `mean +/- 1.5 * Standard Deviation` as 
   horizontal lines on the plot. This would help you identify the outliers in your data. Describe your observations. 
   
   ![image](images/timeseries.png)

4. Plot the distribution of the daily user counts for all months as a histogram. Fit a 
   [KDE](http://glowingpython.blogspot.com/2012/08/kernel-density-estimation-with-scipy.html) to the histogram.
   What is the distribution and explain why the distribution might be shaped as such. 
    
   ![image](images/kde.png)
  
   Replot the distribution of daily user counts after binning them into weekday or weekend rides. Refit  
   KDEs onto the weekday and weekend histograms.
   
   ![image](images/weekdayweekend.png)

5. Now we are going to explore hourly trends of user activity. Group the bike rides by `date` and `hour` and count 
   the number of rides in the given hour on the given date. Make a 
   [boxplot](http://blog.bharatbhole.com/creating-boxplots-with-matplotlib/) of the hours in the day **(x)** against
   the number of users **(y)** in that given hour. 
   
6. Someone from the analytics team made a line plot (_right_) that he claims is showing the same information as your
   boxplot (_left_). What information can you gain from the boxplot that is missing in the line plot?
   
   ![image](images/q1_pair.png)

7. Replot the boxplot in `6.` after binning your data into weekday and weekend. Describe the differences you observe
   between hour user activity between weekday and weekend? 
    
8. There are two types of bike users (specified by column `Subscription Type`: `Subscriber` and `Customer`. Given this
   information and the weekend and weekday categorization, plot and inspect the user activity trends. Suppose the 
   product team wants to run a promotional campaign, what are you suggestions in terms of who the promotion should 
   apply to and when it should apply for the campaign to be effective?
   
   ![image](images/nonsub.png)
   ![image](images/customer.png)
   
9. **Extra Credit:** You are also interested in identifying stations with low usage. Load the csv file 
   `data/201402_station_data.csv` into a dataframe. The `docksize` column specifies how many bikes the station can hold. 
   The `lat` and `long` columns specify the latitude and longitude of the station. 
   
   - Merge the station data with the trip data
   - Compute usage by counting the total users starting at a particular station divided by the dockcount
   - Normalize usage to range from `0`to `1`
   - Using plotly, plot the latitude and longitude of the stations as scatter points, the usage will be indicated 
     by the transperancy and the size of the points
  
   ![scatter](images/plotly.png)
