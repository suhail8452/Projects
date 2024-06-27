### First SQL project

# Show the first 10 rows
select * from  eurolotto_data
limit 10;

# Show the number of rows
select count(*) from eurolotto_data;

# Describe the data in the table
Describe eurolotto_data;

# The frequency of picking ball1 and ball2
SELECT Ball1, Ball2, COUNT(*) AS Frequency
FROM eurolotto_data
GROUP BY Ball1, Ball2
ORDER BY Frequency DESC;


# The sum of uk winners grouped by the year
create table totalUKwinners as
select substring(DrawDate,7,7) as Date,
sum(Totals_UKWinners) as Total_UKWinners
from eurolotto_data
group by Date;

# The sum of Total winners grouped by the year
create table TotalWinners as
select substring(DrawDate,7,7) as Date,
sum(Totals_TotalWinners) as Total_Winners
from eurolotto_data
group by Date;

# The sum of Total UK prize grouped by the year
create table totalUKPrize as
select substring(DrawDate,7,7) as Date,
round(sum(Totals_UKPrizeFund)) as Total_UKPrize
from eurolotto_data
group by Date;


# Get rid of ball_frequency table
drop table ball_frequency;

#Create a new table called ball_frequency table
# Refresh tables
CREATE TABLE ball_frequency (
    number_value INT,
    frequency INT
);

# Find the frequency of each number of the five balls
# values are 1-50
INSERT INTO ball_frequency (number_value, frequency)
SELECT number_value, COUNT(*) AS frequency
FROM (
	SELECT Ball1 AS number_value FROM eurolotto_data
	UNION ALL
	SELECT Ball2 FROM eurolotto_data
	UNION ALL
	SELECT Ball3 FROM eurolotto_data
	UNION ALL
	SELECT Ball4 FROM eurolotto_data
	UNION ALL
	SELECT Ball5 FROM eurolotto_data
) AS combined_values
GROUP BY number_value;

#Show the table ball_frequency in ascending order
select * from ball_frequency
order by frequency ASC;

# Get rid of frequency table
drop table lucky_frequency;

#Create a new table called frequency table
# Refresh tables
CREATE TABLE lucky_frequency (
    number_value INT,
    frequency INT
);

# Find the frequency of each number of the lucky star balls.
INSERT INTO lucky_frequency (number_value, frequency)
SELECT number_value, COUNT(*) AS frequency
FROM (
	SELECT LuckyStar1 AS number_value FROM eurolotto_data
	UNION ALL
	SELECT LuckyStar2 FROM eurolotto_data
) AS combined_values
GROUP BY number_value;

#Show the table ball_frequency in ascending order
# lucky star values are 1-12
select * from lucky_frequency
order by frequency ASC;
