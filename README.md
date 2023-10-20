# Heterogeneous Time-Series Sandbox

This repository shows the first attempt to approach heterogeneous time-series analysis, which was inspired by the following article:

**Authors:**
- Lukas Neubauer
- Peter Filzmoser

[Improving Forecasts for Heterogeneous Time Series by ”Averaging”, with Application to Food Demand Forecast](https://arxiv.org/pdf/2306.07119.pdf)

Authors describe heterogeneous time-series as a set of time-series data where each series may have different properties such as length, shape, or seasonality which can make them challenging to model in a unified manner​. 

The article proposes a framework for improving forecasts of heterogeneous time-series by leveraging Dynamic Time Warping (DTW) to find similar time series and averaging forecasts, especially useful for time-series which do not have long sample of data. The approach constructs neighborhoods of similar time series, allowing for better one-step ahead forecasts through averaging techniques. 

### Problem Proposal
We can take a step in the direction proposed by authors and try to solve the following problem: given little information about stocks, described by time-series with different properties, is it possible  to use methods such as DTW and Clustering to imply any information from prices?

As an idea, we can use these methods to clusters of stock which would represent different sectors/industries.

