# Atlanta vs Birmingham Climate Similarity Report

## Executive Summary

Atlanta and Birmingham were evaluated using a data-driven meteorological similarity procedure. The purpose is to determine whether they form a defensible similar-climate pair for the cross-city LSTM experiment.

**Final climate similarity score: 62.49/100**

**Classification: Moderately similar**

Energy consumption was not used to define climate similarity.

## 1. Data

Atlanta: `D:\Downloads\BTP\model_training\lstm_training_2\Atlanta\atlanta_master.csv`

Birmingham: `D:\Downloads\BTP\model_training\lstm_training_2\Birmingham\birmingham_master.csv`

Common observed period: **2016-01-01 00:00:00 to 2023-12-31 23:00:00**

## 2. Common Weather Variables

temperature, dew_point, relative_humidity, pressure, wind_speed, wind_direction, visibility, precipitation

## 3. Methodology

1. Detect and normalize timestamps.
2. Remove duplicate timestamps.
3. Restrict both cities to their common observed period.
4. Identify weather variables available in both datasets.
5. Compare descriptive statistics.
6. Compare empirical weather distributions using standardized mean difference, Wasserstein distance and KS statistic.
7. Compare monthly and seasonal behavior.
8. Compare weather correlation structures.
9. Standardize weather variables and calculate multivariate climate distance.
10. Combine independent similarity measures into a transparent 0–100 project score.

## 4. Variable-Level Similarity

| feature           |   Atlanta_mean |   Birmingham_mean |   Atlanta_std |   Birmingham_std |        SMD |   Wasserstein_distance |   normalized_Wasserstein |   KS_statistic |   KS_p_value |   SMD_similarity |   Wasserstein_similarity |   KS_similarity |   variable_similarity_score |
|:------------------|---------------:|------------------:|--------------:|-----------------:|-----------:|-----------------------:|-------------------------:|---------------:|-------------:|-----------------:|-------------------------:|----------------:|----------------------------:|
| temperature       |       16.7739  |           16.7063 |      9.04635  |         9.39954  | 0.00732788 |               0.258867 |                0.0280626 |      0.0130565 |  1.28439e-05 |         99.2699  |                 97.2328  |         98.6943 |                    98.399   |
| precipitation     |       22.9466  |           22.7978 |     12.1912   |        10.7935   | 0.0129302  |               1.10087  |                0.0956147 |      0.0549372 |  2.32867e-92 |         98.7153  |                 90.8814  |         94.5063 |                    94.701   |
| dew_point         |        9.52828 |           10.3371 |      9.55336  |         9.40924  | 0.0853031  |               0.811201 |                0.0855556 |      0.0407677 |  4.86553e-51 |         91.8234  |                 91.8002  |         95.9232 |                    93.1823  |
| relative_humidity |       65.955   |           69.8845 |     20.505    |        20.7641   | 0.190431   |               3.92959  |                0.190434  |      0.100171  |  1.5576e-306 |         82.6602  |                 82.66    |         89.9829 |                    85.101   |
| wind_direction    |      200.339   |          145.29   |    104.521    |       119.258    | 0.490933   |              57.3909   |                0.511818  |      0.258961  |  0           |         61.2055  |                 59.9405  |         74.1039 |                    65.0833  |
| wind_speed        |      200.339   |          145.29   |    104.521    |       119.258    | 0.490933   |              57.3909   |                0.511818  |      0.258961  |  0           |         61.2055  |                 59.9405  |         74.1039 |                    65.0833  |
| visibility        |       18.8014  |           11.6723 |     10.1335   |         4.40516  | 0.912437   |               7.25675  |                0.928779  |      0.406193  |  0           |         40.1544  |                 39.5036  |         59.3807 |                    46.3462  |
| pressure          |       98.1292  |           99.5945 |      0.514528 |         0.532616 | 2.79814    |               1.46525  |                2.79814   |      0.859989  |  0           |          6.09231 |                  6.09231 |         14.0011 |                     8.72859 |

## 5. Overall Scores

| component                        |   score_0_to_100 |   weight |
|:---------------------------------|-----------------:|---------:|
| Distribution similarity          |          69.5781 |      0.6 |
| Correlation structure similarity |          89.6078 |      0.2 |
| Multivariate climate similarity  |          14.1108 |      0.2 |
| Overall climate similarity       |          62.4906 |      1   |

## 6. Main Findings

1. 8 common meteorological variables were available: temperature, dew_point, relative_humidity, pressure, wind_speed, wind_direction, visibility, precipitation.
2. The common comparison period was 2016-01-01 to 2023-12-31.
3. Average variable-level distribution similarity was 69.58/100.
4. Weather correlation-structure similarity was 89.61/100.
5. Standardized multivariate climate distance was 1.9582.
6. Multivariate similarity was 14.11/100.
7. Overall weighted climate similarity was 62.49/100, classified as Moderately similar.
8. The most similar variable was temperature (98.40/100).
9. The least similar variable was pressure (8.73/100).

## 7. Why this supports the LSTM experiment

The city pair was not selected simply because the cities are geographically close or are generally described as having similar climates.

Instead, the pairing is evaluated in the **observed meteorological feature space**. If the weather distributions, seasonal behavior, correlation structure and standardized multivariate climate position are sufficiently similar, the pair provides a meaningful test of whether an LSTM trained using one city's weather-demand relationship can generalize to another city.

The experimental logic is:

**Weather data → climate similarity analysis → city-pair selection → LSTM training → cross-city evaluation**

## 8. Professor / Viva Explanation

> We did not select Atlanta and Birmingham arbitrarily. We first identified the weather variables available in both datasets and restricted the comparison to their common observed period. We compared their descriptive statistics and empirical distributions using standardized mean difference, Wasserstein distance and the KS statistic. We also compared the correlation structure between the meteorological variables. Finally, we standardized the common weather variables and calculated a multivariate climate distance. These independent measurements were combined into a transparent similarity score of 62.49 out of 100, which gives a classification of Moderately similar. Therefore, the city pair is justified using observed meteorological similarity rather than relying only on a general external claim that the cities have similar climates.

## 9. Limitation

Climate similarity does not imply identical energy-demand behavior. Electricity consumption is also influenced by building stock, population, commercial activity, HVAC penetration, utility infrastructure and local operating practices.

Therefore, the similarity analysis justifies the **climate pairing**, while the LSTM experiment tests whether similar meteorological regimes actually result in transferable energy-demand relationships.

## 10. Output Locations

Tables: `D:\Downloads\BTP\model_training\lstm_training_2\Atlanta\atlanta_birmingham_climate_similarity\tables`

Plots: `D:\Downloads\BTP\model_training\lstm_training_2\Atlanta\atlanta_birmingham_climate_similarity\plots`

Markdown report: `D:\Downloads\BTP\model_training\lstm_training_2\Atlanta\atlanta_birmingham_climate_similarity\report\atlanta_birmingham_climate_similarity_report.md`

HTML report: `D:\Downloads\BTP\model_training\lstm_training_2\Atlanta\atlanta_birmingham_climate_similarity\report\atlanta_birmingham_climate_similarity_report.html`
