# Las Vegas vs Phoenix Climate Similarity Report

## Executive conclusion

Las Vegas and Phoenix were evaluated using a **data-driven meteorological similarity procedure** based on their observed 2016–2023 weather data. The city pair was not selected solely from general geographic or climatological assumptions.

The final climate similarity score was **55.12/100**, giving the pair a classification of **Moderately similar** under the project's predefined scoring thresholds.

The purpose of this analysis is to establish whether the two cities represent sufficiently similar climatic regimes to serve as a controlled pair for evaluating cross-city LSTM generalization.

## 1. Data used

### Las Vegas
- Weather: `D:\Downloads\BTP\model_training\lstm_training_3\las_vegas_hourly_weather_2016_2023.csv`
- Energy: `D:\Downloads\BTP\model_training\lstm_training_3\las_vegas_hourly_energy.csv`

### Phoenix
- Weather: `D:\Downloads\BTP\model_training\lstm_training_3\phoenix_hourly_weather_2016_2023.csv`
- Energy: `D:\Downloads\BTP\model_training\lstm_training_3\phoenix_hourly_energy.csv`

The climate similarity score uses **weather only**. Energy is kept as the downstream prediction target and is not used to select the city pair.

## 2. Preprocessing

1. CSV schemas were inspected.
2. Timestamp columns were detected and converted to datetime.
3. Duplicate timestamps were removed after sorting.
4. Weather observations were restricted to the requested 2016–2023 period.
5. Only meteorological variables available in both cities were used.
6. The comparison was restricted to the common overlapping date window:
   **2016-01-01 00:00:00 to 2023-12-31 23:00:00**.
7. Variables were standardized where necessary before multivariate comparison.

## 3. Common meteorological variables

temperature, dew_point, relative_humidity, pressure, wind_speed, wind_direction, visibility, precipitation

## 4. Statistical evidence

### Variable-level comparison

| feature           |    LV_mean |   Phoenix_mean |   standardized_mean_difference |   wasserstein_distance |   normalized_wasserstein |   KS_statistic |   KS_p_value |   variable_similarity_score |
|:------------------|-----------:|---------------:|-------------------------------:|-----------------------:|-------------------------:|---------------:|-------------:|----------------------------:|
| temperature       |  19.849    |       23.8525  |                       0.392317 |                4.00353 |                 0.392317 |       0.141781 |            0 |                  73.64      |
| dew_point         |  -0.509473 |        5.28287 |                       0.78002  |                5.79234 |                 0.78002  |       0.316396 |            0 |                  53.3466    |
| relative_humidity |  31.1109   |       35.8288  |                       0.237098 |                4.72286 |                 0.237347 |       0.11534  |            0 |                  82.0764    |
| pressure          |  93.7795   |       97.3872  |                       7.01707  |                3.60767 |                 7.01707  |       0.999615 |            0 |                   0.0726058 |
| wind_speed        | 197.205    |      152.435   |                       0.455478 |               44.7697  |                 0.455478 |       0.302754 |            0 |                  65.5179    |
| wind_direction    | 197.205    |      152.435   |                       0.455478 |               44.7697  |                 0.455478 |       0.302754 |            0 |                  65.5179    |
| visibility        |  62.9209   |       58.3567  |                       0.14456  |               26.9753  |                 0.854379 |       0.417038 |            0 |                  62.4637    |
| precipitation     |  11.5588   |       13.8824  |                       0.343534 |                2.44011 |                 0.360766 |       0.162143 |            0 |                  74.8086    |

### Overall score components

| component                        |   score_0_to_100 |   weight |
|:---------------------------------|-----------------:|---------:|
| Distribution similarity          |          59.6805 |      0.6 |
| Correlation structure similarity |          85.7867 |      0.2 |
| Multivariate climate similarity  |          10.7813 |      0.2 |
| Overall climate similarity       |          55.1219 |      1   |

## 5. Interpretation

The variable-level analysis examines both central tendency and distributional shape. Standardized mean difference controls for measurement scale, while Wasserstein distance and the KS statistic provide distribution-level comparisons.

The correlation analysis additionally evaluates whether relationships among meteorological variables are similar between the cities.

The standardized multivariate distance provides a single geometry-based comparison in the common weather feature space.

## 6. Main findings

1. The analysis used 8 common meteorological variables: temperature, dew_point, relative_humidity, pressure, wind_speed, wind_direction, visibility, precipitation.
2. The comparison used the common observed weather period from 2016-01-01 to 2023-12-31, after timestamp normalization and duplicate removal.
3. The mean variable-level distribution similarity was 59.68/100.
4. The weather correlation-structure similarity was 85.79/100.
5. The standardized multivariate climate distance was 2.2274, corresponding to a multivariate similarity score of 10.78/100.
6. The final weighted climate similarity score was 55.12/100, classified as 'Moderately similar' using the predefined project thresholds.
7. The most similar weather variable by the combined variable score was relative_humidity (82.08/100).
8. The least similar weather variable by the combined variable score was pressure (0.07/100).


## 7. Why this supports the LSTM city-pair experiment

The intended model experiment is not simply to predict energy independently in two arbitrary cities. It is to investigate whether meteorological patterns learned in one city can generalize to another city with a similar climatic regime.

Therefore, the workflow is:

**Weather data → climate similarity analysis → city-pair selection → LSTM training → cross-city evaluation**

This makes Las Vegas and Phoenix a data-supported test case for model transfer/generalization, provided that the final score and supporting plots show sufficient similarity.

## 8. Professor-ready explanation

> We did not select Las Vegas and Phoenix arbitrarily. We first identified common meteorological variables in the two datasets and restricted the analysis to their overlapping 2016–2023 period. We compared their descriptive statistics, empirical weather distributions, and correlation structures. We then standardized the weather variables and calculated a multivariate climate distance. These independent measures were combined into a transparent similarity score. The resulting score was 55.12/100, classified as Moderately similar under our predefined thresholds. Therefore, the city pair was selected based on similarity in the observed meteorological feature space, and we then use the pair to test whether an LSTM trained on one city can generalize to another city under a comparable climate regime.

## 9. Important limitation

This analysis establishes **similarity under the selected meteorological variables and the available 2016–2023 data**. It does not claim that the two cities are identical climates in every climatological sense.

Similarity also does not guarantee identical electricity-demand behavior. Differences in population, building stock, utility infrastructure, socioeconomic activity, air-conditioning penetration, and local energy practices can cause different load responses even when weather is similar. Those factors are part of the reason cross-city model evaluation is meaningful.

## 10. Output files

All generated tables are stored in:

`D:\Downloads\BTP\model_training\lstm_training_3\climate_similarity_analysis\tables`

All generated figures are stored in:

`D:\Downloads\BTP\model_training\lstm_training_3\climate_similarity_analysis\plots`

This report is stored in:

`D:\Downloads\BTP\model_training\lstm_training_3\climate_similarity_analysis\report`
