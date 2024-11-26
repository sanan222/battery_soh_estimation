# Project Description
Over the years, both conventional and state-of-the-art techniques have been developed 
to improve parameter estimations. Traditional methods predominantly employ 
mathematical formulas and equivalent circuits, whereas contemporary approaches like 
sensor fusion, Kalman Filters, and Artificial Intelligence (AI) leverage data to 
comprehend battery behavior and make more precise predictions. However, the high 
nonlinearity in battery load levels and sensor noise complicate precise battery aging 
estimation, highlighting a significant research gap that demands attention.

To tackle the aforementioned challenges, this thesis focuses on using Artificial Neural 
Networks (ANN) to model battery aging behavior based on battery SOH values taking 
measured discharge capacity values as a ground truth. Prior to modelling, the battery 
dataset undergoes analysis as a time-series phenomenon, with data-based 
preprocessing techniques including noise cleaning, feature and cycle extraction, and 
outlier detection. By utilizing data-based AI approach, both model performance and 
computational efficiency are aimed to be enhanced, leading to the adoption of much 
simpler ANN models

# Results
| Evaluation Metric | Metric Value (Shallow Neural Network) | Metric Value (Deep Neural Network) |
|-------------------|---------------------------------------|------------------------------------|
| MSE               | 0.00025099                            | 0.000292455                       |
| MAE               | 0.006431                              | 0.00600695                        |
| RMSE              | 0.015842905                           | 0.017101318                       |
| R-squared (R²)    | 0.997557                              | 0.997154                          |

# Data Analysis

| ![Battery aging graph](https://github.com/user-attachments/assets/78e9a7ba-75c5-4834-b9a9-43e1775688f5) | ![Cycle extraction](https://github.com/user-attachments/assets/ed21169c-b014-49ca-bbdf-4b2b5a23dc0c) |
|------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| **Figure 1:** Battery aging graph for different cells with different C-rates | **Figure 2:** Cycle extraction using sliding windows technique |




