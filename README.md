# Movie Revenue Analysis and Prediction

An end-to-end data science pipeline utilizing clustering algorithms and deep learning to analyze movie metadata and forecast box office performance.

## Overview
This project explores the relationship between film attributes (such as budget, runtime, and ratings) and commercial success. By combining unsupervised machine learning for movie segmentation with deep neural networks for regression, we demonstrate a structured approach to analyzing high-volatility financial datasets.

## Project Structure
- `Barekati-404422029-HW2.ipynb`: Jupyter notebook containing the full data pipeline.
- `Barekati-404422029-HW2.html`: Compiled HTML report of the findings.
- `movies_metadata.csv`: The primary dataset used for analysis.
- `Report.pdf` (Optional): The final academic summary of the findings.

## Key Phases
1. **Exploratory Data Analysis (EDA):** Performed cleaning and visualization (histograms, boxplots, scatter plots) to identify trends and outliers in financial metrics.
2. **Clustering Analysis:** Implemented **K-Means** ($k=3$) and **Hierarchical Clustering** ($k=2$) to segment movies based on their inherent characteristics.
3. **Predictive Modeling:** Developed a deep fully connected neural network (64-32-1 architecture) using TensorFlow/Keras.
    - **Optimization:** Addressed gradient instability by scaling the target variable (`revenue`).
    - **Performance:** Achieved an **R² score of 0.7234**, successfully explaining 72.3% of revenue variance.

## Key Technologies
- **Python:** Data manipulation and modeling.
- **Scikit-Learn:** Feature scaling, K-Means, Agglomerative Clustering, and metrics.
- **TensorFlow/Keras:** Deep neural network construction and training.
- **Seaborn/Matplotlib:** Statistical data visualization.
- **LaTeX:** Report typesetting.

## Metrics & Findings
| Metric | Value |
| :--- | :--- |
| **Optimal K (K-Means)** | 3 |
| **Optimal K (Hierarchical)** | 2 |
| **Test MSE** | 5,804,614,575,561,271.00 |
| **Test R² Score** | 0.7234 |

## Author
**Sepehr Barekati**
