# Phishing Detection Model Report

## Introduction
This report details the development and evaluation of a machine learning model designed to detect phishing emails using a provided dataset. The objective was to construct a robust binary classification model capable of distinguishing between safe and phishing emails based on extracted features. The following sections outline the approach taken, assumptions made, results achieved, challenges encountered, potential improvements, and the strategy for tracking model performance over time.

## Approach
The development process adhered to a systematic machine learning pipeline. Initially, the dataset was loaded and inspected, confirming the absence of missing values. Exploratory Data Analysis (EDA) was conducted to evaluate feature distributions, correlations, and class balance, revealing significant skewness and class imbalance. To address these issues, a `LogTransformer` was applied to reduce skewness in feature distributions, followed by `RobustScaler` to scale the data and mitigate the impact of outliers while preserving all data points.

Several machine learning algorithms were assessed, including Random Forest, Gradient Boosting, XGBoost, LightGBM, and Logistic Regression. Hyperparameter optimization was performed using `GridSearchCV`, with Area Under the Precision-Recall Curve (AUPRC) selected as the primary evaluation metric due to its appropriateness for imbalanced datasets. To ensure robustness and prevent overfitting, the dataset was divided into training, validation, and holdout sets. The Random Forest model was ultimately chosen based on its superior validation AUPRC, with final performance evaluated on the holdout set.

## Assumptions
The model development rested on several key assumptions:
- The dataset contains no missing values, as verified during initial inspection.
- All features are numerical and non-negative, consistent with their real-world interpretations (i.e. word counts, link counts).
- The dataset is representative of typical phishing and safe emails, enabling the model to generalize effectively to similar distributions.

These assumptions are grounded in the dataset's structure and the problem context, providing a solid foundation for the modeling process.

## Results
The Random Forest model demonstrated exceptional performance, achieving a validation AUPRC of 0.9945. On the holdout set, with a decision threshold of 0.8369, the model yielded the following metrics:
- **Precision (Safe)**: 0.91
- **Recall (Safe)**: 0.99
- **F1-score (Safe)**: 0.95
- **Precision (Phishing)**: 0.99
- **Recall (Phishing)**: 0.90
- **F1-score (Phishing)**: 0.94
- **ROC-AUC**: 0.9951

The confusion matrix provides further insight into the model's classification performance:
```
[[76881   804]
 [ 7596 70215]]
```
- **True Negatives (TN)**: 76,881 (safe emails correctly identified)
- **False Positives (FP)**: 804 (safe emails misclassified as phishing)
- **False Negatives (FN)**: 7,596 (phishing emails misclassified as safe)
- **True Positives (TP)**: 70,215 (phishing emails correctly identified)
- **Support**: 77,685 safe, 77,811 phishing (total 155,496 samples)

Feature importance analysis highlighted the top three contributors to the model's predictions:
1. `num_unique_words` (importance: 0.2194)
2. `num_stopwords` (importance: 0.2119)
3. `num_words` (importance: 0.2083)

These features offer meaningful insights into phishing email characteristics:
- **Higher `num_words`** (phishing mean: 317.17 vs. safe mean: 277.29) indicates that phishing emails tend to be longer.
- **Higher `num_stopwords`** (phishing mean: 103.29 vs. safe mean: 80.57) suggests greater use of common words in phishing attempts.
- **Higher `num_unique_words`** (phishing mean: 157.73 vs. safe mean: 124.25) reflects increased lexical diversity in phishing emails.

## Challenges
Two significant challenges arose during model development, each requiring careful consideration and resolution:
1. **Class Imbalance**: The dataset initially exhibited a severe imbalance, with 98.676% safe emails (label 0) and only 1.324% phishing emails (label 1). This was addressed using `ADASYN` to oversample the minority class, achieving a balanced holdout set (77,685 safe, 77,811 phishing). This resampling strategy improved the model's ability to detect phishing emails without skewing toward the majority class.
2. **Outliers**: Outliers were prevalent across multiple features, with counts ranging from 36,226 to 103,013 instances:
   - `num_words`: 46,375 outliers
   - `num_unique_words`: 36,226 outliers
   - `num_stopwords`: 45,847 outliers
   - `num_links`: 93,422 outliers
   - `num_unique_domains`: 93,422 outliers
   - `num_email_addresses`: 76,327 outliers
   - `num_spelling_errors`: 52,217 outliers
   - `num_urgent_keywords`: 103,013 outliers

   These were managed using `RobustScaler`, which minimizes the influence of extreme values while retaining all data points. This approach leverages the inherent robustness of tree-based models like Random Forest. Alternative strategies, such as outlier removal or capping, were evaluated but not implemented to avoid potential loss of critical information, particularly if outliers represent valid phishing behaviors. The decision threshold was also adjusted to 0.8369 to balance precision and recall, targeting a recall of 0.9 for phishing emails while maintaining high precision.

## Improvements
Several opportunities exist to enhance the model's performance in future iterations:
- **Advanced Models**: Exploring neural networks or ensemble techniques (i.e. stacking) could capture more complex patterns within the data.
- **Feature Engineering**: Incorporating additional features, such as email metadata or text embeddings derived from NLP models, could enrich the feature set and improve detection accuracy.
- **Imbalance Handling**: If class distributions shift in future datasets, techniques like SMOTE or cost-sensitive learning could be adapted to maintain model effectiveness.

These enhancements would require additional resources and data but could yield incremental improvements in performance.

## Tracking
To ensure the model's long-term reliability, a comprehensive monitoring strategy has been established:
- **Drift Detection**: Kolmogorov-Smirnov (KS) tests are employed to monitor shifts in feature distributions, providing early warnings of data drift.
- **Performance Monitoring**: The model's AUPRC is tracked continuously on a validation set, with retraining initiated if the metric falls below 0.9.
- **Periodic Evaluations**: Regular assessments using newly labeled data are conducted to verify the model's relevance and accuracy in the face of evolving phishing tactics.

This strategy ensures proactive maintenance of the model's effectiveness over time.

## Conclusion
The phishing detection model developed in this project exhibits strong performance, effectively identifying phishing emails with high precision and recall. By addressing key challenges such as class imbalance and outliers, and employing a rigorous evaluation process, the model is well-suited for practical deployment. Future efforts will focus on integrating advanced techniques and additional features to further refine its capabilities.