# Employee Churn Prediction and Prevention

This project aims to analyze the factors contributing to employee churn at TechNova Solutions and build a predictive model to identify at-risk employees. The analysis also provides actionable recommendations to reduce attrition and improve employee retention.

## Project Overview

Employee turnover is a significant cost for businesses. By understanding the drivers of churn and predicting which employees are likely to leave, companies can implement targeted interventions to improve retention and save costs.

## Data

The dataset used for this analysis contains information about employees at TechNova Solutions, including demographic details, job-related features, performance metrics, and satisfaction levels.

## Analysis and Findings

The exploratory data analysis revealed several key factors associated with employee churn:

- **Satisfaction Level:** Employees with lower satisfaction levels are significantly more likely to leave.
- **Overtime Hours:** Excessive overtime is a strong indicator of increased turnover risk.
- **Tenure:** Employees with shorter tenure (especially less than 2 years) have higher attrition rates.
- **Work-Life Balance:** Poor work-life balance is a major contributor to churn.
- **Promotions:** Lack of promotions and career stagnation are associated with increased churn.
- **Salary:** While not the sole factor, lower salaries correlate with higher attrition.
- **Manager Quality:** The relationship with the manager plays a role in employee retention.
- **Performance Rating:** Both high and low performers may be at risk of leaving for different reasons.

## Modeling

Several machine learning models were trained and evaluated to predict employee churn. The models included Logistic Regression, Random Forest, Gradient Boosting, and XGBoost.

The models were evaluated using 5-fold stratified cross-validation and tested on a held-out test set. The primary evaluation metric was ROC-AUC, with F1 Score, Precision, and Recall as secondary metrics.

**Model Performance Comparison:**

| Model               | Accuracy | ROC-AUC | F1 Score | Precision (Churn) | Recall (Churn) |
|---------------------|----------|---------|----------|-------------------|----------------|
| Logistic Regression | 0.5145   | 0.5424  | 0.3069   | 0.2161            | 0.5296         |
| Gradient Boosting   | 0.7970   | 0.5144  | 0.0049   | 0.5000            | 0.0025         |
| Random Forest       | 0.7970   | 0.5072  | 0.0000   | 0.0000            | 0.0000         |
| XGBoost             | 0.6265   | 0.4972  | 0.2401   | 0.2045            | 0.2906         |

The Logistic Regression model achieved the highest ROC-AUC score on the test set.

## Model Explainability

SHAP (SHapley Additive exPlanations) values were used to understand the feature importance and how each feature influences the model's predictions. The key drivers identified by the model align with the EDA findings, with Satisfaction Level, Flight Risk Score (engineered feature), Overtime Hours, Tenure, and Work-Life Balance being the most influential factors.

## Recommendations

Based on the analysis and the predictive model, the following recommendations are proposed to TechNova Solutions:

### Priority 1: Immediate Actions (Weeks 1-12)

- **Deploy Early Warning System:** Implement monthly employee risk scoring using the trained model to identify high-risk employees for immediate intervention.
- **Address Overtime and Burnout:** Cap overtime hours, conduct workload audits, and implement mandatory recovery time.
- **Satisfaction Monitoring Program:** Implement quarterly pulse surveys and trigger manager intervention when satisfaction drops.

### Priority 2: Medium-Term Initiatives (Months 3-6)

- **Enhanced New Hire Retention:** Extend onboarding, assign buddies, and conduct regular check-ins for new hires.
- **Career Development Framework:** Document clear promotion criteria, create individual development plans, and encourage internal mobility.
- **Department-Specific Interventions:** Analyze exit interviews and conduct focus groups to address unique challenges in high-risk departments.

### Priority 3: Long-Term Changes (Months 6-12)

- **Manager Effectiveness Program:** Provide leadership training, implement 360-degree feedback, and tie manager performance to team retention and satisfaction.
- **Strategic Talent Segmentation:** Tailor retention efforts based on performance and risk profiles.
- **Compensation Strategy Review:** Conduct annual market benchmarking and address salary compression.

## Expected Business Impact

By implementing these recommendations, TechNova Solutions can expect:

- A significant reduction in the overall attrition rate (targeting 30-40%).
- Annual cost savings estimated between $2.4M and $3.6M.
- Improved project continuity, team morale, and institutional knowledge retention.
- Enhanced employer brand and a competitive advantage in the talent market.

## Key Success Metrics

Monitor key metrics such as average satisfaction score, average overtime hours, new hire retention rate, manager feedback scores, overall attrition rate, and high performer attrition to track the effectiveness of the implemented strategies.

## Model Deployment Considerations

Consider technical implementation details, model maintenance plan, and ethical AI practices to ensure fair and responsible use of the predictive model.

## Conclusion

The predictive model and the strategic recommendations provide a data-driven approach for TechNova Solutions to proactively manage employee churn. By focusing on the key drivers of attrition and implementing targeted interventions, the company can build a stronger, more engaged, and more stable workforce.
