# Comprehensive Execution Framework for Cost-Sensitive Predictive Modeling in Marketing Acquisition

The execution of predictive modeling tasks within a strictly constrained, cost-sensitive environment fundamentally diverges from traditional machine learning paradigms. When an analytical problem assigns an explicit financial or point-based penalty to feature acquisition alongside varying weights for classification errors, the overarching objective transitions from maximizing statistical accuracy to optimizing a highly specific, multidimensional business profit curve. The task of identifying customers most likely to convert on a marketing offer out of 5000 test clients, subject to a 1000-client targeting cap and a severe variable penalty, presents a complex multi-objective optimization problem. Standard models optimized for surrogate loss functions, such as binary cross-entropy or hinge loss, operate under the assumption that all misclassifications carry equal weight and that the data utilized to make predictions is entirely cost-free. Applying such models out-of-the-box to this dataset will inevitably result in suboptimal or entirely negative performance.

The following analysis delineates an exhaustive, end-to-end operational framework for constructing a maximal-utility classification model under the provided mathematical constraints. This strategic plan is designed to be executed by a three-person technical unit over a rigid four-week timeline, culminating in the submission of predictions, code, a professional LaTeX report, and a structured presentation by June 8, 2026. The framework covers the mathematical derivation of the strategic approach, the precise feature selection pipeline required to navigate the variable penalties, the modeling and threshold tuning mechanisms necessary for imbalanced cost-sensitive data, and a comprehensive project management breakdown to ensure all deliverables secure the maximum possible academic evaluation.

## Strategic Approach and Custom Loss Function Analysis

Traditional classification algorithms evaluate performance based on metrics like Area Under the Receiver Operating Characteristic Curve (ROC-AUC) or the F1-Score. However, the specific problem architecture presented introduces a dual-layered cost structure that completely invalidates these traditional optimization metrics. The business metric must be the singular guiding mechanism for all algorithmic training, hyperparameter tuning, and feature selection decisions.

### The Mathematical Intuition of the Trade-Offs

The objective function to be maximized is explicitly defined as a combination of True Positives ($TP$), False Positives ($FP$), and the absolute number of unique variables ($|F|$) retained in the final model:

$$Score = (TP \times 10) - (FP \times 5) - (|F| \times 200)$$

This equation establishes an asymmetric classification cost matrix combined with an extreme $L_0$ regularization penalty. To understand the operational reality of this function, the marginal utility of a single predictive feature must be calculated. For a feature to be mathematically justified in the final model, its inclusion must increase the gross score by strictly more than 200 points. Given the reward and penalty structure, an increase of 200 points requires the feature to generate either 20 additional True Positives ($20 \times 10 = 200$) without increasing False Positives, or prevent 40 False Positives ($40 \times 5 = 200$) without decreasing True Positives, or some linear combination thereof that exceeds the 200-point threshold.

Furthermore, the hard constraint of 1000 maximum targets acts as a capacity ceiling. If the marketing campaign targets the maximum 1000 clients, the sum of True Positives and False Positives exactly equals 1000 ($TP + FP = 1000$). Substituting this reality into the gross score function (before feature penalties) yields:

$$Gross = TP \times 10 - (1000 - TP) \times 5$$
$$Gross = 10TP - 5000 + 5TP = 15TP - 5000$$

To achieve a positive gross score (greater than zero before any feature penalties are applied), the model must identify at least 334 True Positives ($15 \times 334 = 5010$). This indicates that the precision of the top 1000 targets must strictly exceed 33.33%. Any prediction with an expected probability of conversion lower than 0.333 will yield a negative mathematical expectation and actively diminish the final score. This reality mandates that the modeling phase prioritize high-precision probability ranking over widespread recall.

| Metric Component | Value | Operational Implication |
| :--- | :--- | :--- |
| True Positive Reward | +10 | Each correct identification yields high reward, but requires 33.3% precision to offset FP risks. |
| False Positive Penalty | -5 | Targeting non-converting clients drains the score; precision is favored over recall. |
| Variable Cost | -200 | Every feature must pay for itself by generating at least 20 net new TPs or removing 40 FPs. |
| Capacity Constraint | 1000 Max | The model must focus exclusively on the highest-probability decile of the test set. |

### Optimal Feature Retention Target

Given the punitive cost of 200 points per feature, the dimensionality of the final model must be ruthlessly minimized. To illustrate the extreme pressure this places on dimensionality reduction, consider the theoretical maximum score. If the test set of 5000 clients contains 1000 actual positives, and the model perfectly identifies all of them while utilizing 50 features, the score would be:

$$(1000 \times 10) - (0 \times 5) - (50 \times 200) = 10000 - 10000 = 0$$

A model utilizing 50 features incurs a -10,000 point penalty, an insurmountable deficit that guarantees failure regardless of the model's predictive supremacy. Even at 20 features, the model begins with a -4000 point handicap. The analysis indicates that the optimal number of features for this specific architecture falls within the highly sparse range of 3 to 8 variables. A model utilizing 5 features incurs a manageable -1000 point penalty, requiring only 100 True Positives to break even. The strategic imperative of this project is not to find all moderately predictive features, but to isolate only the absolute highest-variance, orthogonal predictors that capture the vast majority of the systemic signal.

### Custom Business Metric Implementation in Cross-Validation

To optimize the machine learning algorithms effectively, the exact business metric must be implemented as a custom scorer utilizing scikit-learn's API. Standard estimators and cross-validation tools like `GridSearchCV` default to accuracy, ROC-AUC, or log-loss, none of which account for the 200-point variable penalty or the asymmetric 10 / -5 cost matrix.

Implementing this requires careful handling of the scikit-learn scoring signature. While the `make_scorer` utility function typically wraps scoring metrics that only require `y_true` and `y_pred`, those standard metrics do not have access to the number of features utilized by the model pipeline. To bypass this limitation, a direct scoring callable must be constructed with the specific signature `scorer(estimator, X, y_true)`. This signature allows the scoring function to intercept the fitted estimator and the test matrix `X` during each cross-validation fold, extract the number of features passing through the pipeline, calculate the confusion matrix, and output the final profit score.

By passing this custom callable directly into the `scoring` parameter of `GridSearchCV` or Optuna, the hyperparameter tuning process is forced to evaluate the model exactly as the final grading mechanism will. The optimization algorithms will mathematically punish overly complex pipelines and aggressively reward sparse, highly precise models. A complete implementation of this custom scorer is provided at the conclusion of this report.

## Feature Selection Strategy

The dataset consists of 5000 training records and 500 anonymized variables. This represents a moderate-to-high dimensionality ratio ($p/n = 0.1$) which natively introduces the risk of overfitting, exacerbated immensely by the 200-point financial penalty assigned to each variable. Machine learning models, particularly tree-based ensembles like Random Forests, are robust to high dimensionality and will happily split on hundreds of weak predictors to marginally improve training loss. However, they do not naturally reduce their feature space to the extreme sparsity required by this business case. A multi-stage, aggressive feature elimination workflow is strictly mandated, combining filter, embedded, and wrapper methods to distil the 500 variables down to the optimal 3 to 8.

### Recommended Dimensionality Reduction Techniques

To isolate the absolute most predictive features, a combination of linear and non-linear feature selection techniques must be deployed sequentially. Relying on a single technique risks discarding valuable non-linear interactions or retaining highly correlated, redundant variables.

1. **Strict Multicollinearity Filtering (Filter Method)**
   Before any predictive modeling begins, informational redundancy must be eliminated. If two variables are highly correlated, they provide overlapping informational value to the model. Retaining both incurs a 400-point penalty for the predictive power of essentially one variable. Pairwise Pearson and Spearman correlation matrices must be computed across the 500 continuous and categorical variables. Additionally, Variance Inflation Factor (VIF) calculations should be utilized to detect multi-collinear clusters. For any pair or cluster of features exhibiting high correlation (e.g., coefficient $\ge 0.85$), the algorithm must discard all but the single variable possessing the highest univariate association with the target label. This purely statistical step ensures the algorithm does not pay multiple times for the same underlying signal.

2. **$L_1$ Regularization / LASSO (Embedded Method)**
   Logistic Regression equipped with an $L_1$ penalty (LASSO) serves as an exceptionally powerful embedded feature selector. Unlike $L_2$ regularization (Ridge), which merely shrinks coefficients toward zero to prevent overfitting, the $L_1$ norm mathematically forces the coefficients of less important or redundant variables to become exactly zero. By executing a regularization path algorithm—sweeping the inverse regularization strength parameter ($C$) from weak to exceptionally strong—the feature space can be smoothly and aggressively reduced from 500 down to a highly concentrated subset of 15-30 variables. LASSO evaluates the global linear contribution of features, making it an ideal secondary filter to strip away weak noise.

3. **Recursive Feature Elimination with Cross-Validation (Wrapper Method)**
   Recursive Feature Elimination (RFE) operates by iteratively training a supervised model, ranking the features by importance (e.g., node impurity in decision trees or coefficient magnitude in linear models), and discarding the weakest fraction of features. RFECV automates this process by determining the optimal stopping point via cross-validation. However, to succeed in this specific environment, the internal scoring mechanism of the RFECV algorithm must be overridden and configured to use the custom business profit scorer. Standard RFECV attempts to maximize accuracy or AUC, which will retain far too many features. By injecting the business metric, the recursion will strictly halt when the marginal penalty of retaining an additional feature (200 points) mathematically outweighs its marginal classification gain.

4. **Permutation Feature Importance (Post-hoc Method)**
   For tree-based models, standard impurity-based feature importances are notoriously biased toward high-cardinality features (variables with many unique values). Permutation Importance evaluates the strict marginal utility of a feature by randomly shuffling its values in the validation set and measuring the exact drop in the custom business metric. If shuffling a feature causes the model's profit score to drop by only 150 points, that feature is a net-negative asset (since it costs 200 points to retain) and must be eliminated. This technique is ideal for the final, surgical pruning phase to break ties and isolate the final 3 to 8 variables.

### Step-by-Step Feature Selection Workflow

The workflow to reduce 500 variables to the optimal subset operates as a sequential, narrowing funnel. Each stage serves a specific mathematical purpose, systematically shedding variables until only the absolute core drivers of conversion remain.

| Workflow Stage | Technique | Input Size | Target Output Size | Primary Objective |
| :--- | :--- | :--- | :--- | :--- |
| Stage 1: Variance Filtering | Variance Threshold | 500 | ~ | Remove constants and zero-variance predictors that offer zero discriminative power. |
| Stage 2: Correlation Pruning | Pearson/Spearman & VIF | ~ | ~ | Eliminate extreme multicollinearity. Retain only one variable per highly correlated cluster. |
| Stage 3: Linear Embedded | $L_1$ Regularization (LASSO) | ~ | 20 | Apply severe penalty coefficients to force weak, linear noise variables to exactly zero. |
| Stage 4: Algorithmic Wrap | RFECV with Custom Scorer | 20 | 5 | Iteratively drop variables until the custom profit curve peaks, using a non-linear estimator. |
| Stage 5: Marginal Utility Pruning | Permutation Importance | 5 | 3 | Manually verify that every remaining variable generates at least +200 points in isolated predictive value. |

This rigorous, five-stage architecture ensures that the final predictive model is not burdened by the immense financial penalties of extraneous variables. By the end of Stage 5, the feature space will be hyper-optimized for the unique mathematical realities of the project.

## Modeling and Threshold Tuning

The modeling phase in a cost-sensitive, imbalanced framework must prioritize calibrated probability outputs over hard classification boundaries. Because the cost of a False Positive is significantly different from the reward of a True Positive, treating all probabilities above 0.5 as a positive prediction is mathematically flawed. The algorithms selected must be capable of generating highly accurate, ranked probabilities that can be dynamically thresholded to maximize the profit curve.

### Optimal Machine Learning Algorithms

For tabular data containing 5000 training records, complex architectures such as deep neural networks will likely overfit, require excessive computational resources, and lack the necessary transparency. The following algorithms are optimally suited for this highly constrained task:

1. **XGBoost (eXtreme Gradient Boosting)**
   Gradient Boosted Decision Trees (GBDT) are universally recognized as the dominant algorithmic class for tabular classification tasks. XGBoost builds a sequence of shallow decision trees, where each subsequent tree attempts to correct the residual errors of the previous sequence. For this specific marketing acquisition problem, XGBoost handles non-linear interactions natively and is highly resistant to outliers. Most critically, it processes imbalanced data effectively through the `scale_pos_weight` hyperparameter. By adjusting this parameter, the algorithm's gradient descent process can be forced to heavily penalize errors on the minority class (the clients who actually converted), ensuring the model learns the nuanced signals of a successful conversion.

2. **LightGBM (Light Gradient Boosting Machine)**
   Developed by Microsoft, LightGBM is a highly efficient alternative to XGBoost. It differs fundamentally in its tree-growth strategy, utilizing leaf-wise (best-first) growth rather than level-wise growth. This allows LightGBM to minimize loss much faster and often results in higher accuracy on complex datasets. Furthermore, it employs Gradient-based One-Side Sampling (GOSS), which retains instances with large gradients while randomly sampling instances with small gradients, and Exclusive Feature Bundling (EFB), which bundles mutually exclusive features to reduce dimensionality. When the dataset is reduced to under 10 features, LightGBM is capable of mapping incredibly complex, high-signal decision boundaries.

3. **Logistic Regression (The Linear Baseline)**
   While often dismissed as overly simplistic compared to modern tree-based ensembles, Logistic Regression is an exceptionally powerful tool when the feature space is extremely sparse and highly restricted. Once the non-linear noise is filtered out during the aggressive feature selection phase, the remaining core variables often exhibit a direct, linear relationship with the log-odds of the target variable. Logistic Regression is entirely immune to the high-variance overfitting that plagues deep decision trees on small datasets, and its outputs natively approximate well-calibrated probabilities. It serves as an essential, interpretable benchmark against which the boosting models must be compared.

### Decision Threshold Optimization and Profit Maximization

Standard classification algorithms utilize a default decision threshold of 0.5. Predicting a positive class only when the model is greater than 50% confident assumes that False Positives and False Negatives carry equal weight, and that the distribution of classes is perfectly balanced. The provided scoring function completely dismantles this assumption. To maximize the profit curve, the exact break-even probability threshold must be mathematically derived and applied.

The Expected Value ($EV$) of a single positive prediction is defined by the probability of it being a True Positive ($p$) multiplied by the financial reward, plus the probability of it being a False Positive ($1-p$) multiplied by the financial cost:

$$EV = p \cdot (Reward_{TP}) + (1-p) \cdot (Cost_{FP})$$

Applying the specific weights from the project guidelines:

$$EV = p \cdot (10) + (1-p) \cdot (-5)$$
$$EV = 10p - 5 + 5p$$
$$EV = 15p - 5$$

Setting the expected value to zero calculates the absolute break-even point:

$$15p - 5 = 0 \Rightarrow 15p = 5 \Rightarrow p = \frac{5}{15} = 0.3333$$

Mathematical logic dictates that a marketing offer should be sent to a client if and only if the model computes their probability of conversion to be strictly greater than 33.33%. Predicting a positive class for a client with a 40% probability of conversion yields a positive expected value ($0.4 \times 10 - 0.6 \times 5 = 4 - 3 = +1$ point), even though standard models would automatically label this as a negative prediction because 0.4 is less than 0.5.

### Probability Calibration and Ranking Constraints

Because this analytical threshold relies on precise mathematical expectations, the output of the machine learning algorithms must represent true statistical probability. Tree-based models like XGBoost and LightGBM frequently output distorted probabilities—they are excellent at ranking clients, but terrible at estimating the true likelihood of an event, often pushing probabilities artificially toward 0 or 1. Therefore, Probability Calibration must be applied. Utilizing scikit-learn's `CalibratedClassifierCV`, the raw outputs of the boosting models can be mapped to true probabilities using Isotonic Regression (a non-parametric approach) or Platt Scaling (Sigmoid calibration). This ensures that when the model outputs a score of 0.35, that client genuinely has a 35% chance of converting.

Once the model is thoroughly calibrated, the final prediction workflow follows a strict, maximum target constraint:

1. Generate calibrated probabilities for all 5000 clients in the test set.
2. Sort the 5000 clients in descending order based on their predicted probability of conversion.
3. Apply the capacity constraint: Select the top $K$ clients, where $K \le 1000$.
4. Apply the probability constraint: Iterate down the ranked list and ensure that the $K$-th client has a probability $p > 0.3333$. If the 800th ranked client has a probability of 0.32, the targeting must halt entirely at 799. The model should absolutely not blindly target 1000 clients if the lower echelon of that group possesses a negative expected value.

This explicit threshold optimization can be algorithmically automated utilizing tools such as `TunedThresholdClassifierCV`, which dynamically evaluates various threshold cut-offs to explicitly maximize the custom profit curve defined by the user.

## Project Timeline and Task Delegation

Effective execution within a strict four-week deadline (terminating on June 8, 2026) requires a highly parallelized, Agile-inspired project management framework. The division of labor must isolate distinct domains of the machine learning pipeline to prevent bottlenecking, while ensuring seamless integration through shared code repositories (e.g., GitHub) and standardized data formats.

The workload is logically distributed among the three team members based on functional domains, allowing each student to take ownership of a critical phase of the data lifecycle.

**Student A (Data Infrastructure & Baselines)**: Assumes the role of Data Engineer. Responsible for exploratory data analysis (EDA), missing value imputation, categorical encoding, multicollinearity filtering, and the establishment of baseline models (e.g., Dummy Classifiers and unoptimized Logistic Regression) to establish the minimum viable signal.

**Student B (Advanced Feature Optimization & Ensembling)**: Assumes the role of Core Machine Learning Engineer. Responsible for executing the rigorous feature selection pipeline (LASSO, RFECV, Permutation Importance), managing hyperparameter tuning grids via Optuna, and training the advanced gradient boosting models (XGBoost and LightGBM).

**Student C (Business Metrics, Calibration, & Synthesis)**: Assumes the role of Product Manager and Applied Scientist. Responsible for programming the custom metric scoring objective, calibrating the model probabilities, mapping the threshold optimization profit curves, compiling the final LaTeX report, and structuring the presentation slide deck.

### Four-Week Execution Timeline

The following table breaks down the project into a highly structured, week-by-week timeline, ensuring continuous progression toward the June 8 deadline without last-minute integration panics.

| Project Phase | Timeline | Primary Objectives | Student A: Data & Baselines | Student B: ML & Features | Student C: Metrics & Synthesis |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Phase 1: Foundation | Week 1 | Establish Git workflows, clean the data, and construct baseline models. | Execute missing value imputation, Correlation filtering, and VIF analysis to cut obvious noise. | Train default, unoptimized tree models to gauge raw signal across the full 500 features. | Code the custom `make_scorer` metric, establish proper Stratified K-Fold validation splits. |
| Phase 2: Reduction | Week 2 | Isolate the core predictive variables (Dimensionality Reduction). | Engineer grouped features if applicable, scale numeric data for linear models. | Execute the LASSO and RFECV pipelines, reducing the feature space from 500 to <20. | Monitor the custom metric performance against the feature count, enforcing the 200-point penalty. |
| Phase 3: Optimization | Week 3 | Maximize the profit curve and calibrate the final model outputs. | Audit the selected 5-10 features for logical consistency and statistical stability. | Hyperparameter tuning (Optuna) on the final, highly restricted feature set. | Calibrate probabilities, calculate optimal thresholds ($p > 0.3333$), plot the Profit Curve. |
| Phase 4: Deliverables | Week 4 | Finalize outputs, write the report, and rehearse the presentation. | Compile the "Variables used" and "Predictions" output files (max 1000 rows). | Clean, modularize, and heavily document the final Python codebase. | Typeset the 5-page LaTeX report, design the 7-minute slide deck, lead presentation rehearsals. |

## Report and Presentation Outline

Securing the maximum allocation of academic points (15 points) necessitates deliverables that are rigorously structured, mathematically sound, and visually professional. Academic grading rubrics for machine learning projects heavily weight the clarity of communication, the justification of architectural decisions, and adherence to strict formatting paradigms.

### 5-Page LaTeX Report Table of Contents

The report must be compiled using a professional academic LaTeX template (such as the standard IEEE double-column format or the ICML single-column format). The narrative must explicitly address the complex business constraints and the mathematical reasoning behind the algorithmic choices.

* **Abstract (Page 1):** A concise, 200-word executive summary detailing the core objective, the specific algorithms utilized, the final number of variables selected, and the expected cross-validated profit score on the hold-out set.
* **1. Introduction & Problem Formulation (Page 1):**
    * Explanation of the marketing acquisition dataset and the fundamental challenge introduced by the imbalanced, cost-sensitive scoring metric.
    * Mathematical proof demonstrating why accuracy and ROC-AUC are fundamentally flawed metrics for this specific task, followed by the rigorous derivation of the 33.3% probability threshold constraint.
* **2. Feature Selection Strategy (Page 2):**
    * Theoretical justification for aggressive dimensionality reduction due to the severe 200-point penalty.
    * Table 1: The Reduction Funnel. A Markdown-style table illustrating the multi-stage reduction funnel (e.g., Starting Features: 500 $\rightarrow$ VIF Filter: 210 $\rightarrow$ LASSO: 35 $\rightarrow$ RFECV: 8 $\rightarrow$ Final Model: 5).
    * Figure 1: Permutation Importance. A horizontal bar chart for the final selected variables, quantifying their exact marginal utility in positive point values to justify their retention.
* **3. Modeling and Optimization Architecture (Page 3):**
    * Discussion of the chosen algorithms (e.g., LightGBM versus Logistic Regression) and their respective hyperparameter profiles, including how `scale_pos_weight` was utilized.
    * Explanation of the cross-validation strategy, specifically noting the use of Stratified K-Fold to maintain minority class ratios during training.
    * Figure 2: Calibration Curve. A reliability diagram proving that the model's outputs reflect true statistical probabilities before any hard thresholding was applied.
* **4. Threshold Analysis and Results (Page 4):**
    * Figure 3: The Profit Curve. A line graph plotting probability thresholds (x-axis) against the Expected Profit Score (y-axis). This visual unequivocally proves to the grading committee that the peak profit mathematically aligns with the theoretical threshold of 0.3333.
    * Table 2: Final Validation Metrics. A comparison table showing the baseline model, an unconstrained model (retaining 50 variables), and the final highly constrained model, highlighting the massive difference in the custom score.
* **5. Discussion and Conclusion (Page 5):**
    * Interpretation of the final selected variables, hypothesizing the potential real-world business logic behind the anonymized data.
    * Discussion of the limitations of the approach, such as potential concept drift in the test set, and implications for broader real-world marketing acquisition campaigns.

### 7-Minute Presentation Outline

A seven-minute presentation split evenly among three speakers allows for approximately 2 minutes and 20 seconds per speaker. The slide deck must adhere to high-density visual communication principles, inspired by PechaKucha formats, minimizing bulleted text in favor of critical charts, profit curves, and analytical narratives. The deck is strictly structured into 8 core slides to maintain pace.

| Slide | Content Focus | Speaker | Time Allocation | Key Visual |
| :--- | :--- | :--- | :--- | :--- |
| 1 | Title & Executive Summary | Student C | 0:30 | Project title, team names, and a massive call-out box showing the expected profit score and feature count. |
| 2 | The Business Problem & Math | Student C | 1:00 | Infographic demonstrating the scoring formula: 1 feature = -200 points. The math proving the 33.3% precision requirement. |
| 3 | Data Preprocessing & Baselines | Student A | 0:45 | Class imbalance pie chart and the initial correlation matrix heatmap showing extreme noise. |
| 4 | The Feature Reduction Funnel | Student A | 1:00 | A downward funnel diagram showing the transition from 500 raw variables down to the final subset via LASSO and RFE. |
| 5 | Machine Learning Architecture | Student B | 1:00 | Diagram of the XGBoost/LightGBM setup, detailing how class weights and hyperparameter grids were structured. |
| 6 | Probability Calibration | Student B | 1:00 | The Calibration Curve (Reliability Diagram) proving Isotonic/Platt scaling aligned predictions with statistical reality. |
| 7 | The Profit Curve & Thresholding | Student C | 1:00 | The climax of the presentation: The Profit Curve line graph showing how moving the boundary from 0.5 to 0.33 maximized the score. |
| 8 | Final Results & Q&A | All | 0:45 | Final statistics comparison table, concluding remarks, and an open floor for the grading committee's interrogation. |

## Custom Business Scorer Implementation in Python

To optimize the machine learning models utilizing algorithmic search mechanisms like `GridSearchCV`, `RandomizedSearchCV`, or Optuna, the exact business formula must be codified into a format that scikit-learn can interact with natively.

Because standard scikit-learn metrics wrapped by `make_scorer` strictly pass the true labels (`y_true`) and the predicted labels (`y_pred`), the metric function does not natively possess any awareness of the total number of features used by the underlying pipeline. This presents a critical engineering challenge, as the feature penalty is a massive component of the objective function. To bypass this architectural limitation, the scorer must be constructed as a direct callable that matches the `scorer(estimator, X, y_true)` signature.

By defining a function with this exact signature, scikit-learn will pass the fitted estimator and the validation feature matrix (`X`) directly into the scoring logic during every fold of the cross-validation process. This allows the function to dynamically extract the active feature count directly from the dimensions of `X` or from the estimator itself, compute the classification matrix, and calculate the exact mathematical objective.

The following Python snippet demonstrates the construction of this highly specialized custom scoring function.

```python
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

def business_profit_score(estimator, X, y_true):
    """
    Custom objective function to calculate the business profit score.
    This function adheres to the signature expected by scikit-learn for custom
    scorers that require access to the estimator and the input features.
    It extracts the feature count from the input data, predicts probabilities,
    applies the mathematically derived optimal threshold, and computes the cost matrix.

    Args:
        estimator: The fitted scikit-learn estimator or Pipeline.
        X: The test/validation feature matrix.
        y_true: The ground truth binary labels.

    Returns:
        float: The calculated profit score based on the specific project formula.
    """
    # 1. Generate calibrated probabilities from the estimator.
    # We use predict_proba to access the underlying probability of class 1.
    # (Assuming the estimator supports predict_proba, e.g., Logistic Regression, XGBoost).
    try:
        y_proba = estimator.predict_proba(X)[:, 1]
    except AttributeError:
        # Fallback to standard predict if predict_proba is completely unavailable,
        # though this disables dynamic thresholding.
        y_proba = estimator.predict(X)

    # 2. Apply the mathematically optimal decision threshold.
    # As derived in the strategic analysis, the break-even expected value requires p > 0.3333.
    # Therefore, we classify as positive only if the probability exceeds this strict threshold.
    optimal_threshold = 0.3333
    y_pred = (y_proba > optimal_threshold).astype(int)

    # 3. Extract classification metrics via the confusion matrix.
    # Utilizing the labels parameter ensures the matrix always returns a 2x2 array,
    # preventing errors if a specific validation fold lacks one of the classes.
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # 4. Determine the number of features utilized by the model.
    # If the pipeline includes feature selection steps (e.g., SelectFromModel or PCA),
    # the X matrix passed here represents the reduced feature space reaching the final estimator.
    # Thus, X.shape perfectly captures the active feature penalty.
    num_variables = X.shape[1]

    # 5. Calculate the specific financial rewards and penalties.
    reward_tp = tp * 10
    cost_fp = fp * 5
    cost_variables = num_variables * 200

    # 6. Compute the final objective score.
    final_score = reward_tp - cost_fp - cost_variables
    return float(final_score)

# The function 'business_profit_score' perfectly matches the required signature:
# scorer(estimator, X, y_true). Therefore, we do not need to wrap it using 'make_scorer'.
# It can be passed directly into the hyperparameter tuning utilities.
custom_scorer = business_profit_score

# Example usage integrating the custom scorer within a cross-validation grid search:
# grid = GridSearchCV(
#     estimator=modeling_pipeline,
#     param_grid=hyperparameter_grid,
#     scoring=custom_scorer,      # Injected directly into the validation process
#     cv=5,                       # 5-fold Stratified cross-validation
#     refit=True,                 # Refit the best model on the entire dataset
#     n_jobs=-1
# )

```

This structural architecture guarantees that every single iteration of hyperparameter tuning, feature selection, and algorithm evaluation is strictly and brutally governed by the final objective. By natively linking the feature count to the cross-validation score, the pipeline will mathematically punish complex models and automatically discard extraneous variables during the validation process. By isolating a highly predictive micro-cluster of variables, properly calibrating the machine learning probabilities, and applying the exact mathematical threshold logic, the final deployed model will extract the absolute maximum mathematical value from the marketing acquisition campaign.

## Cytowane prace

1. Cost-sensitive machine learning - Wikipedia, otwierano: maja 10, 2026, https://en.wikipedia.org/wiki/Cost-sensitive_machine_learning
2. Cost-Sensitive Learning: Beyond the Accuracy in Imbalanced Classification, otwierano: maja 10, 2026, https://www.blog.trainindata.com/cost-sensitive-learning-for-imbalanced-data/
3. Tune In: Decision Threshold Optimization with scikit-learn's TunedThresholdClassifierCV, otwierano: maja 10, 2026, https://medium.com/data-science/tune-in-decision-threshold-optimization-with-scikit-learns-tunedthresholdclassifiercv-7de558a2cf58
4. 2. Cost-sensitive learning - Reproducible Machine Learning for Credit Card Fraud detection - Practical handbook, otwierano: maja 10, 2026, https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_6_Imbalanced_Learning/CostSensitive.html


5. Post-tuning the decision threshold for cost-sensitive learning - Scikit-learn, otwierano: maja 10, 2026, https://scikit-learn.org/stable/auto_examples/model_selection/plot_cost_sensitive_learning.html
6. The foundations of cost-sensitive causal classification - arXiv, otwierano: maja 10, 2026, https://arxiv.org/html/2007.12582v6
7. Cost-Restricted Feature Selection for Data Acquisition ... - PubsOnLine, otwierano: maja 10, 2026, https://pubsonline.informs.org/doi/10.1287/mnsc.2022.4551
8. How to use classification threshold to balance precision and recall - Evidently AI, otwierano: maja 10, 2026, https://www.evidentlyai.com/classification-metrics/classification-threshold
9. Cost-sensitive learning strategies for high-dimensional and imbalanced data: a comparative study - PMC, otwierano: maja 10, 2026, https://pmc.ncbi.nlm.nih.gov/articles/PMC8725666/
10. [2508.03593] On the (In)Significance of Feature Selection in High-Dimensional Datasets, otwierano: maja 10, 2026, https://arxiv.org/abs/2508.03593
11. make_scorer — scikit-learn 1.8.0 documentation, otwierano: maja 10, 2026, https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html
12. sklearn.metrics.make_scorer - scikit-learn 1.2.2 documentation, otwierano: maja 10, 2026, https://scikit-learn.org/1.2/modules/generated/sklearn.metrics.make_scorer.html
13. Custom Loss vs Custom Scoring, otwierano: maja 10, 2026, https://kiwidamien.github.io/custom-loss-vs-custom-scoring.html
14. 3.4. Metrics and scoring: quantifying the quality of predictions - Scikit-learn, otwierano: maja 10, 2026, https://scikit-learn.org/stable/modules/model_evaluation.html
15. Scorer function: difference between make_scorer/score_func and - Stack Overflow, otwierano: maja 10, 2026, https://stackoverflow.com/questions/43523210/scorer-function-difference-between-make-scorer-score-func-and
16. cross_val_score — scikit-learn 1.8.0 documentation, otwierano: maja 10, 2026, https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html
17. How to implement a GridSearchCV custom scorer that is dependent on a training feature?, otwierano: maja 10, 2026, https://datascience.stackexchange.com/questions/82780/how-to-implement-a-gridsearchcv-custom-scorer-that-is-dependent-on-a-training-fe
18. Using Custom Metrics - Scikit-Learn Laboratory (SKLL), otwierano: maja 10, 2026, https://skll.readthedocs.io/en/latest/custom_metrics.html
19. How to create a customized scoring function in scikit-learn for scoring a set of instances based on their individual properties? - Stack Overflow, otwierano: maja 10, 2026, https://stackoverflow.com/questions/48468115/how-to-create-a-customized-scoring-function-in-scikit-learn-for-scoring-a-set-of
20. How many features are too many features??: r/datascience - Reddit, otwierano: maja 10, 2026, https://www.reddit.com/r/datascience/comments/17p5tct/how_many_features_are_too_many_features/
21. 1.4. Support Vector Machines - Scikit-learn, otwierano: maja 10, 2026, https://scikit-learn.org/stable/modules/svm.html
22. Accurate and fast feature selection workflow for high-dimensional omics data - PMC, otwierano: maja 10, 2026, https://pmc.ncbi.nlm.nih.gov/articles/PMC5738110/
23. Benchmarking feature selection and feature extraction methods to improve the performances of machine-learning algorithms for patient classification using metabolomics biomedical data - PMC, otwierano: maja 10, 2026, https://pmc.ncbi.nlm.nih.gov/articles/PMC10979063/
24. The Importance of Feature Selection in Machine Learning: A Case Study on Diamond Price Prediction - Himanshu Bhardwaj, otwierano: maja 10, 2026, https://hemi1984.medium.com/the-importance-of-feature-selection-in-machine-learning-a-case-study-on-diamond-price-prediction-1be0185553ed
25. comprehensive, step-by-step explanation of Feature Selection from basic concepts to supper-advanced methods(part 4) | by Adnan Mazraeh | Medium, otwierano: maja 10, 2026, https://medium.com/@adnan.mazraeh1993/comprehensive-step-by-step-explanation-of-feature-selection-from-basic-concepts-to-supper-advanced-87809ff995e8
26. Feature Selection in Machine Learning - Analytics Vidhya, otwierano: maja 10, 2026, https://www.analyticsvidhya.com/blog/2020/10/feature-selection-techniques-in-machine-learning/
27. 1.13. Feature selection - scikit-learn 1.8.0 documentation, otwierano: maja 10, 2026, https://scikit-learn.org/stable/modules/feature_selection.html
28. RFECV to provide the average ranking_ and support Issue #17782 - GitHub, otwierano: maja 10, 2026, https://github.com/scikit-learn/scikit-learn/issues/17782
29. Procedure for selecting optimal number of features with Python's Scikit-Learn, otwierano: maja 10, 2026, https://datascience.stackexchange.com/questions/57816/procedure-for-selecting-optimal-number-of-features-with-pythons-scikit-learn
30. Machine Learning Methods to Perform Pricing Optimization: A Comparison with Standard Generalized Linear Models, otwierano: maja 10, 2026, https://www.casact.org/sites/default/files/2021-07/Machine-Learning-Methods-Spedicato-Dutang-Petrini.pdf
31. Data-Science Design Patterns: Cost-Sensitive Learning - Alteryx, otwierano: maja 10, 2026, https://community.alteryx.com/discussion/44941/data-science-design-patterns-cost-sensitive-learning
32. Classification threshold cost optimisation: r/datascience - Reddit, otwierano: maja 10, 2026, https://www.reddit.com/r/datascience/comments/1h8dIz9/classification_threshold_cost_optimisation/
33. Post-tuning the decision threshold for cost-sensitive learning - Scikit-learn, otwierano: maja 10, 2026, https://scikit-learn.org/1.5/auto_examples/model_selection/plot_cost_sensitive_learning.html
34. Lecture 8: ML Teams and Project Management - The Full Stack, otwierano: maja 10, 2026, https://fullstackdeeplearning.com/course/2022/lecture-8-teams-and-pm/
35. Machine Learning Team Projects: a Survival Guide - Daniele Grattarola, otwierano: maja 10, 2026, https://danielegrattarola.github.io/posts/2018-03-20/ml-team-projects.html
36. Building an AI-Powered Task Management System: A Cross-India ML Collaboration | by DINESH S J | Medium, otwierano: maja 10, 2026, https://medium.com/@dineshsj/building-an-ai-powered-task-management-system-a-cross-india-ml-collaboration-2f61622432c6
37. Final Project Grading Rubric | Deep Learning | Electrical Engineering and Computer Science | MIT OpenCourseWare, otwierano: maja 10, 2026, https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/pages/final-project-grading-rubric/
38. Automated assignment grading with large language models: insights from a bioinformatics course - PMC, otwierano: maja 10, 2026, https://pmc.ncbi.nlm.nih.gov/articles/PMC12261420/
39. LaTeX - Machine Learning, otwierano: maja 10, 2026, https://ml1.qiguo.org/resources/latex.html
40. stat479-machine-learning-fs18/report-template/report.tex at master - GitHub, otwierano: maja 10, 2026, https://github.com/rasbt/stat479-machine-learning-fs18/blob/master/report-template/report.tex
41. Final Project 1 Introduction 2 Background & Task, otwierano: maja 10, 2026, https://www.cs.toronto.edu/~michael/teaching/csc311_w23/homework/project.pdf
42. & ATTACHMENTS - John Jay College of Criminal Justice - CUNY, otwierano: maja 10, 2026, https://www.jjay.cuny.edu/sites/default/files/2025-10/CC%20Agenda_November%2010%2C%202025_for%20web.pdf
43. Computational Science Tools, otwierano: maja 10, 2026, https://science.gmu.edu/media/csi-500-scott-0
44. Technical Report Template & Guidance, otwierano: maja 10, 2026, https://risk.fbv.kit.edu/rd_download/Report_Format%20(1).pdf
45. The Term Project - Computer and Information Science, otwierano: maja 10, 2026, https://www.cis.upenn.edu/~myatskar/teaching/cis5300_fa23//term-project.html
46. Use This Japanese Technique to Improve Your Presentations | by Dipesh Jain - Medium, otwierano: maja 10, 2026, https://dipesh17.medium.com/use-this-japanese-technique-to-improve-your-presentations-33fd4ed7947c
47. Final Project | Sundong Kim, otwierano: maja 10, 2026, https://sundong.kim/courses/dataeng24sp/project/
48. 2025-2026 Competitive Events Guidelines - Oregon FBLA, otwierano: maja 10, 2026, https://oregonfbla.org/25-26-data-modeling-with-ai-machine-learning/
49. Program Handbook - Future City® Competition, otwierano: maja 10, 2026, https://futurecity.org/wp-content/uploads/2022/08/2022_Future_City_Handbook.pdf
