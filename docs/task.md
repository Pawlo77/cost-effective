# Project 2: Cost-Sensitive Predictive Modeling

## 1. Goal
The objective is to develop a high-performance, cost-effective predictive model to identify customers most likely to convert on a marketing offer. The challenge requires striking a delicate balance: maximizing revenue while strictly managing the costs associated with data acquisition and negative customer impact.

In modern marketing, "more" is not always "better." Sending unsolicited offers to uninterested clients creates "Marketing Fatigue." This annoyance degrades brand value, reduces customer trust, and can lead to long-term churn.

Your mandate is to design a targeted campaign that treats customer attention as a finite, valuable resource. You must demonstrate that your model can act as a precise filter, identifying only those customers who provide the highest net value to the firm. So, model should meet two operational constraints:

* **Data Investment:** Every variable used to inform your model carries a specific cost for acquisition and processing. You must demonstrate that every data point included provides enough value to justify its cost.
* **Targeting Efficiency:** You are authorized to contact a maximum of 1,000 customers. However, to maximize profitability and protect the brand, you should only contact those individuals where the model shows a high probability of conversion.

## 2. Data
* `x_train.txt`: Variable matrix for 5000 training clients (500 anonymized variables).
* `y_train.txt`: Labels ($1=$ accepted, $0=$ did not accept).
* `x_test.txt`: Variable matrix for 5,000 clients for prediction.

## 3. Task
Build a model to identify up to 1000 customers in the test set who will benefit from the offer. You must also indicate which variables were used. You may choose to target max. 1000 customers to maximize profit.

## 4. Scoring
The effectiveness is assessed based on the following financial impact:
* + EUR 10 for each True Positive (TP, selected customers who took the offer).
* - EUR 5 for each False Positive (FP, selected customers who did not take the offer).
* - EUR 200 for each variable used in the model (NoVariables indicates the number of used variables).

**Formula:**
`Score = (TP * 10) - (FP * 5) - (NoVariables * 200)`

**Examples:**
* **Team A:** Targets 1,000 people (800 correct), uses 20 variables.
  Score: (800 * 10) - (200 * 5) - (20 * 200) = EUR 3000
* **Team B:** Targets 500 people (450 correct), uses 5 variables.
  Score: (450 * 10) - (50 * 5) - (5 * 200) = EUR 3250

## 5. Project evaluation (50 points)

**Score - 25 points**
For details how models are evaluated, please see previous section. Final score will be assigned according to the leaderboard of model performance attained by all teams.

**Report - 15 points**
The investigated strategies and the finally selected model should be described in the report. The report should include key information to enable reproduction of the solution and, in addition, the results of the experiments arguing the design decisions made.
* Maximum number of pages of the report: 5 pages
* The report should be prepared in Latex

**Presentation - 10 points**
Presentation will be given during project meeting in front of the whole group, so you should prepare slides. Presentation should take max 7 minutes. Attendance during the presentation is obligatory to get points for the presentation.

## 6. Additional remarks:
1. You can choose any programming language (Python/R are preferred), as long as the resulting files are in the correct format.
2. Projects are prepared in groups of 3 students.

## 7. How to submit a solution?
Your solution should be contained in two files:
* File `STUDENT1ID_STUDENT2ID_STUDENT3ID_obs.txt` should contain up tp 1000 indexes of customers from testing data to whom you want to send the offer. In the case of sending higher number of indexes, only first 1000 ids will be assessed.
* File `STUDENT1ID_STUDENT2ID_STUDENT3ID_vars.txt` should contain the indexes of variables used by the proposed model.

`STUDENTXID` is a student id of the X student from the team.
Please see example files: `123456_98765_98764_obs.txt` and `123456_98765_98764_vars.txt`. The submitted files must be in the same format.

Please save all results to the ZIP file, named `STUDENT1ID_STUDENT2ID_STUDENT3ID.zip`. The archive should contain the following files: `*_obs.txt`, `*_vars.txt`, `report.pdf`, `presentation.pdf` (ppt, pptx, etc.) and folder named `code` with source codes.

Please upload your solution using the assignment available in the MS Teams channel.

**Deadlines**
* Solutions should be submitted until 08.06.2026 23:59
* Final presentations: 10.06.2026

## 8. Organization of work
Project consultations are held every two weeks. Since project classes are not mandatory, the team reports attendance at consultations in the Consultations file, and in addition, we ask for a message to the corresponding instructor. Consultation attendance should be reported by the preceding Tuesday by 6 pm.
