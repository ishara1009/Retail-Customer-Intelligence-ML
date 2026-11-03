
# Retail Customer Intelligence — සිංහල Walkthrough

මෙය fictional/synthetic customer data භාවිත කරන learning project එකක්. Real
customer කෙනෙකු ගැන production decision ගන්න මෙය සුදුසු බව claim කරන්නේ නැහැ.

## Data flow එක

`data_loader.load_monthly_data()` CSV එක load කර dates parse කර required columns,
unique `customer_id + snapshot_date`, non-negative values සහ targets දෙකේ
consistency validate කරනවා. Error එකක් තිබුණොත් row එකක් හොරෙන් drop නොකර clear
exception එකක් දෙනවා.

එක් row එකක් customer කෙනෙකුගේ එක් monthly snapshot එකක්. 2025-04-01 row එකේ
historical information April 1 ට කලින් පමණයි. 90-day window එක `[2025-01-01,
2025-04-01)`; target window එක `[2025-04-01, 2025-05-01)`. එහෙම නම් “next
month” කියන්නේ historical observation window එකට වහාම පසුව එන target month එක.

`chronological_split()` මගින් train මාස 8, validation මාස 2, test මාස 2 ලෙස
වෙන් කරනවා. Customer කෙනෙක් මාස කිහිපයක නැවත පෙනෙන නිසා මේ test එක **existing
customers ගේ future outcomes** ගැනයි; totally unseen customers ගැන නොවේ.

## Pipeline සහ leakage

`FEATURE_COLUMNS` කියන allowlist එක පමණක් `X` ලෙස යොදාගන්නවා. `customer_id`,
`snapshot_date`, `next_month_spend_lkr`, `will_purchase_next_month` කිසිම model
input එකකට යන්නේ නැහැ. Future target එක feature එකකට ගියොත් ඒක **data leakage**.

- `fit`: train data වලින් median, scale, coefficients හෝ tree rules ඉගෙනගන්නවා.
- `transform`: ඉගෙනගත් median/scale භාවිත කර data වෙනස් කරනවා.
- `predict`: final answer/label එක ලබාදෙනවා.
- `predict_proba`: class 1 සඳහා probability එක ලබාදෙනවා.
- `Pipeline`: preprocessing හා model එක එක chain එකක තබා train-only fitting
  enforce කරනවා.

Regression එක LKR spend අගයක් predict කරනවා. MAE සාමාන්‍ය absolute error එක;
RMSE ලොකු වැරදිවලට වැඩි penalty එකක්; R² mean baseline එකට සාපේක්ෂ score එක.
Classification එක buy/not-buy target එක predict කර accuracy, precision, recall,
F1, ROC-AUC සහ confusion matrix බලනවා. False positive එකක් නිසා unnecessary
promotion cost එකක් ඇතිවිය හැකි අතර false negative එකක් නිසා potential buyer
කෙනෙක් miss විය හැකියි.

Validation result එකෙන් model/hyperparameter තෝරා train+validation මත නැවත fit
කර test එක එක් වරක් පමණක් බලනවා. Test result බලලා model එක වෙනස් කරන්නේ නැහැ.

## Clustering සහ experiments

KMeans 2025-11-01 snapshot එකේ recency, order count, spend standardize කර geometric
groups හදනවා. Cluster 0/1 කියන අංකවල inherent business meaning එකක් නැහැ.
Silhouette score එක separation පෙන්වනවා; business value එකක් prove කරන්නේ නැහැ.

Overfitting experiment එක retail regression trees භාවිත කරනවා. Bias/variance
experiment එක retail CSV නොව, known `sin(x)` function එකෙන් independent noisy
datasets 30ක් සාදන වෙනම simulation එකක්. ඒ දෙක mix නොකළ යුතුයි.

Feature selection experiment එක Logistic Regression පෙර `SelectKBest(k=4)` fit
කර all-seven-features result එක සමඟ validation මත compare කරනවා.

## Q-learning

GridWorld එක retail model එකක් නොවේ. State එක cell එක; action එක up/down/left/right;
ordinary reward `-1`; goal reward `+10`. Update එක:

`new Q = old Q + learning_rate × (target - old Q)`

Goal එක terminal නම් `target = reward`; future Q bootstrap කරන්නේ නැහැ. Steps 50ට
නවත්වන truncation එක goal reached termination එකට වෙනස්. Training අවසානයේ greedy
policy එක update/exploration නැතුව evaluate කර uniform-random baseline එක සමඟ
compare කරනවා.

## කෙටි Interview Q&A

1. **ML කියන්නේ?** Examples වලින් patterns/parameters ඉගෙන prediction හෝ decision
   support කරන AI subset එකක්.
2. **AI, ML, DL සම්බන්ධය?** DL ⊂ ML ⊂ AI. මේ tabular data එකට DL unnecessary.
3. **Supervised vs unsupervised?** Regression/classification targets තියෙනවා;
   clustering target නැහැ.
4. **RL වෙනස?** Agent actions, rewards සහ experience මගින් policy එකක් ඉගෙනගන්නවා.
5. **Regression vs classification?** Continuous spend අගයක් vs binary purchase class.
6. **Train/validation/test ඇයි?** Learn, choose, unbiased final estimate කියන roles
   වෙන් කිරීමට.
7. **Overfit/underfit?** Train details වැඩියෙන් learn කිරීම vs useful patternවත්
   capture නොකිරීම.
8. **Bias/variance?** Average systematic error vs datasets මාරුවෙද්දී prediction
   variability.
9. **Feature selection ඇයි?** Noise/redundancy අඩු කිරීමට; performance improve වන
   බව guarantee නැහැ.
10. **Metrics තෝරන්නේ කොහොමද?** Business error type සහ target type අනුව; spend සඳහා
    MAE/RMSE, buyers සඳහා precision/recall/F1 වගේ metrics.
11. **Parameter vs hyperparameter?** `fit` එක coefficient/tree split ඉගෙනගනී;
    validation එක tree depth වගේ hyperparameter තෝරයි.
12. **Leakage එකක් කියන්නේ?** Prediction time එකේ නොදන්න future/target information
    model training input එකට ඇතුළත් වීම.

Beginner reading order: `config.py` → `data_loader.py` → `preprocessing.py` →
`regression.py` → `classification.py` → `clustering.py` → experiments → RL.
