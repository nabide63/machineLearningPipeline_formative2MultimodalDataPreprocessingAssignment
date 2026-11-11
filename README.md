# Formative 2: Multimodal Data Preprocessing

This is our group repo for the **Machine Learning Pipeline – Formative 2** assignment. The idea is to build a user identity and product recommendation system that uses three kinds of data: tabular customer data, face images, and voice samples.

The planned system flow looks like this:

```
Face recognised ─► Product model predicts ─► Voice approves ─► Prediction shown
      │                                             │
      └── not recognised ─► ACCESS DENIED           └── voice rejected ─► ACCESS DENIED
```

So far **Task 1** (data merge + product recommendation model) and **Task 2** (image data collection and processing) are done. The rest of the tasks are still in progress and will be added here when they're finished.

---

## Table of Contents

- [Team](#team)
- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [Task 1 – Data Merge and Product Recommendation Model](#task-1--data-merge-and-product-recommendation-model)
- [Task 2 – Image Data Collection and Processing](#task-2--image-data-collection-and-processing)
- [Upcoming Tasks](#upcoming-tasks)

---

## Team

| Task | Description | Done by |
|---|---|---|
| Task 1 | Data merge, feature engineering, product recommendation model | Yusuf Nabide |
| Task 2 | Image collection, augmentation, feature extraction | Esther Digne Irakoze |

---

## Repository Structure

```
.
├── data/
│   ├── customer_social_profiles.csv          # raw social media profiles (Task 1 input)
│   ├── customer_transactions.csv             # raw transactions (Task 1 input)
│   ├── merged_customer_transactions_social.csv  # merged + engineered dataset (Task 1 output)
│   └── customer_feature_store.csv            # latest features per customer, used by the CLI
├── Images/
│   ├── Esther/   Laura/   Rwema/   Yusuf/    # neutral, smiling, surprised photos per member
├── models/
│   └── product_recommender.joblib            # trained product model (Task 1)
├── scripts/
│   └── product_model.py                      # command-line inference for the product model
├── image_features.csv                        # extracted image features (Task 2 output)
├── task1_data_merge_and_predictiveModel.ipynb
├── task2_image_processing.ipynb
├── requirements.txt
└── README.md
```

---

## Setup

1. Clone the repo:
   ```bash
   git clone https://github.com/nabide63/machineLearningPipeline_formative2MultimodalDataPreprocessingAssignment.git
   cd machineLearningPipeline_formative2MultimodalDataPreprocessingAssignment
   ```
2. (Optional) Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   pip install pillow            # needed for the Task 2 notebook
   ```
4. Open the notebooks with Jupyter:
   ```bash
   jupyter notebook
   ```

---

## Task 1 – Data Merge and Product Recommendation Model

**Author:** Yusuf Nabide
**Notebook:** [`task1_data_merge_and_predictiveModel.ipynb`](task1_data_merge_and_predictiveModel.ipynb)

### Goal

Merge the `customer_social_profiles` and `customer_transactions` datasets, engineer useful features, and train a **Product Recommendation Model** that predicts which product category a customer is likely to buy when they visit our social media pages.

### Data

| File | Rows | Notes |
|---|---|---|
| `customer_social_profiles.csv` | 155 | IDs like `A178`, platform, engagement score, purchase interest score, review sentiment |
| `customer_transactions.csv` | 150 | IDs like `178`, purchase amount, date, product category (the target), customer rating |

### What I did

1. **Loading and inspection:** I converted all column names to `snake_case` and checked samples, summary statistics, missing values and duplicates for both files.
2. **EDA:** I made distribution plots, boxplots with an IQR outlier check, the target class balance, platform/sentiment counts, a correlation heatmap and a category transition heatmap. All numeric features are close to uniform with no outliers, and the five product categories are roughly balanced (17%–23%).
3. **Cleaning:**
   - Removed 5 exact duplicate rows from the social profiles.
   - Converted `purchase_date` to a datetime and checked that every numeric column is within its valid range.
   - Left the 10 missing `customer_rating` values as NaN on purpose. They get imputed *inside* the model pipeline so no test data leaks into training.
4. **Merge key:** the two files name customers differently (`A178` vs `178`), so I stripped the `A` with a strict regex to build a shared `customer_id`.
5. **Aggregation:** many customers have more than one social profile row (one per platform), so I collapsed them into one row per customer first (mean/max engagement and interest, mean sentiment score, share of rows on each platform). This avoids a many-to-many join.
6. **Merge:** a left join from transactions to profiles with `validate="many_to_one"`. All 150 transactions are kept, 117 (78%) of them have a social profile. I also spot-checked one customer by recomputing their values by hand.
7. **Feature engineering:** I only used features that would exist *at the moment of the visit*, so nothing from the current purchase leaks in. These are the number of past purchases, average past spend and rating, last category bought, days since the last purchase, and visit month/weekday (built with `shift()` so each row only sees earlier rows).
8. **Modelling:**
   - Stratified 80/20 train/test split, with repeated 5-fold CV on the training set for model selection.
   - Compared a dummy baseline, Logistic Regression, Random Forest and Gradient Boosting.
   - Ran an ablation to check that purchase history features actually help and that the "leaky" features aren't needed.
   - Tuned with a small grid scored by log loss. **Random Forest** was selected.
9. **Deployment:** I refit the final model on all profile-matched rows and saved it to `models/product_recommender.joblib`, together with a per-customer feature store (`data/customer_feature_store.csv`) so the CLI can predict from just a customer ID.

### Results (held-out test set, 24 rows)

| Metric | Random Forest | Baseline |
|---|---|---|
| Accuracy | **0.33** | 0.25 |
| Macro-F1 | **0.28** | 0.08 |
| Top-3 accuracy | **0.75** | 0.67 |
| Log loss | 1.596 | 1.593 |

The model beats the baseline on accuracy, F1 and top-3 accuracy, but to be honest the data doesn't separate the categories very well, and the predicted probabilities are close to the class shares. That's why I present the output as **top-3 recommendations**: the right product is in the three suggestions 75% of the time.

### Running the product model from the command line

```bash
python scripts/product_model.py --customer-id 125
python scripts/product_model.py --customer-id 125 --top-k 5
```

Or from another script:

```python
from scripts.product_model import recommend_products
result = recommend_products(125)
```

- A customer with a social profile gets predictions from the Random Forest.
- A known customer **without** a social profile gets a popularity-based fallback.
- An unknown customer ID returns **ACCESS DENIED**.

---

## Task 2 – Image Data Collection and Processing

**Author:** Esther Digne Irakoze
**Notebook:** [`task2_image_processing.ipynb`](task2_image_processing.ipynb)

### Goal

Collect facial images of every group member, augment them, and extract features that the facial recognition model can use later.

### What was done

1. **Image collection:** each of the 4 members (Esther, Laura, Rwema, Yusuf) took 3 photos with different expressions: **neutral, smiling and surprised**. That gives 12 original images, all JPEG, stored in `Images/<member>/`.
2. **Loading and display:** the images were scanned into a table (member, filename, size, format) and shown in a grid to check them. There were no missing values, and the sizes ranged from 540–1123 px wide and 960–1280 px tall.
3. **Augmentation:** each original image got 3 augmentations:
   - horizontal flip
   - rotation by 10°
   - grayscale conversion

   That's 36 augmented images, so **48 images in total**.
4. **Feature extraction:** each image was resized to 128×128 and **normalised colour histograms** were computed with 8 bins for each RGB channel, which gives 24 features per image.
5. **Saving the features:** the features were cleaned (only the original + 3 augmentation types kept, duplicates removed) and saved to `image_features.csv`.

### Output: `image_features.csv`

| Rows | Columns | Details |
|---|---|---|
| 48 | 27 | `member`, `filename`, `image_type` + 24 histogram features (`R_bin_0` … `B_bin_7`) |

Images by type: 12 original, 12 flipped, 12 rotated, 12 grayscale.

### Running the notebook

The notebook was written in **Google Colab** and reads images from Google Drive (`/content/drive/MyDrive/Formative2_Facial_Recognition/images`). To run it locally, change `project_path` at the top of the notebook to point at this repo, and the images folder to `Images/`.

---

## Upcoming Tasks

- Audio data collection and processing (voice samples + audio features)
- Facial recognition and voiceprint verification models
- Full command-line system that connects face → product → voice

These will be added to this README once they're done.
