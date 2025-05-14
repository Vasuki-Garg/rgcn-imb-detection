# RGCN_IMB_Detection Pipeline

This repository contains the official implementation for our **IJCAI 2025** paper in the *AI for Social Good* track:

**"Detecting Illicit Massage Businesses by Leveraging Graph Machine Learning"**

---

## Dataset Download and Setup
**[Click here to download the synthetic dataset from Google Drive](https://drive.google.com/file/d/1bJYk59VQZ-2zDdBYdY5jjmTtVIBeZjAK/view?usp=sharing)**

## Dataset Description:
This dataset is constructed by integrating multi-source information to build an information-rich heterogeneous graph for detecting illicit massage businesses (IMBs). It combines business-level metadata, review content, reviewer information, geospatial features, and demographic indicators to support relational learning and classification.
The dataset consists:

### Business Features
Derived from Yelp and RubMaps business metadata, GIS data, the U.S. Census Bureau, and the National Land Cover Database (NLCD). These are primarily binary features, created through quantile-based thresholding or one-hot encoding:
```
yelp_close9, yelp_close10, yelp_close11, 
yelp_avg_rating_moreThan4, yelp_avg_rating_lessThan2, 
yelp_reviewRating_min_is5, yelp_reviewRating_max_is1, 
yelp_massageCat, yelp_spaCat, yelp_phone_advertisement, 
yelp_business_name_combine, yelp_category_reflexology, 
owner_listed_worker_out_of_state, min_dist_base_high, min_dist_base_low, 
min_dist_police_low, census_pct_nonwhite_high, census_avg_household_size_high, 
census_pct_20_to_29_low, census_pct_housing_vacant_low, 
census_pct_households_with_children_low, census_pct_over25_with_bachelors_low, 
census_pct_manufacturing_industry_low, landcover_type_developed_high_intensity
```
### Review Features
Extracted from Yelp review texts using NLP techniques and pre-trained models. These are numerical features, including dense embeddings and sentiment scores:
```
review_vector, roberta_neu, roberta_pos, roberta_neg, 
reviewRating, lexicon_score, lexicon_prediction
```
### Reviewer Features
Based on Yelp user metadata and enriched using external tools (e.g., gender prediction libraries):
```
authorName, authorGender
```
### Label
A binary classification label indicating whether a business is illicit (1) or non-illicit (0). The label is derived from RubMaps review activity and business license status.

## Setup

1. **Dataset**:
   - Use the provided synthetic `data.csv` file for demonstration, or request access to the original (restricted) data through the Global Emancipation Network (GEN).

2. **Save the Dataset to Google Drive**:
   - Create the following folder structure in your Google Drive:
     ```
     /MyDrive/RGCN_IMB_Detection/
     ```
   - Place your `data.csv` file in that folder:
     ```
     /MyDrive/RGCN_IMB_Detection/data.csv
     ```
   - This path is referenced in the code as:
     ```python
     data_path = '/content/drive/MyDrive/RGCN_IMB_Detection/data.csv'
     ```

---

## Running the Project in Google Colab

Run the complete pipeline on Colab by following these steps:

### Step 1: Clone the Repository
```python
!git clone https://github.com/Vasuki-Garg/rgcn-imb-detection.git
%cd rgcn-imb-detection
```

### Step 2: Install Dependencies
```python
!pip uninstall torch -y
!pip install torch==2.4.0
!pip uninstall dgl -y
!pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/dgl-2.4.0-cp311-cp311-manylinux1_x86_64.whl
!pip install -r requirements.txt
```

### Step 3: Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```
Ensure the dataset is stored at:
```
/content/drive/MyDrive/RGCN_IMB_Detection/data.csv
```

### Step 4: Run the Main Script
```python
!python main.py
```

This runs:
- Data preprocessing and graph construction
- Feature engineering for business, review, and reviewer nodes
- RGCN model training with early stopping
- Evaluation and embedding extraction
- t-SNE and graph neighborhood visualizations
- Saving model checkpoints, results, and plots to Google Drive

---

## Output Files (in Google Drive)
- `final_trained_model.pt`: Final model weights
- `val_embeddings_<model>.csv`: Validation node embeddings with labels
- `output.csv`: Aggregated metrics from each run
- `graph_plot.png`, `tsne_plot.png`: Optional visualizations

---

## About the Authors
**Vasuki Garg** is a PhD student in Industrial and Systems Engineering at NC State University. He holds an MS from Politecnico di Milano, Italy and a BEng from the University of Southampton, UK. His research applies machine learning and optimization to social impact problems, including human trafficking detection and decision-focused modeling. He specializes in graph ML, NLP, and data-driven analytics.

**Osman Özaltın** is an Associate Professor in Industrial and Systems Engineering at NC State and a member of the Personalized Medicine Faculty Cluster. He holds MS and PhD degrees from the University of Pittsburgh. His research focuses on mathematical programming for personalized medical decisions and illicit supply chains, using methods like integer and bilevel programming.

**Maria E. Mayorga** is a Professor at NC State specializing in personalized medicine and health systems. She earned her MS and PhD from UC Berkeley. Her work applies mathematical and stochastic modeling to health care operations, emergency response, and humanitarian logistics. She was named an IISE Fellow in 2022.

**Sherrie Bosisto** is the Founder and Executive Director of the Global Emancipation Network (GEN), a nonprofit leveraging data analytics to combat human trafficking. Previously, she was a Policy Advisor at Orphan Secure and began her anti-trafficking work with the Protection Project at Johns Hopkins University.

## Usage
This repository includes a dummy dataset for demonstration purposes. The original datasets used in the study contain sensitive information and are available upon request from the Global Emancipation Network (GEN).

## Citation

If you use the dataset, code or find our work useful, please cite our paper:

```bibtex
@inproceedings{garg2025graph,
  title     = {Detecting Illicit Massage Businesses by Leveraging Graph Machine Learning},
  author    = {Garg, Vasuki and Özaltın, Y. Osman and Mayorga, E. Maria and Caltigirone, Sherrie},
  booktitle = {Proceedings of the Thirty-Fourth International Joint Conference on
               Artificial Intelligence, {IJCAI-25}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {XYZ},
  pages     = {5689-5697},
  month     = {8},
  year      = {2025},
  note      = {AI and Social Good Track},
  url       = {https://github.com/Vasuki-Garg/rgcn-imb-detection}
```


