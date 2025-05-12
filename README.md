# IMB-RGCN Classification Pipeline

This repository contains the official implementation for our **IJCAI 2025** paper in the *AI for Social Good* track:

**"Detecting Illicit Massage Businesses by Leveraging Graph Machine Learning"**

---

## 📁 Dataset Download and Setup

1. **Obtain the Dataset**:
   - Use the provided synthetic `data.csv` file for demonstration, or request access to the original (restricted) data through the Global Emancipation Network (GEN).
   - The dataset includes:
     - Business metadata (e.g., Yelp-based features)
     - Reviewer details
     - Review embeddings (e.g., Doc2Vec, BERT)
     - Census and geospatial information

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

## 🚀 Running the Project in Google Colab

Run the complete pipeline on Colab by following these steps:

### ✅ Step 1: Clone the Repository
```python
!git clone https://github.com/Vasuki-Garg/rgcn-imb-detection.git
%cd rgcn-imb-detection
```

### ✅ Step 2: Install Dependencies
```python
!pip install torch==2.4.0
!pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/repo.html
!pip install -r requirements.txt
```

### ✅ Step 3: Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```
Ensure the dataset is stored at:
```
/content/drive/MyDrive/RGCN_IMB_Detection/data.csv
```

### ✅ Step 4: Run the Main Script
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

## 💾 Output Files (in Google Drive)
- `final_trained_model.pt`: Final model weights
- `val_embeddings_<model>.csv`: Validation node embeddings with labels
- `output.csv`: Aggregated metrics from each run
- `graph_plot.png`, `tsne_plot.png`: Optional visualizations

---

## About the Authors
**Vasuki Garg** is a PhD student in Industrial and Systems Engineering at NC State University. He holds an MS from Politecnico di Milano, Italy and a BEng from the University of Southampton, UK. His research applies machine learning and optimization to social impact problems, including human trafficking detection and decision-focused modeling. He specializes in graph ML, NLP, and data-driven analytics.

**Osman Özaltın** is an Associate Professor in Industrial and Systems Engineering at NC State and a member of the Personalized Medicine Faculty Cluster. He holds MS and PhD degrees from the University of Pittsburgh. His research focuses on mathematical programming for personalized medical decisions and illicit supply chains, using methods like integer and bilevel programming.

**Maria E. Mayorga** is a Professor at NC State specializing in personalized medicine and health systems. She earned her MS and PhD from UC Berkeley. Her work applies mathematical and stochastic modeling to health care operations, emergency response, and humanitarian logistics. She was named an IISE Fellow in 2022.

**Sherrie Caltagirone** is the Founder and Executive Director of the Global Emancipation Network (GEN), a nonprofit leveraging data analytics to combat human trafficking. Previously, she was a Policy Advisor at Orphan Secure and began her anti-trafficking work with the Protection Project at Johns Hopkins University.

## Usage
This repository includes a dummy dataset for demonstration purposes. The original datasets used in the study contain sensitive information and are available upon request from the Global Emancipation Network (GEN).

## Citation

If you use this code or find our work useful, please cite our paper:

bibtex
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
}


