The repository is the official implementation for the **IJCAI'25** AI for Social Good Track paper: **Detecting Illicit Massage Businesses by Leveraging Graph Machine Learning**

# IMB-RGCN Classification Pipeline

## 📁 Dataset Download and Setup

1. **Obtain the Synthetic Dataset**:
   - Request or generate the cleaned dataset file `data.csv` containing:
     - Business metadata (e.g., Yelp features)
     - Reviewer information
     - Review embeddings (from Doc2Vec or transformer-based models)
     - Census and geospatial features

2. **Save the Dataset to Google Drive**:
   - Create the following folder structure in your Google Drive:
     ```
     /MyDrive/RGCN_IMB_Detection/
     ```
   - Place your `data.csv` file inside this folder:
     ```
     /MyDrive/RGCN_IMB_Detection/data.csv
     ```
   - This path must match the one used in `main.py`:
     ```python
     data_path = '/content/drive/MyDrive/RGCN_IMB_Detection/data.csv'
     ```

## 🚀 Running the Project in Google Colab

You can execute this project end-to-end in Google Colab. Here's how:

## 1. Clone the repository
```python
!git clone https://github.com/Vasuki-Garg/rgcn-imb-detection.git
%cd rgcn-imb-detection

## 2. Install dependencies
```python
!pip install torch==2.4.0
!pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/repo.html
!pip install -r requirements.txt

# 3. Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')

# 4. Ensure your dataset is saved at:
```python
# /content/drive/MyDrive/RGCN_IMB_Detection/data.csv

# 5. Run the main pipeline
```python
!python main.py

## About the Authors
**Vasuki Garg** is a PhD student in Industrial and Systems Engineering at NC State University. He holds an MS from Politecnico di Milano, Italy and a BEng from the University of Southampton, UK. His research applies machine learning and optimization to social impact problems, including human trafficking detection and decision-focused modeling. He specializes in graph ML, NLP, and data-driven analytics.

**Osman Özaltın** is an Associate Professor in Industrial and Systems Engineering at NC State and a member of the Personalized Medicine Faculty Cluster. He holds MS and PhD degrees from the University of Pittsburgh. His research focuses on mathematical programming for personalized medical decisions and illicit supply chains, using methods like integer and bilevel programming.

**Maria E. Mayorga** is a Professor at NC State specializing in personalized medicine and health systems. She earned her MS and PhD from UC Berkeley. Her work applies mathematical and stochastic modeling to health care operations, emergency response, and humanitarian logistics. She was named an IISE Fellow in 2022.

**Sherrie Caltagirone** is the Founder and Executive Director of the Global Emancipation Network (GEN), a nonprofit leveraging data analytics to combat human trafficking. Previously, she was a Policy Advisor at Orphan Secure and began her anti-trafficking work with the Protection Project at Johns Hopkins University.

## Usage
This repository includes a dummy dataset for demonstration purposes. The original datasets used in the study contain sensitive information and are available upon request from the Global Emancipation Network (GEN).

## Citation

If you use this code or find our work useful, please cite our paper:

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
}

