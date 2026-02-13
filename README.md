# Brain_Tumor_Segmentation_

This project performs brain tumor segmentation on MRI images 
using Otsu and Sauvola thresholding methods. The performance 
is evaluated using Dice and Jaccard scores.


Dataset taken from https://www.kaggle.com/datasets/atikaakter11/brain-tumor-segmentation-dataset




## Installations

1. Create a virtual environment:
   python -m venv venv

2. Activate it:
   Windows:
   venv\Scripts\activate

   Mac/Linux:
   source venv/bin/activate

3. Install dependencies:
   pip install -r requirements.txt



## How to Run

1. Run for dataset download: 
   python data.py

2. Run segmentation and evaluation:
   python run.py