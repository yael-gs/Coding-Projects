# Deep Learning Projects

This repository contains various deep learning projects implemented in Jupyter notebooks, accompanied by relevant visualizations. For each project some selected plots are shown.

---

## Projects

1. **[Project 1: MLP for Classification](1-MLP_byhand.ipynb)**
   - **Description**: A hand-coded Multi-Layer Perceptron (MLP) for basic classification with one hidden layer.
  
- **Plots**:
  - **Input Data**:  
    <img src="plots/1-input_data.png" alt="Input Data" width="500" />  

  - **Training Metrics**:  
    <img src="plots/1-loss_accuracy.png" alt="Training Metrics" width="500" />  

  - **Backward Propagation Equations**:  
   <div style="display: flex; justify-content: center; gap: 10px;">
  <img src="plots/1-BP1.png" alt="Backward Propagation Part 1" width="400" />
  <img src="plots/1-BP2.png" alt="Backward Propagation Part 2" width="400" />
</div>


2. **[Project 2: MLP using Pytroch ](2-MLP_pytorch.ipynb)**
   - Description: For the same use case (classfication) we use Pytorch, with 3 different implementations : by hand, through the nn.sequential class, and creating a custom class. We make use of "autograd" available in pytorch to perform BP automatically 
   - Plots:  
   ![Losses](plots/2-Pytorch.png)  

3. **[Project 3: RNN for movie rating prediction](3-RNN_classfication_tensorflow.ipynb)**
   - Description:  
   We preprocess movie reviews from IMDB database, and aim to predict the rating 'good' or 'bad'.  
   We first use a simple fully connected approach, and we average the embeddings :  
   review_one_hot_encoded (review_len*dim_dictionnary) --EMBEDDING MATRIX (dim_dictionnary x embedding_dim) --> embedings .  
   Then embeddings (review_len*embedding_dim) --AVERAGE--> avg_embedding(embedding_dim) -- FC + activation --> rating (0 or 1)  
     
  Doing so, we train the EMBEDDING MATRIX. Which grasps the "proximity", in terms of meaning, of words for instance.
  Then we replace the averaging of the embeddings by a LSTM many-to-one : we gain 1 pt of accuracy compared to the fully connected approach.
  
   - Plots:
   <div style="display: flex; justify-content: center; gap: 10px;">
  <img src="plots/3-model-mean.png" alt="Simple Architechture" width="400" />
  <img src="plots/3-model-LSTM.png" alt="LSTM architecture" width="400" />
</div>

4. **[Project 3: Autoregressive models for music generation](4-RNN_autoregressive_tensorflow.ipynb)**
   - Description:  
   Implementing many to many RNN architecture, to autoregressively generate music. 
   Two implementations : using a RNN, using transformers architecture



     