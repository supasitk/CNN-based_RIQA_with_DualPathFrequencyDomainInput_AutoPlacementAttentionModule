# Reimplement_RIQA
Reimplement Baseline of Retinal Image Quality Assessment

## To-do list
1. **Dataset**
    - [x] EyeQ (subset of EyePACS)
2. **Data preparation**
    - [x] EyeQ (preprocess images from EyePACS)
3. **Model structure**
    - [x] OneBranch_ThreeInputChannelRGB
4. **Baseline network**  
    - [x] ResNet (18/34/50/101/152)
5. **Optimizer**
    - [x] SGD
6. **Learning rate scheduler**
    - [x] Fix
    - [x] Exponential decay
    - [x] Cosine decay
    - [ ] Reduce LR on Plateau
    - [ ] Early stopping
7. **Loss function and classification**
    - [x] Multiclass with SparseCategoricalCrossentropy
8. **Performance metric**
    - [x] Loss
    - [x] Accuracy
    - [x] Precision (macro/micro)
    - [x] Sensitivity (macro/micro)
    - [x] F1 (macro/micro)
9. **Choosing Model weight for test**
    - [x] Best step/epoch on validation set
10. **Attention block**
     - **Spatial**
       - [x] SpatialAttention0 [-] (from paper 'Learning for retinal image quality assessment with label regularization [2022])
     - **Channel**
       - [x] ChannelAttention0 [SE] (from paper 'Squeeze-and-Excitation (SE) Networks [2017])
              
