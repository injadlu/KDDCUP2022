# SlienceG Team Solution for KDDCUP2022  
for each turbine, we have 3 models:  
1 for lightgbm(namely lgn)  
1 for first 144 timestamps prediction(namely GRU-FH)  
1 for 288 timestamps prediction(namely GRU-ALL).  
more information about [KDDCUP2022](https://aistudio.baidu.com/aistudio/competition/detail/152/0/introduction)  
## model training  
**before training, you need to upload the dataset into folder LGB_train and GRU_train.**
for LGB_train:  
 1. data proprecess  
 '<python datapreprocess.py>'  
 2. LGB Train  
 '<python train_split_smooth.py>'
