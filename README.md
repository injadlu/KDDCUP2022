# SlienceG Team Solution for KDDCUP2022  
for each turbine, we have 3 models:  
1 for lightgbm(namely lgn)  
1 for first 144 timestamps prediction(namely GRU-FH)  
1 for 288 timestamps prediction(namely GRU-ALL).  
more information about [KDDCUP2022](https://aistudio.baidu.com/aistudio/competition/detail/152/0/introduction)  
## model training:  
**before training, you need to upload the dataset into folder LGB_train and GRU_train.**  
### for LGB_train:  
 1. modify the file prepare.py to your path.  
 2. data proprecess  
 ```Bash  
 python datapreprocess.py #Bash
 ```  
 3. LGB Train  
 ```Bash
 python train_split_smooth.py #Bash
 ```
### for GRU_train:
1. modify the file prepare.py to your path.  
2. if you want to train GRU-ALL, set the output_len 288,  
   elif you want to train GRU-FH, set the output_len 144. 
3. model training
```Bash
python train.py #Bash
```
## model testing:
**before testing, put the pretrained model into folder checkpoints of prediction**  
1. modify the file prepare.py and predict.py to your model path.  
2. submit the prediction folder for online test.  
**Any problems please contact me at jackie64321@gmail.com**
