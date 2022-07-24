import os
import time
import numpy as np
from common import Experiment
import lightgbm as lgb
from model import BaselineGruModel
from model_1 import HalfGruModel
import pandas as pd
import torch
flag=0
model_list=[]

def lgn_predict_split_com(x,turb,mod_list):

    result_list = []
    input_list = []
    n = 2
    #k = 96 // n+192//(8*n)

    l=96//n
    k=l
    data = x
    for i in range(len(data)):
        #col = [0,3,9]
        col = [0,2,3,9]

        val = data[i][:, col]
        for i in range(len(val[0])):
            ind=np.isnan(val[:,i])
            m=val[:,i][~ind].mean()
            val[:,i][ind]=m
        val = np.vstack([val[n * i:n * (i + 1)].mean(0) for i in range(len(val) // n)])[::-1]
        val = val.reshape(len(col) * 14 * 144 // n)
        input_list.append(val)
    x = np.array(input_list)
    x = x[:, :int(x.shape[1] * 1.5 // 14)]
    for i in range(k):
        mod = mod_list[i]
        if i>=l:
            res = mod.predict(x).reshape(-1, 1).repeat(8*n, 1).reshape(-1)
        else:
            res = mod.predict(x).reshape(-1, 1).repeat(n, 1).reshape(-1)
        result_list.append(res)
    result_list = np.concatenate(result_list+[np.zeros(192)])
    # result_list = result_list.reshape(1, 288, 1)
    result_list = result_list.reshape(288, 1)
    result_list[result_list>1500]=1500
    result_list[result_list<0]=0
    return result_list

def forecast_one_from_lstm(experiment, test_turbines, mean, std):


    args = experiment.get_args()
    tid = args["turbine_id"]

    model = BaselineGruModel(args)
    model_first_half = HalfGruModel(args)
    model_dir = 'lstm'
    model_dir_first_half = 'model_first_144'
    ############################################
    path_to_model = os.path.join(args["checkpoints"], model_dir, "model_{}".format(str(tid)))
    pretrained_dict = torch.load(path_to_model)
    pretrained_dict = {key.replace("module.", ""): value for key, value in pretrained_dict.items()}
    model.load_state_dict(pretrained_dict)
    ############################################
    path_to_model = os.path.join(args["checkpoints"], model_dir_first_half, "model_{}".format(str(tid)))
    pretrained_dict = torch.load(path_to_model)
    pretrained_dict = {key.replace("module.", ""): value for key, value in pretrained_dict.items()}
    model_first_half.load_state_dict(pretrained_dict)
    ############################################
    test_x = test_turbines.get_turbine(tid)
    test_x = (test_x - mean[tid]) / std[tid]
    input_len = 144
    last_observ = test_x[-input_len:]
    seq_x = torch.from_numpy(last_observ).type(torch.float32)
    sample_x = torch.reshape(seq_x, [-1, seq_x.shape[-2], seq_x.shape[-1]])
    if args["use_gpu"]:
        sample_x = sample_x.cuda()
        model = model.cuda()
        model_first_half = model_first_half.cuda()
    ############################################
    # prediction : (288 ,1)
    # prediction_first_half : (144, 1)
    ############################################
    prediction = experiment.inference_one_sample_from_lstm(model, sample_x)
    prediction_first_half = experiment.inference_one_sample_from_lstm(model_first_half, sample_x)
    prediction = prediction[0]
    prediction_first_half = prediction_first_half[0]
    prediction = prediction.detach().cpu().numpy()
    prediction_first_half = prediction_first_half.detach().cpu().numpy()
    ############################################
    alpha = 0.5
    prediction[0:144, :] = alpha * prediction[0:144, :] + (1. - alpha) * prediction_first_half[0:144, :]
    ############################################
    return np.array(prediction)

def forecast(settings):
    global flag,model_list
    if flag==0:
        for i in range(134):
            model_turb=[]
            for k in range(48):
                path=os.path.join(settings["checkpoints"], 'lgn', 'turb' + str(i) + '_drift' + str(k + 1) + '.txt')
                mod = lgb.Booster(model_file=path)
                model_turb.append(mod)
            model_list.append(model_turb)
        flag=1
    start_time = time.time()
    predictions_lstm = []
    predictions_lgn = []
    settings["turbine_id"] = 0
    ###########################
    path = os.path.join(settings["checkpoints"], "sim.npy")
    sim=np.load(path)
    mean_path = os.path.join(settings["checkpoints"], 'mean.txt')
    mean = np.loadtxt(mean_path, dtype=np.float32)
    std_path = os.path.join(settings["checkpoints"], 'std.txt')
    std = np.loadtxt(std_path, dtype=np.float32)
    ###########################
    exp = Experiment(settings)
    test_x = Experiment.get_test_x(settings)
    ############
    data=[]
    for i in range(134):
        test_data=test_x.get_turbine(i)
        seq_x = test_data[-settings["input_len"]:]
        df=pd.DataFrame(seq_x)
        df['time']=np.arange(len(df))
        df['turb']=i
        # print(df)
        data.append(df)
    data=pd.concat(data)
    df_median = data.groupby(['time'])[2, 3].median().reset_index()
    new_col = []
    for s in df_median.columns:
        if s != 'time':
            new_col.append(s + 100)
        else:
            new_col.append(s)
    df_median.columns = new_col
    data = pd.merge(data, df_median, how='left', on=['time'])
    col = [2, 3]
    for s in col:
        v1 = data[s].values
        v2 = data[s + 100].values
        ind = np.abs(v1 - v2) > 10
        v1[ind] = v2[ind]

    for i in range(settings["capacity"]):
        settings["turbine_id"] = i
        ####
        ####feature smooth
        sind = np.where(sim[i] != 0)[0]
        data_x = np.array(sim[i][i] * data[data.turb == (i)].values[:,:10])
        for si in sind:
            if si != i:
                data_x += sim[i][si] * np.array(data[data.turb == (si)].values[:,:10])
        #####feature smooth
        prediction_lgn=lgn_predict_split_com(data_x[np.newaxis,:],i,model_list[i])
        prediction_lstm = forecast_one_from_lstm(exp, test_x, mean, std)
        ###################
        # inverse transform
        prediction_lstm = (prediction_lstm * std[i]) + mean[i]
        ###
        torch.cuda.empty_cache()
        predictions_lstm.append(prediction_lstm)
        predictions_lgn.append(prediction_lgn)

    predictions_lgn = np.array(predictions_lgn)
    predictions_lstm = np.array(predictions_lstm)
    predictions = np.zeros([134, 288, 1])
    predictions[:,0:96,:] =  0.8 * predictions_lgn[:,0:96,:] + 0.2 * predictions_lstm[:,0:96,:]
    predictions[:,96:288,:] = predictions_lstm[:,96:288,:]
    predictions = np.array(predictions).reshape(134,288)
    predictions = sim.dot(sim).dot(predictions).reshape(134,288,1)
    predictions[:,96:288,:] = predictions_lstm[:,96:288,:]
    ######################################################
    predictions[:,0:96,:] += 40.0
    predictions[:,96:144,:] += 5.0
    predictions[:,144:288,:] += 11.0
    ######################################################
    return np.array(predictions)