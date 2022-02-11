#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import os
import _pickle as pickle
from datetime import datetime 
from scipy.ndimage import gaussian_filter1d
import sklearn
from sklearn import svm
from sklearn.metrics import confusion_matrix



def run_decoding_analyses():
    _config_filename = 'config.pckl'
    with open(_config_filename, 'rb') as _file:
        _config = pickle.load(_file)
    import_path = _config['basic']['data_path']
    export_path = _config['basic']['result_path']
    os.makedirs(import_path,exist_ok=True)
    os.makedirs(export_path,exist_ok=True)

    _dir_pckl = os.listdir(import_path)
    _sort_str = '.pckl'
    _dir_pckl = [x for x in _dir_pckl if x[-len(_sort_str):]==_sort_str]

    df_files = pd.DataFrame(_dir_pckl,columns=['filename'])
    df_files['cate_name'] = df_files['filename'].apply(lambda x:x.split('.')[0])
    df_files['seed'] = df_files['cate_name'].apply(lambda x:int(x.split('_')[-1]))

    _file_list = df_files['filename'].unique()
    for _i_file,_filename in enumerate(_file_list):
        print('>> Decoding dataset ',_i_file+1,' out of ',len(_file_list))
        run_decoding_analysis_on_single_round(_filename,import_path,export_path)



def run_decoding_analysis_on_single_round(_filename,import_path,export_path):

    start_time = datetime.now()

    ## load data
    _key_set = ['MSO_L','MSO_R']
    _load_file = import_path+_filename
    print('> loading simulation data:',_load_file)
    with open(_load_file, 'rb') as _file:
        _loaded_list = pickle.load(_file)
    dict_spike,_elapsed,_key_set,start_position_set_R,ITD_set,fs,seed,df_ITD = _loaded_list

    ## generating spike count dataframe

    print('> computing spike counts')
    # import spike train
    _df_spike_set = []
    for _key in _key_set:
        _spike_set = list(dict_spike[_key].values())
        _firing_rate_matrix = []
        _df_spikes = pd.DataFrame([_spike_set]).T
        _df_spikes.columns = ['Spike']
        _df_spikes['Object'] = _key
        _df_spike_set.append(_df_spikes)
    _df_spike_set = pd.concat(_df_spike_set)
    _df_spike_set['Neuron'] = _df_spike_set.index

    # compute spike count
    def get_spike_count_list(_row):
        _spike = _row['Spike']
        _object = _row['Object']
        _neuron = _row['Neuron']
        _df = df_ITD[['TrialStart','Start_R','End_R']].copy()
        _seq_list = df_ITD['Sequence'].to_list()
        _seq_list = [str(_object)+'^'+str(_neuron)+'^'+str(x) for x in _seq_list] # object, neuron, stimuli sequence
        _sc_list = _df.apply(lambda x:len(_spike[(_spike > (x['TrialStart']+x['Start_R'])/fs)&(_spike < (x['TrialStart']+x['End_R'])/fs)]),axis = 1).to_list()
        _combined_list = list(zip(_seq_list,_sc_list))
        return _combined_list
    
    _df_spike_set['SpikeCountSet'] = _df_spike_set.apply(get_spike_count_list,axis = 1) 
    _df_exploded = _df_spike_set['SpikeCountSet'].explode()
    _df_exploded = pd.DataFrame(_df_exploded)
    _df_exploded['Identifier'] = _df_exploded['SpikeCountSet'].apply(lambda x:x[0])
    _df_exploded['Object'] = _df_exploded['Identifier'].apply(lambda x:x.split('^')[0])
    _df_exploded['Neuron'] = _df_exploded['Identifier'].apply(lambda x:int(x.split('^')[1]))
    _df_exploded['Sequence'] = _df_exploded['Identifier'].apply(lambda x:int(x.split('^')[2]))
    _df_exploded['SpikeCount'] = _df_exploded['SpikeCountSet'].apply(lambda x:x[1])
    _df_exploded = _df_exploded[['Object','Neuron','Sequence','SpikeCount']]


    # merge information
    _df_spike_count = _df_exploded.copy()
    _df_ITD_info = df_ITD[['Sequence','ITD','Replica','Stimulus','Length']]
    df_firing_rate = _df_spike_count.merge(_df_ITD_info,on='Sequence')

    ## decoding analysis
    print('> decoding dataset')
    # pre-processing for decoding analysis 
    _object_set = df_firing_rate.Object.unique()
    _object_ID_mapping = dict(zip(_object_set,range(len(_object_set))))
    df_firing_rate['ObjectID'] = df_firing_rate['Object'].map(_object_ID_mapping)*10000
    df_firing_rate['NeuronID'] = df_firing_rate['Neuron']+df_firing_rate['ObjectID']
    df_firing_rate['ITD'] *=1e6  #change unit to us
    df_firing_rate['SamplingDuration'] = df_firing_rate['Length']/fs
    df_firing_rate['FiringRate'] = df_firing_rate['SpikeCount']/df_firing_rate['SamplingDuration'] 
    df_firing_rate['ITD'] = df_firing_rate['ITD'].apply(int)

    # decoding classification for whole dataset
    df_decoding = df_firing_rate.copy()

    _stimuli_set = df_decoding['Stimulus'].unique()
    _stimulus_cate_mapping = dict(zip(_stimuli_set,range(len(_stimuli_set))))
    df_decoding['StimulusCate'] = df_decoding['Stimulus'].map(_stimulus_cate_mapping)
    df_decoding['Replica'] = df_decoding['Replica']+df_decoding['StimulusCate'] *10000
    df_decoding['Class'] = df_decoding['ITD']
    df_decoding = df_decoding.sort_values('ITD').reset_index(drop=True)
    _class_set = df_decoding.Class.unique()
    _df_subset =df_decoding.copy()
    _table = _df_subset.pivot_table(columns=['Object','Neuron'],index=['ITD','Class','Replica'],values='SpikeCount')
    _data = _table.to_numpy()
    _label = [x[1] for x in _table.index]
    _ITDs = [x[0] for x in _table.index]

    _dict_class_info = {}
    for _class in _df_subset.Class.unique():
        _df = _df_subset[_df_subset['Class']==_class]
        _ITD_set = [_df.ITD.min(),_df.ITD.max()]
        _class_info = str(_ITD_set[0])
        _dict_class_info[_class]=_class_info

    # SVM classification 
    _acc,_conf,_tar,_pred = get_classification_accuracy(response_set=_data,
                                                        label_set = _label,
                                                        switch_plot=False,
                                                        switch_matrix=True)
    print('** decoding accuracy = ',_acc)

    # compute mean squared error
    _df_decoding_error = pd.DataFrame(dict(zip(['Target','Predicted'],[_tar,_pred])))
    _df_decoding_error['SquaredError'] = (_df_decoding_error['Target'] - _df_decoding_error['Predicted'])**2
    _MSE = _df_decoding_error.SquaredError.mean()
    print('** mean squared error = ',np.round(_MSE,2))

    # change save name for full dataset
    _acc_all,_conf_all,_df_decoding_error_all,_MSE_all,_dict_class_info_all = _acc,_conf,_df_decoding_error,_MSE,_dict_class_info

    # compute sensitivity
    df_decoding = df_firing_rate.copy()
    _stimuli_set = df_decoding['Stimulus'].unique()
    _stimulus_cate_mapping = dict(zip(_stimuli_set,range(len(_stimuli_set))))
    df_decoding['StimulusCate'] = df_decoding['Stimulus'].map(_stimulus_cate_mapping)
    df_decoding['Replica'] = df_decoding['Replica']+df_decoding['StimulusCate'] *10000
    df_decoding['Class'] = df_decoding['ITD']
    df_decoding = df_decoding.sort_values('ITD').reset_index(drop=True)

    _class_set = df_decoding.Class.unique()
    _class_sym_set = list(set(abs(_class_set)))
    _class_sym_set.sort()
    _class_sym_set = np.array(_class_sym_set)
    _class_sym_set = _class_sym_set[1:]# remove 0 ITD

    _df_sensitivity = []
    _dict_sensitivity = {}
    for _ITD in _class_sym_set:
        _class_sym = np.concatenate((-np.array([_ITD]),[_ITD]))
        _df_decoding_sym = df_decoding[df_decoding['Class'].isin(_class_sym)]
        _df_subset =_df_decoding_sym.copy()
        _table = _df_subset.pivot_table(columns=['Object','Neuron'],index=['ITD','Class','Replica'],values='SpikeCount')
        _data = _table.to_numpy()
        _label = [x[1] for x in _table.index]
        _ITDs = [x[0] for x in _table.index]

        _dict_class_info = {}
        for _class in _df_subset.Class.unique():
            _df = _df_subset[_df_subset['Class']==_class]
            _ITD_set = [_df.ITD.min(),_df.ITD.max()]
            _class_info = str(_ITD_set[0])
            _dict_class_info[_class]=_class_info

        _acc = get_classification_accuracy(response_set=_data,label_set = _label,switch_plot=False,switch_matrix=False)
        _df_sensitivity.append(pd.DataFrame([[_ITD,_acc]],columns=['ITD','Accuracy']))
        _dict_sensitivity[_ITD] = _acc

    _df_sensitivity = pd.concat(_df_sensitivity)  

    _report_ITD = _df_sensitivity.reset_index().iloc[0].ITD
    _report_acc = _df_sensitivity.reset_index().iloc[0].Accuracy
    print('** accuracy at',_report_ITD,'us ITD = ',_report_acc)

    ## _save_results
    _save_name = export_path+'Decoding_'+_filename
    _save_obj = [  
                    [df_firing_rate['SpikeCount'].to_list(),df_firing_rate['NeuronID'].to_list(),df_firing_rate['Sequence'].to_list()],
                    [_acc_all,_conf_all,_df_decoding_error_all,_MSE_all,_dict_class_info_all],
                    _df_sensitivity,
                ]

    with open(_save_name, 'wb') as _file:
        pickle.dump(_save_obj, _file)
    print('> saved results: '+_save_name)

    _elapsed = datetime.now() - start_time
    print('** time spent: '+str(_elapsed))




# function to compute PSTH using gaussian filter
def get_firing_rate_function(
                             spike_train,      # input spike train
                             sigma = 0.025,    # sigma for gaussian filter in sec
                             precision = 1000, # precision of input spike time 
                                               #    e.g. default case 100 precision: 1.2345 -> 1.234
                            ):

    # input regularization to integers 
    spike_train = np.array(spike_train)
    _spike_train = np.round(spike_train.astype(np.double)*precision, decimals = 0).astype(int)
    _length = _spike_train.max()+1
    _sigma = sigma*precision

    # generate time and spike stack
    _time = np.arange(0,_length,1)/precision
    _spike_stack = np.zeros(len(_time))
    for _spike in _spike_train:
        _spike_stack[_spike]+=1

    # compute PSTH by gaussian filter 
    _PSTH = gaussian_filter1d(_spike_stack,_sigma)*precision
    return _time,_PSTH


# function to compute decoding accuracy using SVM
def get_classification_accuracy(response_set = None,label_set = None,switch_plot = False,switch_matrix = False,title =''):
    if response_set is None:
        response_set = self.response_set.copy()
    if label_set is None:
        label_set = self.label_set.copy()
    chosen_data = response_set.copy()
    chosen_target = label_set.copy()

    #result vector
    ML_target = []
    ML_prediction = []

    #leave-one-out cross-validation
    np.random.seed(0)
    validation_sequence = np.random.permutation(len(chosen_data))

    for _i_trial,_taken_out_trial in enumerate(validation_sequence):
        train_data = np.array([x for i,x in enumerate(chosen_data) if i!=_taken_out_trial])
        train_target = np.array([x for i,x in enumerate(chosen_target) if i!=_taken_out_trial])
        test_data = np.array([x for i,x in enumerate(chosen_data) if i==_taken_out_trial])
        test_target = np.array([x for i,x in enumerate(chosen_target) if i==_taken_out_trial])
        
        #classification 
        clf = sklearn.svm.SVC(kernel='linear')
        clf.fit(train_data, train_target)
        _prediction_result = clf.predict(test_data)
        ML_target.append(test_target)
        ML_prediction.append(_prediction_result)

    ML_target = np.array(ML_target).transpose()[0]
    ML_prediction = np.array(ML_prediction).transpose()[0]
    ML_accuracy = len([i for i in range(len(ML_target)) if ML_target[i]==ML_prediction[i]])/len(ML_target)

    _retrun = ML_accuracy
    if switch_matrix:
        ML_confusion_matrix = confusion_matrix(ML_target, ML_prediction)
        _retrun = [ML_accuracy,ML_confusion_matrix,ML_target,ML_prediction]
        
    return _retrun





