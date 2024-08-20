

# %%
##importing the libraries needed
import alibi
import numpy as np
import pandas as pd
import tensorflow as tf
tf.get_logger().setLevel(40) # suppress deprecation messages
tf.compat.v1.disable_v2_behavior() # disable TF2 behaviour as alibi code still relies on TF1 constructs  
from math import floor, ceil
import sklearn as sk
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Bidirectional, LSTM,Masking,Embedding
from tensorflow.keras.models import Model, load_model
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, accuracy_score, make_scorer
from sklearn.model_selection import cross_validate,train_test_split,GridSearchCV
from sklearn.preprocessing import normalize
from tensorflow.keras.models import load_model 
#from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from scikeras.wrappers import KerasClassifier

import matplotlib.pyplot as pyplot
import seaborn as sns
import time
import json
import ast
import os
import argparse

# Function to parse command line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Description of your program")
    parser.add_argument('--COURSE', type=str, required=True, help='Course name')
    return parser.parse_args()


# %%
# EDIT HERE FOR OTHER COURSES

# Parse arguments
args = parse_arguments()
course = args.COURSE

# Validate the course name
valid_courses = ["villesafricaines_001", "geomatique_003", "dsp_001"]
assert course in valid_courses, f"Invalid course name: {course}. Must be one of {valid_courses}"



# %%

data_path = '../easy-fail/'
week_type = 'eq_week'
feature_types = [ 'boroujeni_et_al', 
                 'chen_cui',  'lalle_conati', 
                 'marras_et_al']

marras_et_al_id = feature_types.index('marras_et_al')
#akpinar_et_al_id = feature_types.index('akpinar_et_al')
remove_obvious = False

# %% [markdown]
# ## Loading Data

# %%
def fillNaN(feature):
    shape = feature.shape
    feature_min = np.nanmin(feature.reshape(-1,shape[2]),axis=0)
    feature = feature.reshape(-1,shape[2])
    inds = np.where(np.isnan(feature))
    feature[inds] = np.take(feature_min.reshape(-1), inds[1])
    feature = feature.reshape(shape)
    return feature

# %%
# Loading the features
feature_list = {}

feature_type_list = []
for feature_type in feature_types:

    filepath = data_path + week_type + '-' + feature_type + '-' + course
    feature_current = np.load(filepath+'/feature_values.npz')['feature_values']
    print(feature_current.shape)
    feature_norm = feature_current.reshape(-1,feature_current.shape[2] )
    print(feature_norm.shape)
    feature_type_list.append(pd.DataFrame(feature_norm))
feature_list[course] = feature_type_list

print('course: ', course)
print('week_type: ', week_type)
print('feature_type: ', feature_types)

# %%
# Loading feature names
feature_names= dict()

for feature_type in feature_types:
    
    filepath = data_path + week_type + '-' + feature_type + '-' + course + '/settings.txt'
    file = open(filepath, "r")
    contents = file.read()
    dictionary = ast.literal_eval(contents)
    file.close()
    
    feature_type_name = dictionary['feature_names']
    
    if feature_type == 'akpinar_et_al':
        feature_type_name = [clean_akp_name(x) for x in feature_type_name]
        akp_mask = np.where(np.isin(feature_type_name, 
                 ["TotalClicks", "NumberSessions", "Time-video-sum", "Time-problem-sum",
                  'problem.check-problem.check-problem.check', 
                  'problem.check-problem.check-video.load', 
                  'video.play-video.play-video.play',
                  'video.play-video.pause-video.load',
                  'video.play-problem.check-problem.check',
                  'video.play-video.stop-video.play',
                  'video.pause-video.speedchange-video.play',
                  'video.stop-video.play-video.seek',
                  'video.stop-problem.check-video.load']))
        print(akp_mask)
        feature_type_name = list(np.array(feature_type_name)[akp_mask[0]])
        feature_list[course][akpinar_et_al_id] = feature_list[course][akpinar_et_al_id][akp_mask[0]]
        
    feature_names[feature_type] = feature_type_name
    print(feature_type, len(feature_type_name))

if remove_obvious: 
    # drop 'student shape', 'competency strength', 'competency alignment' in marras at al
    
    mask = np.where(np.isin(feature_names['marras_et_al'], 
                 ['StudentShape', 'CompetencyStrength', 'CompetencyAlignment']))
    
    new_marras = np.delete(np.array(feature_names['marras_et_al']), mask[0])
    feature_names['marras_et_al'] = new_marras
    
    new_features = feature_list[course][marras_et_al_id].drop(mask[0], axis=1)
    feature_list[course][marras_et_al_id] = new_features


# %%
# reformat feature names
# ex: time_sessions_<function sum at 0x7f3bd02cc9d0> -> time_sessions_sum
def clean_name(feature):
    id = feature.find('<')
    if id==-1:
        return feature
    fct = feature[id+9:id+14].strip()
    return feature[0:id]+fct


for k in feature_names.keys():
    cleaned = [clean_name(x) for x in feature_names[k]]
    feature_names[k] = cleaned

def clean_akp_name(feature):
    feature = feature.lower()
    if feature.find("(")!=-1:
        feature = feature[1:-1]
        feature = feature.replace(', ', '-')
    return feature

# %%
# TO CHANGE HERE THE WEEKS NUMBER
overwrite_num_weeks = 5

# %%
def load_labels(course):
    feature_type = "boroujeni_et_al"
    filepath = data_path + week_type + '-' + feature_type + '-' + course + '/feature_labels.csv'
    labels = pd.read_csv(filepath)['label-pass-fail']
    labels[labels.shape[0]] = 1
    return labels.values

def load_features(course, overwrite_num_weeks):
    feature_list = []
    selected_features = []
    total_features = set()
    num_weeks = 0
    num_features = 0
    for i,feature_type in enumerate(feature_types):
        filepath = data_path + week_type + '-' + feature_type + '-' + course 
        feature_current = np.load(filepath+'/feature_values.npz')['feature_values']
        
        shape = feature_current.shape
#         print(shape)

        if remove_obvious and feature_type=='marras_et_al':
            feature_current = np.delete(feature_current, mask[0], axis=2)
        
        if feature_type=='akpinar_et_al':
            akp_mask_dl = np.delete(list(range(shape[2])), akp_mask[0])
            feature_current = np.delete(feature_current, akp_mask_dl, axis=2)
        
        shape = feature_current.shape
        print(shape)
        if i==0:
            num_weeks = shape[1]
            num_weeks=overwrite_num_weeks
            
        selected = np.arange(shape[2])
        # drop existed features
        exist_mask = []
        for i, name in enumerate(feature_names[feature_type]):
            if name in total_features:
                exist_mask.append(i)
            else:
                total_features.add(name)
        feature_current = np.delete(feature_current, exist_mask, axis=2)
        selected = np.delete(selected, exist_mask)
        
        nonNaN = (shape[0]*shape[1] - np.isnan(feature_current.reshape(-1,feature_current.shape[2])).sum(axis=0) > 0)
        feature_current = feature_current[:,:,nonNaN]
        selected = selected[nonNaN]
        feature_current = fillNaN(feature_current)
        nonZero = (abs(feature_current.reshape(-1,feature_current.shape[2])).sum(axis=0)>0)
        selected = selected[nonZero]
        feature_current = feature_current[:,:overwrite_num_weeks,nonZero]
#         print(len(feature_names[feature_type]), selected)
        selected_features.append(np.array(feature_names[feature_type])[[selected]])
        num_features += len(np.array(feature_names[feature_type])[[selected]])


        ##### Normalization with min-max. Added the artifical 1.001 max row for solving the same min max problem
        ##### for features with max=0 I added 1 instead of 1.001 of maximum

        features_min = feature_current.min(axis=0).reshape(-1)
        features_max = feature_current.max(axis=0)
        features_max = np.where(features_max==0,np.ones(features_max.shape),features_max)
        max_instance = 1.001*features_max
        feature_current = np.vstack([feature_current,max_instance.reshape((1,)+max_instance.shape)])
        features_max = features_max.reshape(-1)
        feature_norm = (feature_current.reshape(shape[0]+1,-1)-features_min)/(1.001*features_max-features_min)
        feature_current = feature_norm.reshape(-1,feature_current.shape[1],feature_current.shape[2] )

        num_features += feature_current.shape[2]
        print(num_features)
        feature_list.append(feature_current)
        
    features = np.concatenate(feature_list, axis=2)
    features_min = features.min(axis=0).reshape(-1)
    features_max = features.max(axis=0)
    features = features.reshape(features.shape[0],-1)
#     features = pd.DataFrame(features)
    
    SHAPE = features.shape
    # print(np.isnan(features[0,0,-1]))
    print(features.shape)
    print('course: ', course)
    print('week_type: ', week_type)
    print('feature_type: ', feature_types)
    print(selected_features)
    return features, features_min, features_max, selected_features, num_weeks, num_features


# %%
y = load_labels(course)

# %%
features, features_min, features_max, selected_features, num_weeks, num_features = load_features(course, overwrite_num_weeks)

# %%
feature_dict = dict()
for i, feature_type in enumerate(feature_types):
    feature_dict[feature_type] = list(selected_features[i])
    
selected_features = feature_dict

# %%
# Loading feature names and transforming them to 2D format.
feature_names = []
final_features = []
for i,feature_type in enumerate(feature_types):
    [final_features.append(x) for x in selected_features[feature_type]]

# %%
len(final_features[2])

# %%
flat_final_features = [item for sublist in final_features for item in sublist]
print(final_features)
print(type(final_features))
print(type(num_weeks))

# %%
for i in np.arange(num_weeks):
    feature_type_name_with_weeks = [(x+'_InWeek'+str(i+1)) for x in flat_final_features]
    feature_names.append(feature_type_name_with_weeks)
feature_names = np.concatenate(feature_names, axis=0)
feature_names = feature_names.reshape(-1)
print(feature_names)
#features.columns = feature_names

# %%
labels=np.concatenate(((1-y).reshape(-1,1),y.reshape(-1,1)),axis=1)
labels.shape

# %%
features.shape

# %%
f = pd.DataFrame(features, columns=feature_names)

# %%
s_f = list(selected_features.values())
num_features = len([feature for feature_group in final_features for feature in feature_group])

# %%
num_features

# %% [markdown]
# ## Model
# A new model has to be trained for CEM, since it needs a target variable of a different shape (n_instances, 2)

# %%
def bidirectional_lstm(x_train, y_train, x_test, y_test, x_val, y_val, week_type, feature_types, course,n_weeks,n_features, num_epochs=100):
    n_dims = x_train.shape[0]
    look_back = 3
    # LSTM
    # define model
    lstm = Sequential()
    ###########Reshape layer################
    lstm.add(tf.keras.layers.Reshape((n_weeks, n_features), input_shape=(n_weeks*n_features,)))
    ##########deleting the 1.001 max row added###########
    lstm.add(Masking(mask_value = 1))
    lstm.add(Bidirectional(LSTM(64, return_sequences=True)))
    lstm.add(Bidirectional(LSTM(32)))
    # Add a sigmoid Dense layer with 1 units.
    lstm.add(Dense(2, activation='sigmoid'))
    # compile the model
    lstm.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # fit the model
    history = lstm.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=num_epochs, batch_size=32, verbose=1)
    # evaluate the model
    y_pred = lstm.predict(x_test)
    print(y_pred.shape)
    y_pred = np.array([1 if y >= 0.5 else 0 for y in y_pred[:,1]])
    print(y_pred.shape)
    y_pred = np.concatenate(((1-y_pred).reshape(-1,1),y_pred.reshape(-1,1)),axis=1)
    print(y_pred.shape)
    # evaluate the model
    model_params = {'model': 'LSTM-bi', 
                    'epochs': num_epochs, 
                    'batch_size': 32, 
                    'loss': 'binary_cross_entropy'}
    scores = evaluate(None, x_test, y_test, week_type, feature_types, course, y_pred=y_pred, model_name="TF-LSTM-bi", model_params=model_params)
    lstm.save('../models/lstm_bi_'+course+'_cem')
    return history, scores

# %%
def plot_history(history, file_name):
    # plot loss during training
    pyplot.figure(0)
    pyplot.title('Loss ' + file_name)
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.xlabel("epoch")
    pyplot.ylabel("loss")
    pyplot.legend()
    pyplot.savefig(file_name + "_loss.png")
    # plot accuracy during training
    pyplot.figure(1)
    pyplot.title('Accuracy ' + file_name)
    pyplot.plot(history.history['acc'], label='train')
    pyplot.plot(history.history['val_acc'], label='test')
    pyplot.legend()
    pyplot.xlabel("epoch")
    pyplot.ylabel("accuracy")
    pyplot.savefig(file_name + "_acc.png")


# %%
def evaluate(model, x_test, y_test, week_type, feature_type, course, model_name=None, model_params=None, y_pred=None):
    scores={}
    y_test=y_test[:,1]
    y_pred=y_pred[:,1]
    scores['test_acc'] = accuracy_score(y_test, y_pred)
    scores['test_bac'] = balanced_accuracy_score(y_test, y_pred)
    scores['test_prec'] = precision_score(y_test, y_pred)
    scores['test_rec'] = recall_score(y_test, y_pred)
    scores['test_f1'] = f1_score(y_test, y_pred)
    scores['test_auc'] = roc_auc_score(y_test, y_pred)
    scores['feature_type'] = feature_type
    scores['week_type'] = week_type
    scores['course'] = course
    scores['data_balance'] = sum(y)/len(y)
    return scores

# %%
print(features.shape)
labels.shape

# %% [markdown]
# ## Explainers

# %%
import alibi
from alibi.explainers import CEM

# %%
bilstm = load_model('models/lstm_bi_'+course+'_cem')


# %%
def pn_all(num_instances,features,feature_names):
    mode = 'PN'  # 'PN' (pertinent negative) or 'PP' (pertinent positive)
    shape = (1,) + features.shape[1:]  # instance shape
    kappa = 0.  # minimum difference needed between the prediction probability for the perturbed instance on the
              # class predicted by the original instance and the max probability on the other classes 
              # in order for the first loss term to be minimized
    beta = .1  # weight of the L1 loss term
    gamma = 100  # weight of the optional auto-encoder loss term
    c_init = 1.  # initial weight c of the loss term encouraging to predict a different class (PN) or 
                # the same class (PP) for the perturbed instance compared to the original instance to be explained
    c_steps = 10  # nb of updates for c
    max_iterations = 1000  # nb of iterations per value of c
    feature_range = (features.min(axis=0),features.max(axis=0)) # feature range for the perturbed instance
    clip = (-1000.,1000.)  # gradient clipping
    lr = 1e-2  # initial learning rate
    no_info_val = -1. # a value, float or feature-wise, which can be seen as containing no info to make a prediction
                    # perturbations towards this value means removing features, and away means adding features
                    # for our MNIST images, the background (-0.5) is the least informative, 
                    # so positive/negative perturbations imply adding/removing features
    cem = CEM(bilstm, mode, shape, kappa=kappa, beta=beta, feature_range=feature_range, 
    gamma = gamma, ae_model=None, max_iterations=max_iterations, 
    c_init = c_init, c_steps=c_steps, learning_rate_init=lr, clip=clip, no_info_val=no_info_val)
    changes=[]
    explanations = []
    final_num_instances = []
    count=0
    for i in num_instances:
        try:
            X = features[i].reshape((1,) + features[0].shape)
            explanation = cem.explain(X)
            change = explanation.PN-X
            print(f'counterfactuals generated for instance {i}')
            changes.append(change)
            explanations.append(explanation)
            final_num_instances.append(i)
            count+=1
            print(f'\n\n👩‍🚀🚀 Houston, counterfactuals generated for {count} instances!\n\n')
        except TypeError:
            print(f'Error occured for instance {i}')
            print(change)
    return explanations, changes, final_num_instances

# %%
def pp_all(num_instances,features,feature_names):
    mode = 'PP'  # 'PN' (pertinent negative) or 'PP' (pertinent positive)
    shape = (1,) + features.shape[1:]  # instance shape
    kappa = 0.  # minimum difference needed between the prediction probability for the perturbed instance on the
              # class predicted by the original instance and the max probability on the other classes 
              # in order for the first loss term to be minimized
    beta = .1  # weight of the L1 loss term
    gamma = 100  # weight of the optional auto-encoder loss term
    c_init = 1.  # initial weight c of the loss term encouraging to predict a different class (PN) or 
                # the same class (PP) for the perturbed instance compared to the original instance to be explained
    c_steps = 10  # nb of updates for c
    max_iterations = 1000  # nb of iterations per value of c
    feature_range = (features.min(axis=0),features.max(axis=0)) # feature range for the perturbed instance
    clip = (-1000.,1000.)  # gradient clipping
    lr = 1e-2  # initial learning rate
    no_info_val = -1. # a value, float or feature-wise, which can be seen as containing no info to make a prediction
                    # perturbations towards this value means removing features, and away means adding features
                    # for our MNIST images, the background (-0.5) is the least informative, 
                    # so positive/negative perturbations imply adding/removing features
    cem = CEM(bilstm, mode, shape, kappa=kappa, beta=beta, feature_range=feature_range, 
    gamma = gamma, ae_model=None, max_iterations=max_iterations, 
    c_init = c_init, c_steps=c_steps, learning_rate_init=lr, clip=clip, no_info_val=no_info_val)
    changes=[]
    for i in num_instances:
        try:
            X = features[i].reshape((1,) + features[0].shape)
            explanation = cem.explain(X)
            change = explanation.PP-X
            print(f'counterfactuals generated for instance {i}')
            changes.append(change)
        except TypeError:
            print(f'Error occured for instance {i}')
            print(change)
    return changes

# %%
############################################## CODE TO GET THE INDICES OF THE FEATURES
# I pick any feature set here and take the indices that are not easy fail
index=pd.read_csv("../easy-fail/eq_week-chen_cui-" + str(course) + "/feature_labels.csv",)
index

# Make the index a column called 'feature_index'
index.reset_index(inplace=True)
index.rename(columns={'index': 'feature_index'}, inplace=True)

# Rename 'Unnamed: 0' to 'original_index'
index.rename(columns={'Unnamed: 0': 'original_index'}, inplace=True)

# Remove the other columns except 'feature_index' and 'original_index'
index = index[['feature_index', 'original_index']]

samples_id=pd.read_csv("data/"+str(course) + "_sample_100.csv")
samples_id

# USE THIS INDEX MAPPING TO GET THE INDEX IN THE FEATURES
index_mapping = dict(zip(index['original_index'], index['feature_index']))
index_mapping

samples_id=pd.read_csv("data/"+str(course) + "_sample_100.csv")

samples_id['feature_index'] = samples_id['original_index'].map(index_mapping)

instances=samples_id['feature_index'].values
assert np.all(instances < features.shape[0])

# %%
t1 = time.time()
print('start time', t1)
explanation, changes, final_num_instances = pn_all(instances,features,feature_names)
t2 = time.time()
print('end time', t2)
print(f'time taken: {(t2-t1)/60.0} minutes')

# %%
final_instances = features[final_num_instances]

path = 'uniform_eq_results_ori_5_weeks/CEM/' + course
if not os.path.exists(path):
        os.makedirs(path)
         
np.save(path + '/changes_pn', np.array(changes).reshape(len(final_num_instances),-1))

# %%
pns = np.array([explanation[i].PN for i in range(len(explanation))]).reshape(len(final_num_instances),-1)
np.save(path + '/pns', pns)

# %%
np.save(path +'/instances', final_instances)

# %%
sds = pd.DataFrame(features, columns=feature_names).describe()
sds = sds.loc[:,~sds.columns.duplicated()]
sds = sds.loc['std',:]

# %%
np.array(changes).shape


# %%
sds

# %%
diffs = pd.DataFrame(np.array(changes).reshape(len(final_num_instances),-1), columns=feature_names)
diffs = diffs.loc[:,~diffs.columns.duplicated()]

for col in diffs.columns:
    diffs[col] = np.abs(diffs[col]*(sds[col]))


# %%
diffs.insert(0, 'exp_num', final_num_instances)
diffs.head()

# %%
diffs.to_csv(path +'/importances.csv')


# %%



