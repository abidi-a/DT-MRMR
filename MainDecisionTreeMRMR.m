clc
clear
close all

% Decision Tree (CART)
% for Breast Cancer Detection

%% Load Data with variable's name
data = load('DataSets/breast_cancer_wisconsin.mat');
Name = {'Clump Thickness'         % ضخامت انبوه
    'Uniformity of Cell Size'     % یکنواختی سایز سلول
    'Uniformity of Cell Shape'    % یکنواختی شکل سلول
    'Marginal Adhesion'           % چسبندگی لبه
    'Single Epithelial Cell Size'
    'Bare Nuclei'                 % هسته عریان
    'Bland Chromatin'             % کروماتین بلاند 
    'Normal Nucleoli'             % هسته نرمال
    'Mitoses'};                   % تقسیم
%% Manege Data
[TrainData,TestData] = ManageData(data);
%% Feature Selection MRMR (a filter approach)
% Minimum Redundancy Maximum Relevance
[idx,scores] = fscmrmr(TrainData.Inputs,TrainData.Targets);
nS = 3;   % Number of Selected Features
S = idx(1:nS);  % The Best Feature Set that MRMR has suggested.

%% Trainiing DT Algorithm (Create Model)
DT = fitctree(TrainData.Inputs(:,S),TrainData.Targets,...
'PredictorNames',Name(S));% 'MaxNumSplits',698

% 'ClassNames',{'Benign = 2','Malignant = 4'}

%% Examine the Train model by Trainibg and Testing data
[Groups,Score]= predict(DT,TrainData.Inputs(:,S));
ResultsTrain = EvaluatePlot(TrainData.Targets,Groups,Score,'Train');

[Groups,Score]= predict(DT,TestData.Inputs(:,S));
ResultsTest = EvaluatePlot(TestData.Targets,Groups,Score,'Test');
%% Predictor importance estimates
imp = predictorImportance(DT);

figure;
bar(imp);
title(' The Importance of Features');
ylabel('Importance');
xlabel('Predictors');
xticklabels(Name(S))
%% View Tree
view(DT,'Mode','graph')


