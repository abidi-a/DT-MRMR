function [TrainData,TestData,R] = ManageData(data)


Label = data.lables;
%% Finding Missing(NaN) elements
data = data.data;
data = knnimpute(data);


Inputs = data;
Targets = Label;
NSamples = size(Inputs,1);
%% Normalaization

% Inputs = Normalaization(Inputs);
%% Test and Train Data

TrPercent = 80;
TrNum = round(NSamples * TrPercent / 100);

 R = randperm(NSamples);
load DataSets/R
trIndex = R(1 : TrNum);
tsIndex = R(1+TrNum : end);

TrainData.Inputs = Inputs(trIndex,:);
TrainData.Targets = Targets(trIndex,:);

TestData.Inputs = Inputs(tsIndex,:);
TestData.Targets = Targets(tsIndex,:);

end


function X = Normalaization(X)

Min = min(X);
Max = max(X);

for i = 1:numel(Min)
    
   X(:,i) = 2*(X(:,i) -Min(i))/(Max(i) - Min(i))-1 ; 
    
end



end