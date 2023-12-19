function Results = EvaluatePlot(Targets,Groups,score,Name)

confmat = confusionmat(Targets,Groups);
Results = getMetrics(confmat);

[Targets,Groups] = Vec2Bin(Targets',Groups');
figure,plotconfusion(Targets,Groups)
title(['CM for ',Name,' Data'])
score=score';
if size(confmat,1)==2

[~,~,~,AUC] = perfcurve(Targets(1,:),score(1,:),1);
figure,plotroc(Targets(1,:),score(1,:))
else
for i = 1:size(confmat,1)
[~,~,~,AUC(i)] = perfcurve(Targets(i,:),score(i,:),1);
end


figure,plotroc(Targets,score)
end
title(['ROC for ',Name,' Data'])

Results.AUC = mean(AUC);
disp(['Results For ',Name,' Data'])
disp(Results)
disp(' **************************')
end

function [Targets,Groups] = Vec2Bin(Targets,Groups)

Tu = unique(Targets);
Temp1 = zeros(size(Targets));
Temp2 = Temp1;
for i = 1:numel(Tu)
    Ind1 = find(Targets==Tu(i));
    Temp1(Ind1) = i;

    Ind2 = find(Groups==Tu(i));
    if isempty(Ind2)
        Ind2 = Ind1(1);
    end

    Temp2(Ind2) = i;

end

Targets = full(ind2vec(Temp1));
Groups = full(ind2vec(Temp2));
end


function  [Result,RefereceResult]= getMetrics(c_matrix)

%1.TP-True Positive   %2.FP-False Positive
%3.FN-False Negative  %4.TN-True Negative
n_class = size(c_matrix,1);

switch n_class
    case 2
        TP=c_matrix(1,1);
        FN=c_matrix(1,2);
        FP=c_matrix(2,1);
        TN=c_matrix(2,2);

    otherwise
        TP=zeros(1,n_class);
        FN=zeros(1,n_class);
        FP=zeros(1,n_class);
        TN=zeros(1,n_class);
        for i=1:n_class
            TP(i)=c_matrix(i,i);
            FN(i)=sum(c_matrix(i,:))-c_matrix(i,i);
            FP(i)=sum(c_matrix(:,i))-c_matrix(i,i);
            TN(i)=sum(c_matrix(:))-TP(i)-FP(i)-FN(i);
        end

end

% 1.acuuracy   2.Sensitivity (Recall or True positive rate) 3.Specificity
% 4.Precision  5.FPR-False positive rate   6.F_score  7. Error

P=TP+FN;
N=FP+TN;
switch n_class
    case 2
        accuracy=(TP+TN)/(P+N);
        Error=1-accuracy;
        Result.Accuracy=(accuracy);
        Result.Error=(Error);
    otherwise
        accuracy=(TP)./(P+N);
        Error=(FP)./(P+N);
        Result.Accuracy=sum(accuracy);
        Result.Error=sum(Error);
end
RefereceResult.AccuracyOfSingle=(TP ./ P)';
RefereceResult.ErrorOfSingle=1-RefereceResult.AccuracyOfSingle;
Sensitivity=TP./P;
Specificity=TN./N;
Precision=TP./(TP+FP);
FPR=1-Specificity;
F1_score=( (2)*(Sensitivity.*Precision) ) ./ ( (Precision+Sensitivity) );

%%
%Output Struct for individual Classes
%  RefereceResult.Class=class_ref;
RefereceResult.AccuracyInTotal=accuracy';
RefereceResult.ErrorInTotal=Error';
RefereceResult.Sensitivity=Sensitivity';
RefereceResult.Specificity=Specificity';
RefereceResult.Precision=Precision';
RefereceResult.FalsePositiveRate=FPR';
RefereceResult.F1_score=F1_score';
RefereceResult.TruePositive=TP';
RefereceResult.FalsePositive=FP';
RefereceResult.FalseNegative=FN';
RefereceResult.TrueNegative=TN';


%Output Struct for over all class lists
Result.Sensitivity=nanmean(Sensitivity);
Result.Specificity=nanmean(Specificity);
Result.Precision=nanmean(Precision);
Result.FalsePositiveRate=nanmean(FPR);
Result.F1_score=nanmean(F1_score);
Result.TruePositive=TP';
Result.FalsePositive=FP';
Result.FalseNegative=FN';
Result.TrueNegative=TN';

end