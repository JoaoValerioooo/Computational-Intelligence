%% Data
% Import data
data = importdata("caltech101_silhouettes_28.mat");
% Input + Permutation
Input = permute(data.X, [2, 1]);
% Output + One hot encode
Output = onehotencode(data.Y,1,"ClassNames",1:101);


%% Point 2 of the Report

%[tr,accuracies] = TrainMLP(Input,Output,50,0.4,0.2,0.4,2000,"traingdx","logsig","logsig","mse",0.8,0.01)
%[tr,accuracies] = TrainMLP(Input,Output,50,0.1,0.1,0.8,2000,"traingdx","logsig","logsig","mse",0.8,0.01)
%[tr,accuracies] = TrainMLP(Input,Output,50,0.8,0.1,0.1,2000,"traingdx","logsig","logsig","mse",0.8,0.01)
%[tr,accuracies] = TrainMLP(Input,Output,50,0.4,0.2,0.4,2000,"traingdx","logsig","softmax","crossentropy",0.8,0.01)
%[tr,accuracies] = TrainMLP(Input,Output,50,0.1,0.1,0.8,2000,"traingdx","logsig","softmax","crossentropy",0.8,0.01)
%[tr,accuracies] = TrainMLP(Input,Output,50,0.8,0.1,0.1,2000,"traingdx","logsig","softmax","crossentropy",0.8,0.01)

%[tr,accuracies] = TrainMLP(Input,Output,200,0.4,0.2,0.4,2000,"traingdx","logsig","logsig","mse",0.8,0.01)
%[tr,accuracies] = TrainMLP(Input,Output,200,0.1,0.1,0.8,2000,"traingdx","logsig","logsig","mse",0.8,0.01)
%[tr,accuracies] = TrainMLP(Input,Output,200,0.8,0.1,0.1,2000,"traingdx","logsig","logsig","mse",0.8,0.01)
%[tr,accuracies] = TrainMLP(Input,Output,200,0.4,0.2,0.4,2000,"traingdx","logsig","softmax","crossentropy",0.8,0.01)
%[tr,accuracies] = TrainMLP(Input,Output,200,0.1,0.1,0.8,2000,"traingdx","logsig","softmax","crossentropy",0.8,0.01)
%[tr,accuracies] = TrainMLP(Input,Output,200,0.8,0.1,0.1,2000,"traingdx","logsig","softmax","crossentropy",0.8,0.01)

%[tr,accuracies] = TrainMLP(Input,Output,500,0.4,0.2,0.4,2000,"traingdx","logsig","logsig","mse",0.8,0.01)
%[tr,accuracies] = TrainMLP(Input,Output,500,0.1,0.1,0.8,2000,"traingdx","logsig","logsig","mse",0.8,0.01)
%[tr,accuracies] = TrainMLP(Input,Output,500,0.8,0.1,0.1,2000,"traingdx","logsig","logsig","mse",0.8,0.01)
%[tr,accuracies] = TrainMLP(Input,Output,500,0.4,0.2,0.4,2000,"traingdx","logsig","softmax","crossentropy",0.8,0.01)
%[tr,accuracies] = TrainMLP(Input,Output,500,0.1,0.1,0.8,2000,"traingdx","logsig","softmax","crossentropy",0.8,0.01)
%[tr,accuracies] = TrainMLP(Input,Output,500,0.8,0.1,0.1,2000,"traingdx","logsig","softmax","crossentropy",0.8,0.01)

%% Point 3 of the Report
%[tr,accuracies] = TrainMLP(Input,Output,50,0.8,0.1,0.1,2000,"traingdm","logsig","softmax","crossentropy",0.8,0.01)
%[tr,accuracies] = TrainMLP(Input,Output,50,0.8,0.1,0.1,2000,"traingdx","logsig","softmax","crossentropy",0.8,0.01)
%[tr,accuracies] = TrainMLP(Input,Output,50,0.8,0.1,0.1,2000,"traincgf","logsig","softmax","crossentropy",0.8,0.01)

%[tr,accuracies] = TrainMLP(Input,Output,200,0.8,0.1,0.1,2000,"traingdm","logsig","softmax","crossentropy",0.8,0.01)
%[tr,accuracies] = TrainMLP(Input,Output,200,0.8,0.1,0.1,2000,"traingdx","logsig","softmax","crossentropy",0.8,0.01)
%[tr,accuracies] = TrainMLP(Input,Output,200,0.8,0.1,0.1,2000,"traincgf","logsig","softmax","crossentropy",0.8,0.01)

%[tr,accuracies] = TrainMLP(Input,Output,500,0.8,0.1,0.1,2000,"traingdm","logsig","softmax","crossentropy",0.8,0.01)
%[tr,accuracies] = TrainMLP(Input,Output,500,0.8,0.1,0.1,2000,"traingdx","logsig","softmax","crossentropy",0.8,0.01)
%[tr,accuracies] = TrainMLP(Input,Output,500,0.8,0.1,0.1,2000,"traincgf","logsig","softmax","crossentropy",0.8,0.01)

%% Point 4 of the Report
%[tr,accuracies] = TrainMLP(Input,Output,100,0.8,0.1,0.1,2000,"traingdm","logsig","softmax","crossentropy",0.8,0.01)
%[tr,accuracies] = TrainMLP(Input,Output,100,0.8,0.1,0.1,2000,"traingdm","logsig","softmax","crossentropy",0.9,0.01)
%[tr,accuracies] = TrainMLP(Input,Output,100,0.8,0.1,0.1,2000,"traingdm","logsig","softmax","crossentropy",0.8,0.1)
%[tr,accuracies] = TrainMLP(Input,Output,100,0.8,0.1,0.1,2000,"traingdm","logsig","softmax","crossentropy",0.9,0.1)

%[tr,accuracies] = TrainMLP(Input,Output,150,0.8,0.1,0.1,2000,"traingdm","logsig","softmax","crossentropy",0.8,0.01)
%[tr,accuracies] = TrainMLP(Input,Output,150,0.8,0.1,0.1,2000,"traingdm","logsig","softmax","crossentropy",0.9,0.01)
%[tr,accuracies] = TrainMLP(Input,Output,150,0.8,0.1,0.1,2000,"traingdm","logsig","softmax","crossentropy",0.8,0.1)
%[tr,accuracies] = TrainMLP(Input,Output,150,0.8,0.1,0.1,2000,"traingdm","logsig","softmax","crossentropy",0.9,0.1)

%[tr,accuracies] = TrainMLP(Input,Output,200,0.8,0.1,0.1,2000,"traingdm","logsig","softmax","crossentropy",0.8,0.01)
%[tr,accuracies] = TrainMLP(Input,Output,200,0.8,0.1,0.1,2000,"traingdm","logsig","softmax","crossentropy",0.9,0.01)
%[tr,accuracies] = TrainMLP(Input,Output,200,0.8,0.1,0.1,2000,"traingdm","logsig","softmax","crossentropy",0.8,0.1)
%[tr,accuracies] = TrainMLP(Input,Output,200,0.8,0.1,0.1,2000,"traingdm","logsig","softmax","crossentropy",0.9,0.1)

%[tr,accuracies] = TrainMLP(Input,Output,300,0.8,0.1,0.1,2000,"traingdm","logsig","softmax","crossentropy",0.8,0.01)
%[tr,accuracies] = TrainMLP(Input,Output,300,0.8,0.1,0.1,2000,"traingdm","logsig","softmax","crossentropy",0.9,0.01)
%[tr,accuracies] = TrainMLP(Input,Output,300,0.8,0.1,0.1,2000,"traingdm","logsig","softmax","crossentropy",0.8,0.1)
%[tr,accuracies] = TrainMLP(Input,Output,300,0.8,0.1,0.1,2000,"traingdm","logsig","softmax","crossentropy",0.9,0.1)

%% PLOT

figure()
plot(tr.epoch, tr.perf, tr.epoch, tr.vperf, tr.epoch, tr.tperf)
legend("train", "validation", "test")
title("Performance")
grid on
xlabel epochs

%% Method
function [tr,accuracies] = TrainMLP(input,output,hiddenUnits,trainRatio,valRatio,testRatio,epochs,trainFcn,transferFcn1,transferFcn2,performFcn,trainMomentum,trainRate)
    
    %% INICIALE NET
    
    net=feedforwardnet([hiddenUnits]);
    net.trainFcn = trainFcn%'traingdx';
    
    %% DATA SEPARATION
   
    net.divideParam.trainRatio = trainRatio; % Portion of data used as training set
    net.divideParam.valRatio = valRatio; % Portion of data data used as validation set
    net.divideParam.testRatio = testRatio; % Portion of data data used as test set
    
    %% DEFINE THE LAYERS
    
    
    % When the training functions is used the momentum and learning rate, you
    % need to set these parameters too.
    net.trainParam.mc = trainMomentum%0.8; % momentum parameter
    net.trainParam.lr = trainRate%0.01; % learning rate parameter
    
    net.trainParam.max_fail = 6; % validation check parameter
    net.trainParam.epochs = epochs; % number of epochs parameter
    net.trainParam.min_grad = 1e-5; % minimum performance gradient
    
    net.layers{1}.transferFcn = transferFcn1%'logsig';
    net.layers{2}.transferFcn = transferFcn2%'softmax';
    net.performFcn = performFcn%'crossentropy';
    
    net.inputs{1}.processFcns = {};
    net.outputs{end}.processFcns = {};
    
    %% TRAINING
    
    [net,tr,y,e] = train(net,input,output);
    performances = [tr.best_perf tr.best_vperf tr.best_tperf];
    
    %% VALIDATION
    
    [~, Yargmax] = max(y, [], 1);
    [~, Oargmax] = max(output, [], 1);
    accVector = Yargmax == Oargmax;
    accuracies = [sum(accVector(tr.trainInd)) / length(tr.trainInd), sum(accVector(tr.valInd)) / length(tr.valInd), sum(accVector(tr.testInd)) / length(tr.testInd)]
end