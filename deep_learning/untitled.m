clear
close all
clc

load train.mat

%% 1a
for i=1:10
    subplot(2,5,i)
    imshow(XTrain(:,:,1,i))
    title(YTrain(i))
end

%% 1b
cut = length(YTrain)*0.3;
X_val = XTrain(:,:,1,1:cut);
Y_val = YTrain(1:cut);
X_train = XTrain(:,:,1,cut+1:end);
Y_train = YTrain(cut+1:end);

options = trainingOptions('adam', ...
'MaxEpochs',5, ...
'InitialLearnRate',1e-4, ...
'ValidationData', {X_val,Y_val},'Plots','training-progress','Verbose',true);
%% 2
sizes = size(X_train);
layers=[
imageInputLayer([sizes(1) sizes(2) sizes(3)],'Name','input')
convolution2dLayer(3,6,'Padding','same')
reluLayer
batchNormalizationLayer
maxPooling2dLayer(2,'Stride',2)
convolution2dLayer(3,16)
reluLayer
batchNormalizationLayer
maxPooling2dLayer(2,'Stride',2)
fullyConnectedLayer(120,'name','f1')
reluLayer
fullyConnectedLayer(84,'name','f2')
reluLayer
fullyConnectedLayer(10,'name','f3')
softmaxLayer
classificationLayer];

model = trainNetwork(X_train,Y_train,layers,options);




