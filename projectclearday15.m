%for 15 minutes interval (clear day)

Generationdata=readtable('C:\Users\hp\Desktop\Generationdata.xlsx');
Irradiationdata=readtable('C:\Users\hp\Desktop\Irradiationdata.xlsx');

                                                         
TrainIp=table2array(Irradiationdata(25:2520,14:-2:8));                          % read data from workspace
TestIp=table2array(Generationdata(4:2499,14:-2:8));                            % read data from workspace

TestIp(isnan(TrainIp)) = [];                                                % remove NAN from DATA
TrainIp(isnan(TrainIp)) = [];                                               % remove NAN from DATA
TestIp(isnan(TestIp)) = [];                                                % remove NAN from DATA
TrainIp(isnan(TestIp)) = [];                                               % remove NAN from DATA
TrainIp(TestIp>50)=[];                                                      % remove noise (more than 50) from DATA
TestIp(TestIp>50)=[];                                                       % remove noise (more than 50) from DATA
TrainIp(TestIp<=0)=[];                                                       % remove noise (less than 0) from DATA
TestIp(TestIp<=0)=[];                                                        % remove noise (less than 0) from DATA

TestIp(TrainIp<=0)=[];                                                      % remove noise (less than 0) from DATA
TrainIp(TrainIp<=0)=[];                                                     % remove noise (less than 0) from DATA
                                                             
mn = min(TrainIp);                                                          % minimum of data
mx = max(TrainIp);                                                          % maximum of data
mn2 = min(TestIp);                                                          % minimum of data
mx2 = max(TestIp);                                                          % maximum of data

numTimeStepsTrain = numel(TrainIp); 
XTrainIp = (TrainIp - mn) / (mx-mn);                                            %Normlize the Data
XTestIp = (TestIp - mn2) / (mx2-mn2);                            
figure
plot(XTrainIp(1:50))
hold on
plot(XTestIp(1:50),'.-')
legend(["Input" "Target"])
ylabel("Irradiationdata/Generationdata")
xlabel("Time (15 minutes interval)")
title(" Unit Generation")

%data for clear day 1st April
YTrainIp =table2array(Irradiationdata(25:120,16));                             % testing input data points
YTestIp = table2array(Generationdata(4:99,16));                                % testing target data points

YTestIp(isnan(YTrainIp)) = [];                                                % remove NAN from DATA
YTrainIp(isnan(YTrainIp)) = [];                                               % remove NAN from DATA
YTestIp(isnan(YTestIp)) = [];                                                % remove NAN from DATA
YTrainIp(isnan(YTestIp)) = []; 
YTrainIp(YTestIp>50)=[];                                                      % remove noise (more than 50) from DATA
YTestIp(YTestIp>50)=[];                                                       % remove noise (more than 50) from DATA
YTrainIp(YTestIp<=0)=[];                                                       % remove noise (less than 0) from DATA
YTestIp(YTestIp<=0)=[];                                                        % remove noise (less than 0) from DATA

YTestIp(YTrainIp<=0)=[];                                                      % remove noise (less than 0) from DATA
YTrainIp(YTrainIp<=0)=[];                                                     % remove noise (less than 0) from DATA

YTrainIp=YTrainIp';                                                           % convert row vs column
YTestIp=YTestIp';    % convert row vs column
x=YTrainIp;
y=YTestIp;
mn3 = min(YTrainIp);                                                          % minimum of data
mx3= max(YTrainIp);                                                          % maximum of data
mn4 = min(YTestIp);                                                          % minimum of data
mx4 = max(YTestIp);                                                          % maximum of data

YTrainIp = (YTrainIp - mn3) / (mx3-mn3);                                            %Normlize the Data
YTestIp = (YTestIp - mn4) / (mx4-mn4);

numFeatures = 2;                                                            % number of inputs=2
numResponses = 1;                                                           % number of output=1
numHiddenUnits = 200;                                                       % number of hidden unites

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];                                                       % LSTM layer structure

options = trainingOptions('adam', ...
    'MaxEpochs',250, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'MiniBatchSize',50, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',90, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',false, ...
    'Plots','training-progress');                                           % LSTM other options

net = trainNetwork([XTrainIp(1:end-1);XTestIp(1:end-1)],XTestIp(2:end),layers,options); % LSTM training
net2=net;
net = predictAndUpdateState(net,[XTrainIp(1:end-1);XTestIp(1:end-1)]);
[net,YPred] = predictAndUpdateState(net,[XTrainIp(end-1);XTestIp(end-1)]);  % LSTM prediction and update the network of last element of training data

numTimeStepsTest = numel(YTestIp);
for i = 2:numTimeStepsTest                                                  % LSTM prediction and update the network of next element of testing data
    [net,YPred(:,i)] = predictAndUpdateState(net,[YTrainIp(i-1);YPred(:,i-1)],'ExecutionEnvironment','cpu');
end                                                                         % predicted value is taken as input for the network (loop)

YPred = (mx4-mn4)*YPred + mn4;                                              % denormlize the predicted data as per min and max of target
YTest = YTestIp(1:end);
YTest = (mx4-mn4)*YTest + mn4;                                              % target data

%different types of error of network
PR_rmse = sqrt((YPred-YTest).^2)./YTest*100
Percentage_rmse=mean(PR_rmse)
rmse = sqrt(mean((YPred-YTest).^2))
perf=mae(YPred,YTest)                                                     
perf2=mse(YPred,YTest)
PRerror=abs(YPred-YTest)./YTest;
MAPE= mean((abs(YPred-YTest))./YTest)

XTestIp2 = (mx2-mn2)*XTestIp + mn2;                                          % denormlize the input data as per min and max of input

figure
plot(XTestIp2(1:end-1))
hold on
idx = numTimeStepsTrain:(numTimeStepsTrain+numTimeStepsTest);
plot(idx,[XTestIp2(numTimeStepsTrain) YPred],'.-')
hold off
xlabel("Time (15 minutes interval)")
ylabel("KWh")
title("Forecast for clear day")
legend(["Observed" "Forecast"])

figure
subplot(2,1,1)
plot(YTest)
hold on
plot(YPred,'.-')
hold off
legend(["Observed" "Forecast"])
xlabel("Time (15 minutes interval)")
ylabel("KWh")
title("Forecast for clear day")

subplot(2,1,2)
stem(YPred - YTest)
xlabel("Time (15 minutes interval)")
ylabel("KWh")
title("RMSE = " + rmse + "  percentage rmse = " +Percentage_rmse+"%  MAE = " +perf+ "  MAPE = " +MAPE)

net = resetState(net);
net = predictAndUpdateState(net,[XTrainIp(1:end-1);XTestIp(1:end-1)]);      % train again
YPred = [];
numTimeStepsTest = numel(YTrainIp-1);
for i = 1:numTimeStepsTest                                                  % predict the output considerig new iputs in sequence
    [net,YPred(:,i)] = predictAndUpdateState(net,[YTrainIp(:,i);YTestIp(:,i)],'ExecutionEnvironment','cpu');
end
YPred = (mx4-mn4)*YPred + mn4;                                              % denormlize the predicted data as per min and max of target

%different types of error of network
PR_rmse = sqrt((YPred-YTest).^2)./YTest*100
Percentage_rmse=mean(PR_rmse)
rmse = sqrt(mean((YPred-YTest).^2))
perf=mae(YPred,YTest)                                                     
perf2=mse(YPred,YTest)
PRerror=abs(YPred-YTest)./YTest;
MAPE= mean((abs(YPred-YTest))./YTest)

figure
subplot(2,1,1)
plot(YTest)
hold on
plot(YPred,'.-')
hold off
legend(["Observed" "Predicted"])
xlabel("Time (15 minutes interval)")
ylabel("KWh")
title("Forecast with Updates for clear day")

subplot(2,1,2)
stem(YPred - YTest)
xlabel("Time (15 minutes interval)")
ylabel("Error")
title("RMSE = " + rmse + "  percentage rmse = " +Percentage_rmse+"%  MAE = " +perf+ "  MAPE = " +MAPE)

