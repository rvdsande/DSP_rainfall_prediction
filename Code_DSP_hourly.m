%Project: Statistical Digital Signal Processing
%Subject: Project Rainfall Estimation using ARMA models / Date: 6-10-2021
%Authors: Martijn Hubers, Robin van der Sande

clear all 
close all

%%%%%%%%%% Data analysis %%%%%%%%%%

%The rainfall data matrix T contains 4 collums:
%Collum 1: date 20110102-> 02-01-2011
%Collum 2: time of data in hours 13 -> 12:00 till 13:00
%Collum 3: cumulative rainfall in mm
%Collum 4: Air pressure in 0.1hpa with respect to seelevel
load('Raindata.mat')
rain = T(:,3); % rain in mm 

%Define the date and time as datetime value
for i = 1:length(rain)  
date = num2str(T(i,1))-'0';
day = date(8)+10*date(7);
month = date(6)+10*date(5);
year = date(4)+10*date(3)+100*date(2)+1000*date(1);
date_time(i) = datetime(year,month,day,T(i,2),0,0);
end 
date_time = date_time';
%Split the rainfall data into 3 sets and four seasons:
%   M -> the modeling set: 2011-2014
%   P -> the parameter set: 2015-2018
%   V -> the verification set: 2019-2020

%Winter set-> December, January, Febuary
idx = (((date_time.Month >= 12) | (date_time.Month <= 2)) & (date_time.Year <= 2014));
rain_winter_M = rain(idx);
idx = (((date_time.Month >= 12) | (date_time.Month <= 2)) & ((date_time.Year <= 2018) & (date_time.Year >= 2015)));
rain_winter_P = rain(idx);
idx = (((date_time.Month >= 12) | (date_time.Month <= 2)) & (date_time.Year >= 2019));
rain_winter_V = rain(idx);

%Spring set-> March, April, May
idx = (((date_time.Month >= 3) & (date_time.Month <= 5)) & (date_time.Year <= 2014));
rain_spring_M = rain(idx);
idx = (((date_time.Month >= 3) & (date_time.Month <= 5)) & ((date_time.Year <= 2018) & (date_time.Year >= 2015)));
rain_spring_P = rain(idx);
idx = (((date_time.Month >= 3) & (date_time.Month <= 5)) & (date_time.Year >= 2019));
rain_spring_V = rain(idx);

%Summer set-> June, July, August
idx = (((date_time.Month >= 6) & (date_time.Month <= 8)) & (date_time.Year <= 2014));
rain_summer_M = rain(idx);
idx = (((date_time.Month >= 5) & (date_time.Month <= 8)) & ((date_time.Year <= 2018) & (date_time.Year >= 2015)));
rain_summer_P = rain(idx);
idx = (((date_time.Month >= 5) & (date_time.Month <= 8)) & (date_time.Year >= 2019));
rain_summer_V = rain(idx);

%Autumn set-> Septemeber, October, November
idx = (date_time.Month >= 9) & (date_time.Month <= 11);
idx = (((date_time.Month >= 9) & (date_time.Month <= 11)) & (date_time.Year <= 2014));
rain_autumn_M = rain(idx);
idx = (((date_time.Month >= 9) & (date_time.Month <= 11)) & ((date_time.Year <= 2018) & (date_time.Year >= 2015)));
rain_autumn_P = rain(idx);
idx = (((date_time.Month >= 9) & (date_time.Month <= 11)) & (date_time.Year >= 2019));
rain_autumn_V = rain(idx);

%The complete modeling set 
idx = (date_time.Year <= 2014);
rain_M = rain(idx);

%The complete verification set 
idx = (date_time.Year >= 2019);
rain_V = rain(idx);
date_time_V = date_time(idx);

%Plot the data for the entire data set
figure(1)
subplot(2,1,1)
plot(date_time(1416:3623),rain(1416:3623))
title('Rainfall during spring 2011')
xlabel('time in [h]')
ylabel('hourly rainfall in [mm]')
ylim([0,12])
subplot(2,1,2)
plot(date_time(3624:5831),rain(3624:5831))
title('Rainfall during summer 2011')
xlabel('time in [h]')
ylabel('hourly rainfall in [mm]')
ylim([0,12])

%%%%%%%%%% Data processing %%%%%%%%%%
% Analyse the rainfall data of the modeling set
figure(2)
subplot(2,1,1)
plot(rain_M)
title('Rainfall over time')
xlabel('Time in [h]')
ylabel('Hourly rainfall in [mm]')
subplot(2,1,2)
hist(rain_M,30)
title('Histogram of rainfall')
xlabel('Hourly rainfall in [mm]')

mean_rain_M = mean(rain_M);
std_rain_M = std(rain_M);

% Perform stationarity tests
adftest(rain_M)
kpsstest(rain_M)

% Analyse the first order difference of the rainfall date of the modeling set
figure(3);
rain_M_diff = diff(rain_M);
subplot(2,1,1)
plot(rain_M_diff)
title('Rainfall over time')
xlabel('Time in [h]')
ylabel('Hourly rainfall in [mm]')
subplot(2,1,2)
hist(rain_M_diff,30)
title('Histogram of rainfall')
xlabel('Hourly rainfall in [mm]')
mean_x2 = mean(rain_M_diff);
std_x2 = std(rain_M_diff);

% Perform stationarity tests
adftest(rain_M_diff)
kpsstest(rain_M_diff)

%%%%%%%%%% Model identification %%%%%%%%%%
%plot the autocorrelation and partial autocorrelation function
figure(4)
autocorr(rain_M_diff)
figure(5)
parcorr(rain_M_diff)

%Determine the ARMA model order using AIC
pMax = 10;
qMax = 5;
[AIC, P, Q, minAIC] = calculate_min_aic(pMax, qMax, x2);
P = 9;
Q = 3;

%[P2,Q2] = ARMA_Order_Select(x1,pMax,qMax,1);

save('Raindata2.mat')

function [AIC, P, Q, minAIC] = calculate_min_aic(pMax, qMax, data)

AIC = ones(pMax+1, qMax+1)*10^5; % *10^5 makes sure that aic(1,1) is not selected as minimum

for p=0:pMax
    for q=0:qMax
        if p+q ~= 0 %% Can't calculate for p=q=0
            model = arima(p,0,q); % Create ARMA(p,0,q) model 
            [~, ~, logL] = estimate(model, data, 'Display', 'off'); % Calculate loglikelihood from the ARMA(p,0,q) model fitted with the data
            numParam = p+q;
            AIC(p+1,q+1) = aicbic(logL, numParam); % Calculate AIC
        end        
    end
end

% AIC is a matrix where the row number - 1 is the AR (p) coefficient number, and
% the column number - 1 the MA (q) coefficient number

% Calculate the minimum AIC
[min_aic1, AR_nums_aic] = min(AIC,[],1); % AR_nums_aic is array that contains the AR order (p) at the lowest AIC for each MA coefficient q=1,..,q_max 
[min_aic, MA_num_aic] = min(min_aic1, [], 2); % MA_num_aic gives the MA coefficient order (q) at the lowest AIC
AR_num_aic = AR_nums_aic(MA_num_aic); % AR_nums_aic gives the AR coefficient order (p) at the lowest AIC

% Corresponding values for p and q
P = AR_num_aic-1;
Q = MA_num_aic-1;

% Minimum aic
minAIC = min_aic;
end

function AIC = calculate_aic(p, q, data)
    model = arima(p,0,q); % Create ARMA(p,0,q) model 
    [~, ~, logL] = estimate(model, data, 'Display', 'off'); % Calculate loglikelihood from the ARMA(p,0,q) model with data
    numParam = p+q;
    AIC = aicbic(logL, numParam);
end

%%%%%% Parameter esitmation %%%%%%
load('Raindata2.mat')
P = 9;
Q = 3;

for i = 1:4
    if (i ==1)
    x3 = rain_winter_P;
    x4 = diff(rain_winter_P);
    elseif(i ==2)
    x3 = rain_spring_P;
    x4 = diff(rain_spring_P);
    elseif(i ==3)
    x3 = rain_summer_P;
    x4 = diff(rain_summer_P);
    elseif(i ==4)
    x3 = rain_autumn_P;
    x4 = diff(rain_autumn_P);
    end 
%First fit a AR(P) model uing the autocorrelation method
[a_p,err] = ARfit(x3,P);

model_1(i) = arima(P,0,0);
model_1(i).AR =  num2cell(a_p);
model_1(i).Variance = var(x4);
model_1(i).Constant = 4e-6;

[res,~,loglike] = infer(model_1(i),x4);
stdres = res/sqrt(model_1(i).Variance);

if(i == 1)
figure(6)
subplot(2,2,1)
hist(stdres)
title('Residual plot of the AR(P) model')
subplot(2,2,3)
qqplot(stdres)
subplot(2,2,2)
parcorr(stdres)
subplot(2,2,4)
parcorr(stdres)
end

%Second fit a ARMA(P,Q) model using the estimate fucntion
model_2(i) = arima(P,1,Q);
model_2(i) = estimate(model_2(i),x3);

[res,~,loglike] = infer(model_2(i),x3);
stdres = res/sqrt(model_2(i).Variance);

if(i == 1)
figure(7)
subplot(2,2,1)
hist(stdres)
title('Residual plot of the ARMA(P,Q) model')
subplot(2,2,3)
qqplot(stdres)
subplot(2,2,2)
parcorr(stdres)
subplot(2,2,4)
parcorr(stdres)
end
end

save('Raindata3.mat')

function X = convmatrix(x,p)
%Calulate the convolution matrix X of size p for a given vector x  
N    = length(x)+2*p-2; 
x    = x(:);                            %Make x into a row vector
xpad = [zeros(p-1,1);x;zeros(p-1,1)];   %Make the extende x vector
for  i=1:p
X(:,i)=xpad(p-i+1:N-i+1);               %Shift and create colums of X
end
end

function [a,err] = ARfit(x,p)
x   = x(:);                             %Make x into a colum vector
N   = length(x);                        %Find the lenght of the data vector
if p>=length(x)                         %Check if data is long enough for the AR model fit
     error('Model order too large')
end
X   = convmatrix(x,p+1);                %Find the data matrix X
Xp  = X(1:N+p-1,1:p);                   %make it a [N+p]*[p] matrix
x1 = [x(2:end);zeros(p,1)];
a   = [1;inv(Xp'*Xp)*Xp'*x1];             %Find the AR coeficients using the 
X   = convmatrix(x,p+1);                %Find the data matrix X
err = abs(X(1:N+p,1)'*X*a);
a = a(2:end);
end

%%%%%% Rainfall preciction %%%%%%

















