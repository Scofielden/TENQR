%% Introduction
% Description: Risk spillovers among oil, gold, stock, and foreign exchange markets: Evidence from G20 economies.
% To be published in: North American Journal of Economics and Finance
% Our code is based on the paper: 
% Feng, Y., Wang, G.J., Zhu, Y., Xie, C., 2023. Systemic risk spillovers and the determinants in the stock markets of the Belt and Road countries. Emerging Markets Review 55, 101020.

%% Step1: Calculate the CoES of market return

clc, clear
% Set the path and load data
datapath = 'D:\TENQR\';
codepath = 'D:\TENQR\';
MarketReturn = xlsread([datapath, 'Comm G20 Stock and Forex Return.xlsx']);

% len - Number of sample markets' observations
% N   - Number of sample markets
[len,N]=size(MarketReturn);   

% Parameter setting of rolling time window
WindowSize = 52;    
WindowStep = 01;   
WindowNum = floor((len-WindowSize)/WindowStep+1);   
tau = 0.05; % Consider the risk measure at the 5% quantile level

VaR_005 = zeros(WindowSize,N,WindowNum);
CoES_005 = zeros(N,N,WindowNum);

% Adopt the CAViaR method to estimate the VaR and calculate the CoES by definition

addpath(genpath([codepath])) 
for w = 1:WindowNum
    disp(w)
    tic
    t1 = (w-1)*WindowStep+1;
    t2 = (w-1)*WindowStep+WindowSize;
    Return = MarketReturn(t1:t2,:);
    % Estimate the VaR by using the CAViaR method
    [VaR_005(:,:,w), Table_Coeff_005] = CAViaROptimisation_gjw_final(2, tau, Return); % The dimension of the VaR is WindowSize×N
    for i = 1:N
        % Calculate the CoES by definition
        for j = 1:N
            Return_cond_value = 0;
            Return_cond_number = 0;
            for t = 1:WindowSize
            	if Return(t,j) <= -VaR_005(t,j,w)
                    Return_cond_value = Return_cond_value + Return(t,i);
                    Return_cond_number = Return_cond_number + 1;
            	end
            end
            if Return_cond_number > 0
                CoES_005(i,j,w)= Return_cond_value / Return_cond_number;
            elseif Return_cond_number == 0
                CoES_005(i,j,w) = 0;
            end
        end
    end
    % Save the result 
    save([datapath, 'VaR_CAViaR_Estimation_005.mat'], 'VaR_005')
    save([datapath, 'CoES_definition1_005.mat'], 'CoES_005')
    toc
end


%% Step2: Construct the similarity matrix

load([datapath, 'VaR_CAViaR_Estimation_005.mat'], 'VaR_005')
load([datapath, 'CoES_definition1_005.mat'], 'CoES_005')

standCoES_005 = zeros(N,N,WindowNum);
for w = 1:WindowNum
    for i = 1:N
        standCoES_005(i,:,w) = zscore(CoES_005(i,:,w));
    end
end

% Compute the similarity matrix based on the similarity of the risk profiles (CoES)
S = zeros(N,N,WindowNum);  % Similarity matrix (N×N) of WindowNum time Windows 
Connectedness_Sim = zeros(WindowNum,1); % The total connectedness of the similarity matrix
Connectedness_Dec = zeros(WindowNum,N); % The individual connectedness of the similarity matrix


for w = 1:WindowNum
    for i = 1:N
        for j = (i+1):N
            %  Construct similarity matrices based on cosine similarity
            S(i,j,w) = (dot(standCoES_005(i,:,w),standCoES_005(j,:,w))/(norm(standCoES_005(i,:,w))*norm(standCoES_005(j,:,w))));
            S(j,i,w) = S(i,j,w);
        end 
        Connectedness_Dec(w,i) = sum(S(i,:,w));
    end
    Connectedness_Sim(w,1) = sum(sum(S(:,:,w)));
end

% % Save the result 
xlswrite([datapath, 'NetworkConnectedness.xlsx'], Connectedness_Sim, 'Connectedness', 'A');
xlswrite([datapath, 'NetworkConnectedness_Dec.xlsx'], Connectedness_Dec, 'Connectedness', 'A');

% Present the similarity matrix by means of a heat map
country_name = {'WTI','Gold',...
                'ARG','AUS','BRA','CAN','CHN','DEU','EU','FRA','GBR','IDN','IND','ITA','JPN','KOR','MEX','RUS','SAU','TUR','USA','ZAF',...
                'ARS','AUD','BRL','CAD','CNY','EUR','GBP','IDR','INR','JPY','KRW','MXN','RUB','SAR','TRY','USD','ZAR'};

year = {'2008' '2009' '2010' '2011' '2012' '2013' '2014' '2015',...
        '2016' '2017' '2018' '2019' '2020' '2021' '2022' '2023'};
    
ind = [52   104   156   208   260   312   364   416   468   520   572   624   676   728   780   829];

sub1 = figure; figure(sub1) % 2008-2011
for iyear = 1:4
    subplot(2,2,iyear)
    clims = [-1 1]; 
    imagesc(S(:,:,ind(iyear)),clims) 
    colorbar
    title(year(iyear), 'FontSize', 9);
    axis image
    set(sub1,'PaperPosition',[0 0, 24 24]) 
    set(sub1, 'PaperSize', [24 24]);
    set(gca, 'YTickLabel',country_name,...
    'YTick',[1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39],...
    'FontSize',5,'FontAngle','Normal','FontName','Times New Roman',...
    'XTickLabel',country_name,....
    'XTick',[1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39],...
    'XTickLabelRotation',90,...
    'FontSize',5,'FontAngle','Normal','FontName','Times New Roman')
end
saveas(sub1, [datapath, 'figure1.jpg'])

sub2 = figure; figure(sub2) % 2012-2015
for iyear = 5:8
    subplot(2,2,iyear-4)
    clims = [-1 1]; 
    imagesc(S(:,:,ind(iyear)),clims) 
    colorbar
    title(year(iyear), 'FontSize', 9);
    axis image
    set(sub2,'PaperPosition',[0 0, 24 24]) 
    set(sub2, 'PaperSize', [24 24]);
    set(gca, 'YTickLabel',country_name,...
    'YTick',[1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39],...
    'FontSize',5,'FontAngle','Normal','FontName','Times New Roman',...
    'XTickLabel',country_name,....
    'XTick',[1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39],...
    'XTickLabelRotation',90,...
    'FontSize',5,'FontAngle','Normal','FontName','Times New Roman')
end
saveas(sub2, [datapath, 'figure2.jpg'])

sub3 = figure; figure(sub3) % 2016-2019
for iyear = 9:12
    subplot(2,2,iyear-8)
    clims = [-1 1]; 
    imagesc(S(:,:,ind(iyear)),clims) 
    colorbar
    title(year(iyear), 'FontSize', 9);
    axis image
    set(sub3,'PaperPosition',[0 0, 24 24]) 
    set(sub3, 'PaperSize', [24 24]);
    set(gca, 'YTickLabel',country_name,...
    'YTick',[1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39],...
    'FontSize',5,'FontAngle','Normal','FontName','Times New Roman',...
    'XTickLabel',country_name,.... 
    'XTick',[1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39],...
    'XTickLabelRotation',90,...
    'FontSize',5,'FontAngle','Normal','FontName','Times New Roman')
end
saveas(sub3, [datapath, 'figure3.jpg'])

sub4 = figure; figure(sub4) % 2020-2023
for iyear = 13:16
    subplot(2,2,iyear-12)
    clims = [-1 1]; 
    imagesc(S(:,:,ind(iyear)),clims) 
    colorbar
    title(year(iyear), 'FontSize', 9);
    axis image
    set(sub4,'PaperPosition',[0 0, 24 24]) 
    set(sub4, 'PaperSize', [24 24]);
    set(gca, 'YTickLabel',country_name,...
    'YTick',[1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39],...
    'FontSize',5,'FontAngle','Normal','FontName','Times New Roman',...
    'XTickLabel',country_name,....
    'XTick',[1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39],...
    'XTickLabelRotation',90,...
    'FontSize',5,'FontAngle','Normal','FontName','Times New Roman')
end
saveas(sub4, [datapath, 'figure4.jpg'])

%% Step3: Use asymmetric breakpoint method to transform the similarity matrix into adjacency matrix (tail risk network)

A = zeros(N,N,WindowNum);                   % Adjacency matrix 
Connectedness_Adj = zeros(WindowNum,1);  
% Connectedness_Adj(1:12,1) = NaN;
Negative_ratio = zeros(WindowNum,1);     % Negative correlation ratio (n2/n)
% Negative_ratio(1:12,1) = NaN;
statistics = zeros(WindowNum,6);            % Statistics of the Joint SVR Test [SVR_Lpos, SVR_Spos, SVR_Lneg, SVR_Sneg, Sta_Large, Sta_Small]
BreakPoint = zeros(WindowNum, 2 + 7+7);     % BreakPoint

% Calculate the adjacency matrix based on an asymmetric breakpoint method
for w = 1:WindowNum
    P_positive = []; P_negative = []; % Positive and negative correlation coefficients
    N1 = 0; N2 = 0;
    for i = 1:N
        for j = (i+1):N
            if S(i,j,w)>=0      
                P_positive = [P_positive; S(i,j,w)];
            elseif S(i,j,w)<0
                P_negative = [P_negative; S(i,j,w)];
            end
        end
    end
    
    % Ordered sequences of positive and negative correlation coefficients (in ascending order)
    P_pos_sorted = sort(P_positive);
    P_neg_sorted = sort(P_negative);
    
    % The number of sequence elements
    NN = N*(N-1)/2;
    Pnum_posivive = length(P_pos_sorted);
    Pnum_negative = length(P_neg_sorted);
    Negative_ratio(w,1) = Pnum_negative/NN;
    
    % Introduce the cumulative distribution function of the standard normal distribution
    sigma_positive = normcdf(sqrt(N)*P_pos_sorted);
    sigma_negative = normcdf(sqrt(N)*P_neg_sorted);
    
    % Difference sequence
    Dsigma_positive = sigma_positive(2:Pnum_posivive) - sigma_positive(1:(Pnum_posivive-1));
    Dsigma_negative = sigma_negative(2:Pnum_negative) - sigma_negative(1:(Pnum_negative-1));
    
    % asymmetric breakpoint method (Ng, 2006; Chen et al., 2019)
    if Pnum_negative > 1
        N1 = splitreg(Dsigma_negative,ones(Pnum_negative-1,1),0.1);
        Limit_N1 =  P_neg_sorted(N1);
    elseif Pnum_negative == 1
        N1 = 1;
        Limit_N1 =  P_neg_sorted(N1);
    elseif Pnum_negative == 0
        Limit_N1 = -1;
    end
    if Pnum_posivive > 1
        N2 = splitreg(Dsigma_positive,ones(Pnum_posivive-1,1),0.1); 
        Limit_N2 =  P_pos_sorted(N2);
    elseif Pnum_posivive == 1 
        N2 = 1;
        Limit_N2 =  P_pos_sorted(N2);
    elseif Pnum_posivive == 0 
        Limit_N2 =  1;
    end
    BreakPoint(w,1) = N1;
    BreakPoint(w,2) = N2;
    % The statistic of Variance Ratio Test (Cochrane, 1988)
    Large_positive = []; Small_positive = [];
    Large_negative = []; Small_negative = [];
    if N2>2
        Large_positive = P_pos_sorted((BreakPoint(w,2)+1):Pnum_posivive, 1);
        Small_positive = P_pos_sorted((1:BreakPoint(w,2)), 1);
        statistics(w,1) = SVR_Calculation(Large_positive,N);
        statistics(w,2) = SVR_Calculation(Small_positive,N);
    else  
        statistics(w,1) = NaN;
        statistics(w,2) = NaN;
    end
    if N1>2
        Large_negative = P_neg_sorted((1:(BreakPoint(w,1)-1)), 1);
        Small_negative = P_neg_sorted((BreakPoint(w,1):Pnum_negative), 1);
        statistics(w,3) = SVR_Calculation(Large_negative,N);
        statistics(w,4) = SVR_Calculation(Small_negative,N);
    else
        statistics(w,3) = NaN;
        statistics(w,4) = NaN;
    end
    statistics(w,5) = length(Large_positive) * statistics(w,1)^2 + length(Large_negative) * statistics(w,3)^2;
    statistics(w,6) = length(Small_positive) * statistics(w,2)^2 + length(Small_negative) * statistics(w,4)^2;
    % Construct the adjacency matrix according to the breakpoint
    for i = 1:N
        for j = (i+1):N
            if S(i,j,w) < Limit_N1
                A(i,j,w) = -1;
            elseif S(i,j,w) > Limit_N2
                A(i,j,w) = 1;
            else
                A(i,j,w) = 0;
            end
            A(j,i,w)=A(i,j,w);
        end
    end
    Connectedness_Adj(w,1) = sum(sum(A(:,:,w)));
end

% plot Joint SVR Test for Small and Large Groups
Joint_test = xlsread([datapath, 'Joint SVR test result.xlsx'], 'Sheet1', 'B:D');
% 创建一个新的图形窗口
figure(1)
set(gcf, 'Position', [0, 0, 800, 700]);
% 绘制第一个子图
subplot(2,1,1); % 2x1 网格中的第一个子图
scatter(Joint_test(:,1), Joint_test(:,2), 16, 'filled'); % 绘制散点图
title('Joint SVR Test for Small Groups', 'FontSize', 16); % 设置子图标题
xlabel('Year', 'FontSize', 12); % x 轴标签
ylabel('p-value', 'FontSize', 12); % y 轴标签
xlim([2008 2023]); % 设置 x 轴范围为 2008 到 2023
ylim([0 1]); % 设置 y 轴范围为 0 到 1
set(gca,'YTick',[0 0.5 1], 'FontSize',12,...
        'XTick',[2009 2011 2013 2015 2017 2019 2021 2023], 'FontSize',12)
% 绘制第二个子图
subplot(2,1,2); % 2x1 网格中的第二个子图
scatter(Joint_test(:,1), Joint_test(:,3), 16, 'filled'); % 绘制散点图
title('Joint SVR Test for Large Groups', 'FontSize', 16); % 设置子图标题
xlabel('Year', 'FontSize', 12); % x 轴标签
ylabel('p-value', 'FontSize', 12); % y 轴标签
xlim([2008 2023]); % 设置 x 轴范围
ylim([0 1]); % 设置 y 轴
set(gca,'YTick',[0 0.5 1], 'FontSize',12,...
        'XTick',[2009 2011 2013 2015 2017 2019 2021 2023], 'FontSize',12)


% Save the result 
save([datapath, 'adjacency matrix .mat'], 'A')
save([datapath, 'statistics.mat'], 'statistics')
xlswrite([datapath, 'Negative_ratio.xlsx'], Negative_ratio, 'Neg_ratio', 'A');

% Average adjacency matrix over the sample period
% The adjacency matrix is decomposed into positive and negative ones
Adj_positive = A; Adj_negative = A;  
Adj_positive(find(Adj_positive(:,:,:)==(-1))) = 0;  
Adj_negative(find(Adj_negative(:,:,:)==(1))) = 0;
Adj_posavg = zeros(N,N); Adj_negavg = zeros(N,N);

for i = 1:N
    for j = (i+1):N
        Adj_posavg(i,j) = mean(Adj_positive(i,j,:));
        Adj_negavg(i,j) = mean(Adj_negative(i,j,:));
        Adj_posavg(j,i) = Adj_posavg(i,j);
        Adj_negavg(j,i) = Adj_negavg(i,j);
    end
    Adj_posavg(i,i) = 0;
    Adj_negavg(i,i) = 0;
end

xlswrite([datapath, 'Adj_posavg.xlsx'], Adj_posavg, 'Adj_posavg', 'A');
xlswrite([datapath, 'Adj_negavg.xlsx'], Adj_negavg, 'Adj_negavg', 'A');

% Present the adjacency matrix and the average ones by means of heat maps
country_name = {'WTI','Gold',...
                'ARG','AUS','BRA','CAN','CHN','DEU','EU','FRA','GBR','IDN','IND','ITA','JPN','KOR','MEX','RUS','SAU','TUR','USA','ZAF',...
                'ARS','AUD','BRL','CAD','CNY','EUR','GBP','IDR','INR','JPY','KRW','MXN','RUB','SAR','TRY','USD','ZAR'};
            
year = {'2008' '2009' '2010' '2011' '2012' '2013' '2014' '2015',...
        '2016' '2017' '2018' '2019' '2020' '2021' '2022' '2023'};
    
ind = [43   104   156   208   260   312   364   416   468   520   572   624   676   728   780   829];
            
sub5 = figure;
figure(sub5)    % 2008-2011
for iyear = 1:4
    subplot(2,2,iyear)
    clims = [-1 1]; 
    imagesc(A(:,:,ind(iyear)),clims)
    colormap(gray)
    colorbar
    title(year(iyear), 'FontSize', 9);
    axis image
    set(sub5,'PaperPosition',[0 0, 24 24]) 
    set(sub5, 'PaperSize', [24 24]);
    set(gca, 'YTickLabel',country_name,...
        'YTick',[1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39],...
        'FontSize',5,'FontAngle','Normal','FontName','Times New Roman',...
        'XTickLabel',country_name,....
        'XTick',[1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39],...
        'XTickLabelRotation',90,...
        'FontSize',5,'FontAngle','Normal','FontName','Times New Roman')
end
saveas(sub5, [datapath, 'figure5.jpg'])

sub6 = figure;
figure(sub6)    % 2012-2015
for iyear = 5:8
    subplot(2,2,iyear-4)
    clims = [-1 1]; 
    imagesc(A(:,:,ind(iyear)),clims) 
    colormap(gray)
    colorbar
    title(year(iyear), 'FontSize', 9);
    axis image
    set(sub6,'PaperPosition',[0 0, 24 24]) 
    set(sub6, 'PaperSize', [24 24]);
    set(gca, 'YTickLabel',country_name,...
        'YTick',[1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39],...
        'FontSize',5,'FontAngle','Normal','FontName','Times New Roman',...
        'XTickLabel',country_name,....
        'XTick',[1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39],...
        'XTickLabelRotation',90,...
        'FontSize',5,'FontAngle','Normal','FontName','Times New Roman')
end
saveas(sub6, [datapath, 'figure6.jpg'])

sub7 = figure;
figure(sub7)    % 2016-2019
for iyear = 9:12
    subplot(2,2,iyear-8)
    clims = [-1 1];
    imagesc(A(:,:,ind(iyear)),clims)
    colormap(gray)
    colorbar
    title(year(iyear), 'FontSize', 9);
    axis image
    set(sub7,'PaperPosition',[0 0, 24 24]) 
    set(sub7, 'PaperSize', [24 24]);
    set(gca, 'YTickLabel',country_name,...
        'YTick',[1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39],...
        'FontSize',5,'FontAngle','Normal','FontName','Times New Roman',...
        'XTickLabel',country_name,....
        'XTick',[1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39],...
        'XTickLabelRotation',90,...
        'FontSize',5,'FontAngle','Normal','FontName','Times New Roman')
end
saveas(sub7, [datapath, 'figure7.jpg'])

sub8 = figure;
figure(sub8)    % 2020-2023
for iyear = 13:16
    subplot(2,2,iyear-12)
    clims = [-1 1];
    imagesc(A(:,:,ind(iyear)),clims)
    colormap(gray)
    colorbar
    title(year(iyear), 'FontSize', 9);
    axis image
    set(sub8,'PaperPosition',[0 0, 24 24]) 
    set(sub8, 'PaperSize', [24 24]);
    set(gca, 'YTickLabel',country_name,...
        'YTick',[1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39],...
        'FontSize',5,'FontAngle','Normal','FontName','Times New Roman',...
        'XTickLabel',country_name,....
        'XTick',[1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39],...
        'XTickLabelRotation',90,...
        'FontSize',5,'FontAngle','Normal','FontName','Times New Roman')
end
saveas(sub8, [datapath, 'figure8.jpg'])


%% Step4: Construct the systemic risk score index and individual risk contribution index

MC = reshape(VaR_005(end,:,:), N, WindowNum)';
Systemic_RiskScore = zeros(WindowNum,1); 
Systemic_RiskDecomposition = zeros(WindowNum,N);
for w = 1:WindowNum
    Systemic_RiskScore(w,1) = MC(w,:) * A(:,:,w) * MC(w,:)';
    for i = 1:N
        Systemic_RiskDecomposition(w,i) = MC(w,i) * A(i,:,w) * MC(w,:)';
    end
end

newRows = floor((WindowNum) / WindowSize) + 1;
% 创建一个新的数组来存放求和结果
Systemic_RiskDecomp_avg = zeros(N, newRows);
for i = 1:WindowSize:WindowNum
    % 获取当前区域的行的范围
    rowRange = i:min(i+WindowSize-1, WindowNum);
    disp(i);disp(min(i+WindowSize-1, WindowNum));
    % 获取当前区域的子矩阵并对其进行求和
    chunk = Systemic_RiskDecomposition(rowRange, :);
    Systemic_RiskDecomp_avg(:,(i+WindowSize-1)/WindowSize) = sum(chunk, 1) / size(chunk,1);
end

% Save the result 
xlswrite([datapath, 'Systemic_RiskScore.xlsx'], Systemic_RiskScore, 'RiskScore', 'A');
xlswrite([datapath, 'Systemic_RiskDecomposition.xlsx'], Systemic_RiskDecomposition, 'RiskDecomposition', 'A');
xlswrite([datapath, 'Systemic_RiskDecomp_avg.xlsx'], Systemic_RiskDecomp_avg, 'Systemic_RiskDecomp_avg', 'A');

% Displays the dynamic evolution of the total connectedness, negative correlation ratio and systemic risk score
figure(1)
set(gcf, 'Position', [0, 0, 800, 800]);
Xa = 1:(WindowNum);
subplot(3,1,1)
% Description of important financial events
% ① Window No.30-80: The 2008-2009 global financial crisis
fill([30 80 80 30],[1 1 999 999],[0.90 0.90 0.90],'EdgeColor','none');
hold on
% ② Window No.117-210: The 2010-2012 European sovereign debt crisis
fill([117 210 210 117],[1 1 999 999],[0.90 0.90 0.90],'EdgeColor','none');
hold on
% ③ Window No.355-470: The 2015-2016 Commodity prices collapse and Brexit vote
fill([355 470 470 355],[1 1 999 999],[0.90 0.90 0.90],'EdgeColor','none');
hold on
% ④ Window No.520-560: The 2018 Interest Increase by Federal Reserve 
fill([520 560 560 520],[1 1 999 999],[0.90 0.90 0.90],'EdgeColor','none');
hold on
% ⑤ Window No.620-675: The 2020-2021 global public health emergency
[H2] = fill([620 675 675 620],[1 1 999 999],[0.90 0.90 0.90],'EdgeColor','none');
hold on
% ⑥ Window No.730-770: The 2022 Russia-Ukraine War
[H2] = fill([730 770 770 730],[1 1 999 999],[0.90 0.90 0.90],'EdgeColor','none');
hold on

% Total connectedness (TC)
plot(Xa,Connectedness_Sim,'color',[0.00,0.45,0.74], 'LineWidth',1)
set(gca,'XTickLabel',{'2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019','2020','2021','2022','2023'},...
    'XTick',1:52:829,...
     'FontSize',8,'FontAngle','Normal','FontName','Times New Roman')
% xlabel('Time','FontName','Times New Roman','FontSize',10)
title('Total connectedness (TC)','FontName','Times New Roman','FontSize',10)

% Negative Correlation Ratio (n2/n)
subplot(3,1,2)
plot(Xa, Negative_ratio,'LineWidth',1)
set(gca,'XTickLabel',{'2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019','2020','2021','2022','2023'},...
    'XTick',1:52:829,...
     'FontSize',8,'FontAngle','Normal','FontName','Times New Roman')
title('Negative Correlation Ratio (n2/n)','FontName','Times New Roman','FontSize',10)

% Systemic Risk Score
subplot(3,1,3)
% Description of important financial events
% ① Window No.30-80: The 2008-2009 global financial crisis
fill([30 80 80 30],[-1.8 -1.8 39.5 39.5],[0.90 0.90 0.90],'EdgeColor','none');
hold on
% ② Window No.117-210: The 2010-2012 European sovereign debt crisis
fill([117 210 210 117],[-1.8 -1.8 39.5 39.5],[0.90 0.90 0.90],'EdgeColor','none');
hold on
% ③ Window No.355-470: The 2015-2016 Commodity prices collapse and Brexit vote
fill([355 470 470 355],[-1.8 -1.8 39.5 39.5],[0.90 0.90 0.90],'EdgeColor','none');
hold on
% ④ Window No.520-560: The 2018 Interest Increase by Federal Reserve 
fill([520 560 560 520],[-1.8 -1.8 39.5 39.5],[0.90 0.90 0.90],'EdgeColor','none');
hold on
% ⑤ Window No.620-675: The 2020-2021 global public health emergency
[H2] = fill([620 675 675 620],[-1.8 -1.8 39.5 39.5],[0.90 0.90 0.90],'EdgeColor','none');
hold on
% ⑥ Window No.730-770: The 2022 Russia-Ukraine War
[H2] = fill([730 770 770 730],[-1.8 -1.8 39.5 39.5],[0.90 0.90 0.90],'EdgeColor','none');
hold on
plot(Xa, Systemic_RiskScore,'color',[0.00,0.45,0.74], 'LineWidth',1)
set(gca,'XTickLabel',{'2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019','2020','2021','2022','2023'},...
    'XTick',1:52:829, 'YLim', [-2, 40],...
     'FontSize',8,'FontAngle','Normal','FontName','Times New Roman')
title('Systemic Risk Score','FontName','Times New Roman','FontSize',10)

%% Step5: Adopt the TRNQR model (Chen et al., 2019) to analyze the network factors dynamic

% Load the macro state variables
MarketCovariates = xlsread([datapath, 'market-wide covariates.xlsx']);           

% Standardized the macro state variables
[T, Nx] = size(MarketCovariates);
for j = 1:Nx
    MarketCovariates(:,j) = (MarketCovariates(:,j)-mean(MarketCovariates((WindowSize+1):(len-1),j)))/...
        sqrt(var(MarketCovariates((WindowSize+1):(len-1),j)));
end

% The positive and negative network factors
NetFactor_pos = zeros((len-WindowSize),N);
NetFactor_nev = zeros((len-WindowSize),N);

% Perform the TENQR model from the regional perspective
RscriptFileName = [codepath, 'rq_check.r'];
Rpath = 'D:\program\R-4.3.2\bin'; 
region = {'Comm', 'Stock', 'Forex', 'System'};
varnames = {'const', 'lagged Y', 'network pos', 'network neg', 'US bond yield',...
    'CHN bond yield', 'VIX', 'S&P GSCI'};

pv = 0.1:0.025:0.9; % quantile

geogr_loc = [1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]; 

country_name = {'WTI','Gold',...
                'ARG','AUS','BRA','CAN','CHN','DEU','EU','FRA','GBR','IDN','IND','ITA','JPN','KOR','MEX','RUS','SAU','TUR','USA','ZAF',...
                'ARS','AUD','BRL','CAD','CNY','EUR','GBP','IDR','INR','JPY','KRW','MXN','RUB','SAR','TRY','USD','ZAR'};
            
b_fullCL = zeros(Nx+1+2+1, length(pv), 4);     % The regression coefficient
b_fullCL_sd = zeros(Nx+1+2+1, length(pv), 4);  % The regression residual 

for iCL = 1:4
    disp(iCL)
    if (iCL==1) 
        which_cluster = find(geogr_loc==1);
    end
    if (iCL==2) 
        which_cluster = find(geogr_loc==2);
    end
    if (iCL==3) 
        which_cluster = find(geogr_loc==3);
    end
    if (iCL==4) 
        which_cluster = 1:39;
    end 
    
    Ypart = MarketReturn((WindowSize+1):len,which_cluster);
    Ypartlag = MarketReturn((WindowSize):(len-1),which_cluster);
    MV = MarketCovariates((WindowSize):(len-1),:);
    NetFactor_pos_CL = zeros((len-WindowSize),length(which_cluster));
    NetFactor_nev_CL = zeros((len-WindowSize),length(which_cluster));
    for w = 1:(len-WindowSize)
        % Positive network factor
        NetFactor_pos_CL(w, :) = (Adj_positive(which_cluster,which_cluster,w)*Ypartlag(w,:)')'...
            ./ sum(Adj_positive(which_cluster,which_cluster,w),2)';
        % Negative network factor
        NetFactor_nev_CL(w, :) = (Adj_negative(which_cluster,which_cluster,w)*Ypartlag(w,:)')'...
            ./ sum(Adj_negative(which_cluster,which_cluster,w),2)';        
    end

    % Set the missing value to 0
    NetFactor_pos_CL(find(isnan(NetFactor_pos_CL(:,:)))) = 0;
    NetFactor_nev_CL(find(isnan(NetFactor_nev_CL(:,:)))) = 0;

    Xall = [ones((len-WindowSize)*length(which_cluster),1) Ypartlag(:)...
            NetFactor_pos_CL(:) NetFactor_nev_CL(:) repmat(MV,length(which_cluster),1)];   %Explanatory variables
    YpartV = Ypart(:);  % Explained variables
    save([datapath, 'Xall.mat'], 'Xall') ;
    save([datapath, 'YpartV.mat'],'YpartV');
    save([datapath, 'pv.mat'],'pv');
    RunRcode(RscriptFileName, Rpath);
    fileID = fopen([datapath, 'rq_result.txt'],'r');
    fromR = fscanf(fileID,'%f', [length(pv)*2 Inf]);
    fromR=fromR';
    fclose(fileID);
    for ii = 1:length(pv)
        b_fullCL(:,ii,iCL) = fromR(:, (ii-1)*2+1);
        b_fullCL_sd(:,ii,iCL) = fromR(:,(ii-1)*2+2);
    end
    % Convert the results from txt format to xlsx format
    load([datapath, 'regresult_pos.txt'])
    load([datapath, 'regresult_neg.txt'])
    if (iCL==1) 
        xlswrite([datapath, 'regresult_pos_Comm.xlsx'], regresult_pos, 'regresult_pos', 'A');
        xlswrite([datapath, 'regresult_neg_Comm.xlsx'], regresult_neg, 'regresult_neg', 'A');
    end
    if (iCL==2) 
        xlswrite([datapath, 'regresult_pos_Stock.xlsx'], regresult_pos, 'regresult_pos', 'A');
        xlswrite([datapath, 'regresult_neg_Stock.xlsx'], regresult_neg, 'regresult_neg', 'A');
    end
    if (iCL==3) 
        xlswrite([datapath, 'regresult_pos_Forex.xlsx'], regresult_pos, 'regresult_pos', 'A');
        xlswrite([datapath, 'regresult_neg_Forex.xlsx'], regresult_neg, 'regresult_neg', 'A');
    end
    if (iCL==4) 
        xlswrite([datapath, 'regresult_pos_System.xlsx'], regresult_pos, 'regresult_pos', 'A');
        xlswrite([datapath, 'regresult_neg_System.xlsx'], regresult_neg, 'regresult_neg', 'A');
    end
end

% Slopes from the quantile regressions from different geographic regions
network_posfactor_beta = zeros(4,length(pv));
network_posfactor_sd = zeros(4,length(pv));
network_negfactor_beta = zeros(4,length(pv));
network_negfactor_sd = zeros(4,length(pv));
for j = 1:4
    network_posfactor_beta(j,:) = b_fullCL(3,:,j);
    network_posfactor_sd(j,:) = b_fullCL_sd(3,:,j);
    network_negfactor_beta(j,:) = b_fullCL(4,:,j);
    network_negfactor_sd(j,:) = b_fullCL_sd(4,:,j);
end

sub1 = figure;
figure(sub1)

title('The positive network factor','FontName','Times New Roman','FontSize',10);
colorspec1 = [1, 0, 0]; 
colorspec2 = [0, 1, 0];
colorspec3 = [0, 0, 1];
colorspec4 = [0, 0, 0];
[A1] = shadedErrorBar(pv,network_posfactor_beta(1,:), 0.8*network_posfactor_sd(1,:),'lineprops', {'-','color',colorspec1,'LineWidth',2}, 'transparent',1);   % 绘制阴影误差图
hold on
[A2] = shadedErrorBar(pv,network_posfactor_beta(2,:), 1.96*network_posfactor_sd(2,:),'lineprops', {'-','color',colorspec2,'LineWidth',2}, 'transparent',1);   % 绘制阴影误差图
hold on 
[A3] = shadedErrorBar(pv,network_posfactor_beta(3,:), 1.96*network_posfactor_sd(3,:),'lineprops', {'-','color',colorspec3,'LineWidth',2}, 'transparent',1);  % 绘制阴影误差图
hold on 
[A4] = shadedErrorBar(pv,network_posfactor_beta(4,:), 1.96*network_posfactor_sd(4,:),'lineprops', {'-','color',colorspec4,'LineWidth',2}, 'transparent',1);  % 绘制阴影误差图

legend([A1.mainLine,A2.mainLine,A3.mainLine,A4.mainLine],{'Comm', 'Stock', 'Forex', 'System'}, 'orientation','horizontal',...
    'location','North','FontName','Times New Roman','FontSize',10);
xlabel('quantile','FontName','Times New Roman','FontSize',10);
ylabel('beta','FontName','Times New Roman','FontSize',10);
% set(sub1,'PaperPosition',[0 0, 10 10])
set(sub1, 'PaperSize', [12 10]);
set(gca, 'FontSize',10,'FontAngle','Normal','FontName','Times New Roman')
set(gca, 'YLim', [-0.1 0.3])
hold off;


sub2 = figure;
figure(sub2)
title('The negative network factor','FontName','Times New Roman','FontSize',10);
colorspec1 = [1, 0, 0];
colorspec2 = [0, 1, 0];
colorspec3 = [0, 0, 1];
colorspec4 = [0, 0, 0];
[B1] = shadedErrorBar(pv,network_negfactor_beta(1,:), 0.8*network_negfactor_sd(1,:),'lineprops', {'-','color',colorspec1,'LineWidth',2}, 'transparent',1);   % 绘制阴影误差图
hold on
[B2] = shadedErrorBar(pv,network_negfactor_beta(2,:), 1.96*network_negfactor_sd(2,:),'lineprops', {'-','color',colorspec2,'LineWidth',2}, 'transparent',1);   % 绘制阴影误差图
hold on 
[B3] = shadedErrorBar(pv,network_negfactor_beta(3,:), 1.96*network_negfactor_sd(3,:),'lineprops', {'-','color',colorspec3,'LineWidth',2}, 'transparent',1);  % 绘制阴影误差图
hold on 
[B4] = shadedErrorBar(pv,network_negfactor_beta(4,:), 1.96*network_negfactor_sd(4,:),'lineprops', {'-','color',colorspec4,'LineWidth',2}, 'transparent',1);  % 绘制阴影误差图

legend([B1.mainLine,B2.mainLine,B3.mainLine,B4.mainLine],{'Comm', 'Stock', 'Forex', 'System'}, 'orientation','horizontal',...
    'location','North','FontName','Times New Roman','FontSize',10); 
xlabel('quantile','FontName','Times New Roman','FontSize',10);
ylabel('beta','FontName','Times New Roman','FontSize',10);
set(sub2, 'PaperSize', [12 10]);
set(gca, 'FontSize',10,'FontAngle','Normal','FontName','Times New Roman')
set(gca, 'YLim', [-0.1 0.3])
hold off;
