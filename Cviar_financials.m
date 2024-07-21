%function Cviar_financials
%本函数实现84只金融股票的CViaR的计算。
%日期 2016-3-6
%版权所有 wanggangjin@foxmail.com
clc
clear
load  E:\科研学习\期刊论文\期刊论文2\程序代码\其他参考代码\FinData.txt;

[len,N]=size(FinData);
% for i=1:Model
VaR_Fin_001=zeros(len,N);
Table_Coeff_001=zeros(15,N);
VaR_Fin_005=zeros(len,N);
Table_Coeff_005=zeros(15,N);

for i=1:N  
    [VaR_Fin_001(:,i), Table_Coeff_001(:,i)] = CAViaROptimisation_gjw(2, 0.01, FinData(:,i));
    [VaR_Fin_005(:,i), Table_Coeff_005(:,i)] = CAViaROptimisation_gjw(2, 0.05, FinData(:,i));
end

% csvwrite('H:\Matlab7\Risk_network_financials\Results\VaR_Fin_001.csv', VaR_Fin_001);
% csvwrite('H:\Matlab7\Risk_network_financials\Results\Table_Coeff_001.csv', Table_Coeff_001);
% csvwrite('H:\Matlab7\Risk_network_financials\Results\VaR_Fin_005.csv', VaR_Fin_005);
% csvwrite('H:\Matlab7\Risk_network_financials\Results\Table_Coeff_005.csv', Table_Coeff_005);


% csvwrite('H:\Matlab7\Risk_network_financials\Results\VaR_Fin_001.csv', VaR_Fin_001);
% csvwrite('H:\Matlab7\Risk_network_financials\Results\Table_Coeff_001.csv', Table_Coeff_001);
% csvwrite('H:\Matlab7\Risk_network_financials\Results\VaR_Fin_005.csv', VaR_Fin_005);
% csvwrite('H:\Matlab7\Risk_network_financials\Results\Table_Coeff_005.csv', Table_Coeff_005);

% csvwrite('H:\Matlab7\Risk_network_financials\Results\VaR_Fin_001.txt', VaR_Fin_001);
% csvwrite('H:\Matlab7\Risk_network_financials\Results\Table_Coeff_001.txt', Table_Coeff_001);
% csvwrite('H:\Matlab7\Risk_network_financials\Results\VaR_Fin_005.txt', VaR_Fin_005);
% csvwrite('H:\Matlab7\Risk_network_financials\Results\Table_Coeff_005.txt', Table_Coeff_005);