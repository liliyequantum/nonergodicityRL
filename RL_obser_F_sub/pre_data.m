clear;clc;close all;
maxNumCompThreads(1);
numLattice = 10;
read_dirname = ['./numLattice_',num2str(numLattice),'_numAChain_1_numUp_',num2str(numLattice/2),...
    '_numDown_',num2str(numLattice/2)];
% subBChainIndex2Pattern = h5read([read_dirname,'/subBChainIndex2Pattern.mat'],'/subBChainIndex2Pattern');

load([read_dirname,'/data.mat'])
% savemat(rundir+'/data.mat',{'numLattice':numLattice,\
% 'numHalfChain':numHalfChain,'numAChain':numAChain,'numBChain':numBChain,\
% 'numUp':numUp,'numDown':numDown,\
% 'numUpChainBasis':numUpChainBasis,'numDownChainBasis':numDownChainBasis,\
% 'numAChainBasis':numAChainBasis,'numBChainBasis':numBChainBasis,'numHalfChainBasis':numHalfChainBasis,\
% 'numUp_even_bit':numUp_even_bit,'numDown_even_bit':numDown_even_bit,'halfChainIndex2Pattern':halfChainIndex2Pattern,\
% 'upChainIndex2Pattern' : upChainIndex2Pattern,'downChainIndex2Pattern' : downChainIndex2Pattern,\
% 'HUphop':HUphop,'HDownhop':HDownhop,'VpreUpOrDown':VpreUpOrDown,'VpreUpAndDown':VpreUpAndDown,\
% 'upChain_row':upChain_row, 'downChain_col':downChain_col, \
% 'subAChain_row':subAChain_row, 'subBChain_col':subBChain_col,\
% 'halfChain_row':halfChain_row, 'halfChain_col':halfChain_col})
dt = 0.005;

%% imbalance, VpreUpOrDown, VpreUpAndDown, expHhop 
numUp_odd_bit = numUp - numUp_even_bit;
numDown_odd_bit = numDown - numDown_even_bit;
[down_num_odd_bit,up_num_odd_bit] = meshgrid(numDown_odd_bit,numUp_odd_bit);
[down_num_even_bit,up_num_even_bit] = meshgrid(numDown_even_bit,numUp_even_bit);
imbalance = double((up_num_odd_bit - up_num_even_bit) +...
    (down_num_odd_bit  - down_num_even_bit))/double(numUp+numDown);
imbalanceUp = double(up_num_odd_bit - up_num_even_bit)/double(numUp);
imbalanceDown = double(down_num_odd_bit  - down_num_even_bit)/double(numDown);

imbalance = gpuArray(imbalance);
imbalanceUp = gpuArray(imbalanceUp);
imbalanceDown = gpuArray(imbalanceDown);
VpreUpOrDown = gpuArray(VpreUpOrDown);
VpreUpAndDown = gpuArray(VpreUpAndDown);
HUphop = gpuArray(HUphop);
HDownhop = gpuArray(HDownhop);

expHUphop = expm(-1j*dt*HUphop); %matrix exponential
expHDownhop = expm(-1j*dt*HDownhop); %matrix exponential

%%  Mpsi_init
disp('initial state')
if numLattice==10
    upKet = '0110011001';
    downKet = '1001100110';
elseif   numLattice==12
    upKet = '011001100110';
    downKet = '100110011001';
elseif numLattice==14
    upKet = '01100110011001';
    downKet = '10011001100110';
end

Mpsi_init = gpuArray(single(zeros(numUpChainBasis,numDownChainBasis)));
for i=1:1:numUpChainBasis
    if upKet==upChainIndex2Pattern(i,:)
        rowdx = i;
    end
end

for i=1:1:numDownChainBasis
    if downKet==downChainIndex2Pattern(i,:)
        coldx = i;
    end 
end
Mpsi_init(rowdx,coldx) = 1;

Mpsi_subChain = gpuArray(zeros(numAChainBasis,numBChainBasis)); % each run reinitial for entropy
Mpsi_subChain(sub2ind(size(Mpsi_subChain),subAChain_row+1, subBChain_col+1)) = ...
        Mpsi_init(sub2ind(size(Mpsi_init),upChain_row+1,downChain_col+1));
rho_initial =  gather(complex(Mpsi_subChain*Mpsi_subChain'));

Mpsi_halfChain = gpuArray(zeros(numHalfChainBasis,numHalfChainBasis)); % each run reinitial for entropy
Mpsi_halfChain(sub2ind(size(Mpsi_halfChain),halfChain_row+1, halfChain_col+1)) = ...
        Mpsi_init(sub2ind(size(Mpsi_init),upChain_row+1,downChain_col+1));

numSubChainBasis = min(numAChainBasis,numBChainBasis);
save(['./numLattice_',num2str(numLattice),'_numAChain_',num2str(numAChain),...
    '_numUp_',num2str(numUp),'_numDown_',num2str(numDown),'/pre_data.mat']...
    ,'numLattice','dt','imbalance','imbalanceUp','imbalanceDown',...
    'VpreUpOrDown','VpreUpAndDown','expHUphop','expHDownhop', ...
    'numAChainBasis','numBChainBasis', 'numSubChainBasis','numHalfChainBasis',...
    'upChain_row', 'downChain_col', ...
    'subAChain_row', 'subBChain_col','halfChain_row','halfChain_col',...
    'Mpsi_init','Mpsi_subChain','Mpsi_halfChain','rho_initial','-v7.3')