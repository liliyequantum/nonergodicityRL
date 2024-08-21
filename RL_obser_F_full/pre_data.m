function pre_data(idx_initial_state)
maxNumCompThreads(1);

numLattice = 8;
dirname = ['./numLattice_',num2str(numLattice),'_numUp_',num2str(numLattice/2),...
    '_numDown_',num2str(numLattice/2)];
dir_initial_state = [dirname,'/idx_initial_state_', num2str(idx_initial_state)];
load([dirname,'/data.mat'])

if exist(dir_initial_state, 'dir')
     rmdir(dir_initial_state,'s')
end
mkdir(dir_initial_state)

% savemat(rundir+'/data.mat',{'numLattice':numLattice,\
% 'numHalfChain':numHalfChain,\
% 'numUp':numUp,'numDown':numDown,\
% 'numUpChainBasis':numUpChainBasis,'numDownChainBasis':numDownChainBasis,\
% 'numHalfChainBasis':numHalfChainBasis,\
% 'numUp_even_bit':numUp_even_bit,'numDown_even_bit':numDown_even_bit,'halfChainIndex2Pattern':halfChainIndex2Pattern,\
% 'upChainIndex2Pattern' : upChainIndex2Pattern,'downChainIndex2Pattern' : downChainIndex2Pattern,\
% 'HUphop':HUphop,'HDownhop':HDownhop,'VpreUpOrDown':VpreUpOrDown,'VpreUpAndDown':VpreUpAndDown,\
% 'upChain_row':upChain_row, 'downChain_col':downChain_col, \
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

expHUphop = expm(-1j*dt*HUphop); %matrix exponential
expHDownhop = expm(-1j*dt*HDownhop); %matrix exponential

%%  Mpsi_init
disp('initial state')
if idx_initial_state == 0 %|-+>
    upKet = '01100110';
    downKet = '10011001';
elseif idx_initial_state == 1 % |+->
    upKet = '10011001';
    downKet = '01100110';
elseif idx_initial_state == 2 % red
    upKet = '10101010';
    downKet = '01010101';
elseif idx_initial_state == 3 %blue
    upKet = '10110010';
    downKet = '01001101';
elseif idx_initial_state == 4 % bad performance
    upKet = '00010111';
    downKet ='11110000';
elseif idx_initial_state == 5 % red
    upKet =  '01010101';
    downKet = '10101010';
elseif idx_initial_state == 6 % blue
     upKet = '10101100';
    downKet = '01010011';
elseif idx_initial_state == 7 % blue
     upKet = '11001100';
    downKet = '00110011';
end

Mpsi_init = single(zeros(numUpChainBasis,numDownChainBasis));
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

Mpsi_halfChain = zeros(numHalfChainBasis,numHalfChainBasis); % each run reinitial for entropy
Mpsi_halfChain(sub2ind(size(Mpsi_halfChain),halfChain_row+1, halfChain_col+1)) = ...
        Mpsi_init(sub2ind(size(Mpsi_init),upChain_row+1,downChain_col+1));

save([dir_initial_state,'/pre_data.mat']...
    ,'numLattice','dt','imbalance','imbalanceUp','imbalanceDown',...
    'VpreUpOrDown','VpreUpAndDown','expHUphop','expHDownhop', ...
    'upChain_row', 'downChain_col', ...
    'halfChain_row','halfChain_col','numHalfChainBasis',...
    'Mpsi_init','Mpsi_halfChain','-v7.3')