function test_PPO(numLattice,numAChain, test_steps,train_steps,train_episode)
% clear;clc;close all;
% 
% %% testing phase 
% test_steps = 10;
% 
% numLattice = 8;
% numAChain = 4;
data_dirname = ['../numLattice_',num2str(numLattice),'_numAChain_',num2str(numAChain),...
    '_numUp_',num2str(int32(numLattice/2)),'_numDown_',num2str(int32(numLattice/2))];
DRL_dirname = 'test';

maxNumCompThreads(1);


%% env par %%
ObservationInfo = rlNumericSpec([1 1]); % specifies continuous action or observation
ObservationInfo.Name = "subFidelity";
ObservationInfo.Description = 'subFidelity';
ObservationInfo.LowerLimit = [0];
ObservationInfo.UpperLimit = [1];

ActionInfo =  rlNumericSpec([2 1]);
ActionInfo.Name = "Delta, U";
ActionInfo.Description = 'Delta, U';
maxForce = 1;
ActionInfo.LowerLimit = [-maxForce; -maxForce];
ActionInfo.UpperLimit = [maxForce; maxForce];

% save(['./numLattice_',num2str(numLattice),'_numAChain_',num2str(numAChain),...
%     '_numUp_',num2str(numUp),'_numDown_',num2str(numDown),'/pre_data.mat']...
%     ,'numLattice','dt','imbalance','imbalanceUp','imbalanceDown',...
%     'VpreUpOrDown','VpreUpAndDown','expHUphop','expHDownhop', ...
%     'numAChainBasis','numBChainBasis', 'numSubChainBasis','numHalfChainBasis',...
%     'upChain_row', 'downChain_col', ...
%     'subAChain_row', 'subBChain_col','halfChain_row','halfChain_col',...
%     'Mpsi_init','Mpsi_subChain','Mpsi_halfChain','rho_initial','-v7.3')
dataConstants = load([data_dirname,'/pre_data.mat']);
dataConstants.data_dirname = data_dirname;
dataConstants.DRL_dirname = DRL_dirname;

envConstants.tol_svd = 1e-3;%svd tol for svds package
envConstants.cutoff_tol_svd = 0.005;%svd tol for cut off the minimum singular value
envConstants.ActionLimit = [-maxForce, maxForce];

trainConstants.n_steps = test_steps;

trainConstants.output_interval = 1;
trainConstants.n_episode = 1;

%% 
ResetHandle = @() myResetFunction(dataConstants.Mpsi_init);
StepHandle = @(Action,LoggedSignals) myStepFunction(Action,LoggedSignals,...
    envConstants, dataConstants, trainConstants);
env = rlFunctionEnv(ObservationInfo,ActionInfo,StepHandle,ResetHandle);

%% agent
load([data_dirname,'/trained_Agent_n_steps_',num2str(train_steps),...
    '_n_episode_',num2str(train_episode),'.mat'])

if exist([data_dirname,'/',DRL_dirname,'_picture'], 'dir')
     rmdir([data_dirname,'/',DRL_dirname,'_picture'],'s')
end
if exist([data_dirname,'/',DRL_dirname,'_picture_data'], 'dir')
     rmdir([data_dirname,'/',DRL_dirname,'_picture_data'],'s')
end

mkdir([data_dirname,'/',DRL_dirname,'_picture'])
mkdir([data_dirname,'/',DRL_dirname,'_picture_data'])

%% test
rng(10)
simOptions = rlSimulationOptions(MaxSteps=test_steps);
simOptions.NumSimulations = 1;

tic
experience = sim(env, agent, simOptions);
toc

