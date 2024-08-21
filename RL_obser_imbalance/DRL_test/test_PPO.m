function test_PPO(numLattice,test_steps,train_steps,train_episode)
% clear;clc;close all;
maxNumCompThreads(1);

%% par
% numLattice = 10;
% test_steps = 10;
% train_steps = 200;
% train_episode = 10;

lattice_dirname = ['../numLattice_',num2str(numLattice),...
    '_numUp_',num2str(int32(numLattice/2)),'_numDown_',num2str(int32(numLattice/2))'];
data_dirname = [lattice_dirname,'/n_steps_',num2str(train_steps)];
DRL_dirname = 'test';

%% env par %%
ObservationInfo = rlNumericSpec([2 1]); % specifies continuous action or observation
ObservationInfo.Name = "imbalanceDown, imbalanceUp";
ObservationInfo.Description = 'imbalanceDown, imbalanceUp';
ObservationInfo.LowerLimit = [-5;-5];
ObservationInfo.UpperLimit = [5;5];

ActionInfo =  rlNumericSpec([2 1]);
ActionInfo.Name = "Delta, U";
ActionInfo.Description = 'Delta, U';
maxForce = 1;
ActionInfo.LowerLimit = [-maxForce; -maxForce];
ActionInfo.UpperLimit = [maxForce; maxForce];

%% data
% save(['./numLattice_',num2str(numLattice),...
%     '_numUp_',num2str(numUp),'_numDown_',num2str(numDown),'/pre_data.mat']...
%     ,'numLattice','dt','imbalance','imbalanceUp','imbalanceDown',...
%     'VpreUpOrDown','VpreUpAndDown','expHUphop','expHDownhop', ...
%     'Mpsi_init','Mpsi_halfChain','upChain_row','downChain_col','halfChain_row',...
%     'halfChain_col','numHalfChainBasis','-v7.3')
dataConstants = load([lattice_dirname,'/pre_data.mat']);
dataConstants.data_dirname = data_dirname;
dataConstants.DRL_dirname = DRL_dirname;
dataConstants.I_t_down_0 = ...
    double(gather(sum(sum(abs(dataConstants.Mpsi_init).^2.*dataConstants.imbalanceDown))));
dataConstants.I_t_up_0 = ...
    double(gather(sum(sum(abs(dataConstants.Mpsi_init).^2.*dataConstants.imbalanceUp))));

envConstants.tol_svd = 1e-3;%svd tol for svds package
envConstants.cutoff_tol_svd = 0.005;%svd tol for cut off the minimum singular value
envConstants.ActionLimit = [-maxForce, maxForce];

trainConstants.n_steps = test_steps;

trainConstants.output_interval = 1;
trainConstants.n_episode = 1;

%% configuration
ResetHandle = @() myResetFunction(dataConstants.Mpsi_init, ...
    dataConstants.I_t_down_0,dataConstants.I_t_up_0);
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