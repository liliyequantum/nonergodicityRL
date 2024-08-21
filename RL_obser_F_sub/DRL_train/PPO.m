function PPO(numLattice,numAChain,n_steps,n_episode,output_interval)
% clear;clc;close all;
% 
% numLattice = 10;
% numAChain = 4;
% n_steps = 1000;
% n_episode = 1000;
% output_interval = 50;
data_dirname = ['../numLattice_',num2str(numLattice),'_numAChain_',num2str(numAChain),...
    '_numUp_',num2str(int32(numLattice/2)),'_numDown_',num2str(int32(numLattice/2))];
DRL_dirname = 'train';

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

if exist([data_dirname,'/',DRL_dirname,'_picture'], 'dir')
     rmdir([data_dirname,'/',DRL_dirname,'_picture'],'s')
end
if exist([data_dirname,'/',DRL_dirname,'_picture_data'], 'dir')
     rmdir([data_dirname,'/',DRL_dirname,'_picture_data'],'s')
end

mkdir([data_dirname,'/',DRL_dirname,'_picture'])
mkdir([data_dirname,'/',DRL_dirname,'_picture_data'])

num_episode = 0;
meanReward_episode = [];
subFidelity_episode = [];
fullFidelity_episode = [];
save([data_dirname,'/episode_record.mat'],'meanReward_episode',...
    'subFidelity_episode', 'fullFidelity_episode')
save([data_dirname,'/num_episode_record.mat'],'num_episode')

trainConstants.n_episode = n_episode;%1000;
trainConstants.n_steps = n_steps;%1000;
trainConstants.output_interval = output_interval;%50;

%% 
ResetHandle = @() myResetFunction(dataConstants.Mpsi_init);
StepHandle = @(Action,LoggedSignals) myStepFunction(Action,LoggedSignals,...
    envConstants, dataConstants, trainConstants);
env = rlFunctionEnv(ObservationInfo,ActionInfo,StepHandle,ResetHandle);

obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);
numObs = prod(obsInfo.Dimension);
numAct = prod(actInfo.Dimension);

%% initial random number generator
rng(0)

%% create the critic network layers.
cnet = [
    featureInputLayer(numObs,Name="observation")
    fullyConnectedLayer(256)
    reluLayer
    fullyConnectedLayer(128)
    reluLayer
    fullyConnectedLayer(64)
    reluLayer
    fullyConnectedLayer(1,Name="CriticOutput")];

% Connect the layers.
criticNetwork = layerGraph(cnet);
% plot(criticNetwork)
critic = rlValueFunction(criticNetwork,obsInfo);

%% Create the actor network layers.
commonPath = [
    featureInputLayer(numObs,Name="observation")
    fullyConnectedLayer(256)
    reluLayer
    fullyConnectedLayer(128)
    reluLayer(Name="anet_out")];
meanPath = [
    fullyConnectedLayer(64,Name="meanFC")
    reluLayer(Name="relu3")
    fullyConnectedLayer(numAct,Name="mean")];
stdPath = [
    fullyConnectedLayer(numAct,Name="stdFC")
    reluLayer(Name="relu4")
    softplusLayer(Name="std")];

% Connect the layers.
actorNetwork = layerGraph(commonPath);
actorNetwork = addLayers(actorNetwork,meanPath);
actorNetwork = addLayers(actorNetwork,stdPath);
actorNetwork = connectLayers(actorNetwork,"anet_out","meanFC/in");
actorNetwork = connectLayers(actorNetwork,"anet_out","stdFC/in");
% plot(actorNetwork)

actordlnet = dlnetwork(actorNetwork);
actor = rlContinuousGaussianActor(actordlnet, obsInfo, actInfo, ...
    ObservationInputNames="observation", ...
    ActionMeanOutputNames="mean", ...
    ActionStandardDeviationOutputNames="std");

actorOpts = rlOptimizerOptions(LearnRate=1e-4);
criticOpts = rlOptimizerOptions(LearnRate=1e-4);

agentOpts = rlPPOAgentOptions(...
    ExperienceHorizon=trainConstants.n_steps,...
    MiniBatchSize=trainConstants.n_steps,...
    ClipFactor=0.02,...
    EntropyLossWeight=0.01,...
    ActorOptimizerOptions=actorOpts,...
    CriticOptimizerOptions=criticOpts,...
    NumEpoch=10,...
    AdvantageEstimateMethod="gae",...%Generalized advantage estimator
    GAEFactor=0.95,...
    SampleTime=0.1,...
    DiscountFactor=0.997);

agent = rlPPOAgent(actor,critic,agentOpts);
% save('nontrain_Agent.mat','agent')
%% train agent
trainOpts = rlTrainingOptions(...
    MaxEpisodes=trainConstants.n_episode,...
    MaxStepsPerEpisode=trainConstants.n_steps,...
    Plots="none",...
    StopTrainingCriteria="AverageReward",...
    StopTrainingValue=1e15,...
    ScoreAveragingWindowLength=100);

tic
trainingStats = train(agent, env, trainOpts);
toc
save([data_dirname,'/trained_Agent_n_steps_',num2str(n_steps),'_n_episode_',num2str(n_episode),'.mat'],'agent')
% end
