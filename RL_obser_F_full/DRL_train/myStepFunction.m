function [NextObs,Reward,IsDone,LoggedSignals] = myStepFunction(Action,LoggedSignals,...
    envConstants,dataConstants,trainConstants)
maxNumCompThreads(1);
Action(1) = min(max(Action(1),envConstants.ActionLimit(1)),envConstants.ActionLimit(2));
Action(2) = min(max(Action(2),envConstants.ActionLimit(1)),envConstants.ActionLimit(2));
%% Check if the given action is valid.
if ~discretize(Action(1),[envConstants.ActionLimit(1), envConstants.ActionLimit(2)])==1 ||...
     ~discretize(Action(2),[envConstants.ActionLimit(1) envConstants.ActionLimit(2)])==1
    error('Action must be %g for going left and %g for going right.',...
        envConstants.ActionLimit(1),envConstants.ActionLimit(2));
end
% Action size(2,1)
% disp(Action)
%% time evolution
Delta = 10*Action(1);
U = 10*Action(2);
V = dataConstants.VpreUpOrDown * Delta + ...
dataConstants.VpreUpAndDown *U;
expEWiseV = exp(-1j*dataConstants.dt*V);

LoggedSignals.State = expEWiseV .*(dataConstants.expHUphop ...
    *LoggedSignals.State*dataConstants.expHDownhop);

%% imbalance
I_t =  double(gather(sum(sum(abs(LoggedSignals.State).^2.*dataConstants.imbalance))));
I_t_up =  double(gather(sum(sum(abs(LoggedSignals.State).^2.*dataConstants.imbalanceUp))));
I_t_down =  double(gather(sum(sum(abs(LoggedSignals.State).^2.*dataConstants.imbalanceDown))));

%% fidelity
fullFidelity =  double(gather(abs(sum(sum(conj(LoggedSignals.State).*dataConstants.Mpsi_init)))^2));

Reward = -abs(sqrt(fullFidelity)-1);
NextObs = [fullFidelity]; 

LoggedSignals.num_steps = LoggedSignals.num_steps + 1;

LoggedSignals.Delta_steps = [LoggedSignals.Delta_steps,Delta];
LoggedSignals.U_steps = [LoggedSignals.U_steps,U];
LoggedSignals.reward_steps = [LoggedSignals.reward_steps,Reward];

LoggedSignals.fullFidelity_steps = [LoggedSignals.fullFidelity_steps,fullFidelity];

LoggedSignals.I_t_steps = [LoggedSignals.I_t_steps,I_t];
LoggedSignals.I_t_up_steps = [LoggedSignals.I_t_up_steps,I_t_up];
LoggedSignals.I_t_down_steps = [LoggedSignals.I_t_down_steps,I_t_down];

if LoggedSignals.num_steps == trainConstants.n_steps
    IsDone = true;

    %% step data
    Delta_steps = LoggedSignals.Delta_steps;
    U_steps = LoggedSignals.U_steps;
    reward_steps = LoggedSignals.reward_steps;

    fullFidelity_steps = LoggedSignals.fullFidelity_steps;
    I_t_steps = LoggedSignals.I_t_steps;
    I_t_up_steps = LoggedSignals.I_t_up_steps;
    I_t_down_steps = LoggedSignals.I_t_down_steps;

    n_steps = trainConstants.n_steps;
    data_dirname = dataConstants.data_dirname;
    DRL_dirname = dataConstants.DRL_dirname;
    %% episode
    data = load([data_dirname,'/episode_record.mat']);
    load([data_dirname,'/num_episode_record.mat']);
    meanReward_episode = data.meanReward_episode;

    fullFidelity_episode = data.fullFidelity_episode;
    
    num_episode = num_episode + 1;

    meanReward_episode = [meanReward_episode, sum(reward_steps)/n_steps];
    fullFidelity_episode = [fullFidelity_episode, sum(fullFidelity_steps)/n_steps];
 
    save([data_dirname,'/episode_record.mat'],'meanReward_episode', 'fullFidelity_episode')
    save([data_dirname,'/num_episode_record.mat'],'num_episode')

    disp('Episode')
    disp([num_episode])

    [val,~]=intersect(trainConstants.output_interval:trainConstants.output_interval:trainConstants.n_episode, ...
        num_episode);
    if ~isempty(val)
        %% episode plot
          %% plot reward
        f = figure();
        f.Position = [100 100 600 300];
        plot(1:1:num_episode, meanReward_episode ,'LineWidth',2);hold on;
        set(gca, 'LineWidth',1,'Fontname', 'Times New Roman','FontSize',20)
        xlabel('Episode','Interpreter','latex','FontSize',20)
        ylabel('mean Reward','Interpreter','latex','FontSize',20)
        saveas(gcf,[data_dirname,'/',DRL_dirname,'_picture/meanRewardEp_Ep',num2str(num_episode),'.png'])
        close(f)
        save([data_dirname,'/',DRL_dirname,'_picture_data/meanRewardEp_Ep',num2str(num_episode),'.mat'],'meanReward_episode')


        %% plot fidelity
        f = figure();
        f.Position = [100 100 600 300];
        plot(1:1:num_episode, fullFidelity_episode,'LineWidth',2);hold on;
        set(gca, 'LineWidth',1,'Fontname', 'Times New Roman','FontSize',20)
        xlabel('Episode','Interpreter','latex','FontSize',20)
        ylabel('$|\langle\psi(0)|\psi(t)\rangle|^2$','Interpreter','latex','FontSize',20)
        saveas(gcf,[data_dirname,'/',DRL_dirname,'_picture/full_fidelityEp_Ep',num2str(num_episode),'.png'])
        close(f)
        save([data_dirname,'/',DRL_dirname,'_picture_data/full_fidelityEp_Ep',num2str(num_episode),'.mat'], ...
            'fullFidelity_episode')

        %% step plot
        t = linspace(0,n_steps*dataConstants.dt,n_steps);

        %% plot Delta
        f = figure();
        f.Position = [100 100 600 300];
        plot(t, Delta_steps,'LineWidth',2);hold on;
        set(gca, 'LineWidth',1,'Fontname', 'Times New Roman','FontSize',20)
        xlabel('t','Interpreter','latex','FontSize',20)
        ylabel('$\Delta$','Interpreter','latex','FontSize',20)
        saveas(gcf,[data_dirname,'/',DRL_dirname,'_picture/DeltaStep_Ep',num2str(num_episode),'.png'])
        close(f)
        save([data_dirname,'/',DRL_dirname,'_picture_data/DeltaStep_Ep',num2str(num_episode),'.mat'],...
            'Delta_steps')

        %% plot U
        f = figure();
        f.Position = [100 100 600 300];
        plot(t, U_steps,'LineWidth',2);hold on;
        set(gca, 'LineWidth',1,'Fontname', 'Times New Roman','FontSize',20)
        xlabel('t','Interpreter','latex','FontSize',20)
        ylabel('$U$','Interpreter','latex','FontSize',20)
        saveas(gcf,[data_dirname,'/',DRL_dirname,'_picture/UStep_Ep',num2str(num_episode),'.png'])
        close(f)
        save([data_dirname,'/',DRL_dirname,'_picture_data/UStep_Ep',num2str(num_episode),'.mat'],...
            'U_steps')

        %% plot reward
        f = figure();
        f.Position = [100 100 600 300];
        plot(t, reward_steps,'LineWidth',2);hold on;
        set(gca, 'LineWidth',1,'Fontname', 'Times New Roman','FontSize',20)
        xlabel('t','Interpreter','latex','FontSize',20)
        ylabel('Reward','Interpreter','latex','FontSize',20)
        saveas(gcf,[data_dirname,'/',DRL_dirname,'_picture/rewardStep_Ep',num2str(num_episode),'.png'])
        close(f)
        save([data_dirname,'/',DRL_dirname,'_picture_data/rewardStep_Ep',num2str(num_episode),'.mat'],'reward_steps')


        %% plot fidelity
        f = figure();
        f.Position = [100 100 600 300];
        plot(t, fullFidelity_steps,'LineWidth',2);hold on;
        set(gca, 'LineWidth',1,'Fontname', 'Times New Roman','FontSize',20)
        xlabel('t','Interpreter','latex','FontSize',20)
        ylabel('$|\langle\psi(0)|\psi(t)\rangle|^2$','Interpreter','latex','FontSize',20)
        saveas(gcf,[data_dirname,'/',DRL_dirname,'_picture/full_fidelityStep_Ep',num2str(num_episode),'.png'])
        close(f)     
        save([data_dirname,'/',DRL_dirname,'_picture_data/full_fidelityStep_Ep',num2str(num_episode),'.mat'], ...
           'fullFidelity_steps')

        %% plot imbalance
        f = figure();
        f.Position = [100 100 600 300];
        plot(t, I_t_steps,'LineWidth',2);hold on;
        plot(t, I_t_up_steps,'LineWidth',2);hold on;
        plot(t, I_t_down_steps,'LineWidth',2);hold on;
        legend({'I(t)','I^{\uparrow}(t)', 'I^{\downarrow}(t)'},'Fontname', ...
            'Times New Roman','FontSize',20,'Location','best')
        set(gca, 'LineWidth',1,'Fontname', 'Times New Roman','FontSize',20)
        xlabel('t','Interpreter','latex','FontSize',20)
        ylabel('$\mathcal{I}(t)$','Interpreter','latex','FontSize',20)
        saveas(gcf,[data_dirname,'/',DRL_dirname,'_picture/imbalanceStep_Ep',num2str(num_episode),'.png'])
        close(f)
        save([data_dirname,'/',DRL_dirname,'_picture_data/imbalanceStep_Ep',num2str(num_episode),'.mat'],...
            'I_t_steps','I_t_up_steps','I_t_down_steps')

    end

else
    IsDone = false;
end

end