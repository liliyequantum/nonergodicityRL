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

%% Mpsi_subChain 
dataConstants.Mpsi_subChain(sub2ind(size(dataConstants.Mpsi_subChain), ...
    dataConstants.subAChain_row+1, dataConstants.subBChain_col+1)) = ...
         LoggedSignals.State(sub2ind(size(LoggedSignals.State), ...
         dataConstants.upChain_row+1,dataConstants.downChain_col+1));

%% sub entropy
[U_svd,S,~] = svds(dataConstants.Mpsi_subChain,LoggedSignals.max_k_sub,'L','Tolerance',envConstants.tol_svd);
if abs(S(end,end))>envConstants.cutoff_tol_svd
    LoggedSignals.max_k_sub= min(2*LoggedSignals.max_k_sub,dataConstants.numSubChainBasis);
end
subEntropy = gather(-sum(diag(S).^2.*log(diag(S).^2)));

% Mpsi_halfChain 
dataConstants.Mpsi_halfChain(sub2ind(size(dataConstants.Mpsi_halfChain), ...
    dataConstants.halfChain_row+1, dataConstants.halfChain_col+1)) = ...
         LoggedSignals.State(sub2ind(size(LoggedSignals.State), ...
         dataConstants.upChain_row+1,dataConstants.downChain_col+1));

%% half entropy
[~,S_half,~] = svds(dataConstants.Mpsi_halfChain,LoggedSignals.max_k_half, 'L','Tolerance',envConstants.tol_svd);
if abs(S_half(end,end))>envConstants.cutoff_tol_svd
    LoggedSignals.max_k_half= min(2*LoggedSignals.max_k_half,dataConstants.numHalfChainBasis);
end
halfEntropy = gather(-sum(diag(S_half).^2.*log(diag(S_half).^2)));

%% sub Fidelity
sqrtm_rho = U_svd*S*U_svd';
% rho =  dataConstants.Mpsi_subChain*dataConstants.Mpsi_subChain';
% sqrtm_rho = sqrtm(gather(rho));
eig_vals = eig(sqrtm_rho*dataConstants.rho_initial*sqrtm_rho);
subFidelity = gather(real(sum(sqrt(eig_vals(eig_vals>0)))));

Reward = -abs(sqrt(subFidelity)-1);
NextObs = [subFidelity]; % subFidelity

LoggedSignals.num_steps = LoggedSignals.num_steps + 1;

LoggedSignals.Delta_steps = [LoggedSignals.Delta_steps,Delta];
LoggedSignals.U_steps = [LoggedSignals.U_steps,U];
LoggedSignals.reward_steps = [LoggedSignals.reward_steps,Reward];

LoggedSignals.subFidelity_steps =[LoggedSignals.subFidelity_steps, subFidelity];
LoggedSignals.fullFidelity_steps = [LoggedSignals.fullFidelity_steps,fullFidelity];

LoggedSignals.subEntropy_steps = [LoggedSignals.subEntropy_steps, subEntropy];
LoggedSignals.halfEntropy_steps = [LoggedSignals.halfEntropy_steps, halfEntropy];

LoggedSignals.I_t_steps = [LoggedSignals.I_t_steps,I_t];
LoggedSignals.I_t_up_steps = [LoggedSignals.I_t_up_steps,I_t_up];
LoggedSignals.I_t_down_steps = [LoggedSignals.I_t_down_steps,I_t_down];

disp(['num steps: ',num2str(LoggedSignals.num_steps)])
disp(['max_k sub: ',num2str(LoggedSignals.max_k_sub)])
disp(['max_k half: ',num2str(LoggedSignals.max_k_half)])


if LoggedSignals.num_steps == trainConstants.n_steps
    IsDone = true;

    %% step data
    Delta_steps = LoggedSignals.Delta_steps;
    U_steps = LoggedSignals.U_steps;
    reward_steps = LoggedSignals.reward_steps;

    subFidelity_steps = LoggedSignals.subFidelity_steps;
    fullFidelity_steps = LoggedSignals.fullFidelity_steps;
    subEntropy_steps = LoggedSignals.subEntropy_steps;
    halfEntropy_steps = LoggedSignals.halfEntropy_steps;
    I_t_steps = LoggedSignals.I_t_steps;
    I_t_up_steps = LoggedSignals.I_t_up_steps;
    I_t_down_steps = LoggedSignals.I_t_down_steps;

    n_steps = trainConstants.n_steps;
    data_dirname = dataConstants.data_dirname;
    DRL_dirname = dataConstants.DRL_dirname;

    %% step plot
    t = linspace(0,n_steps*dataConstants.dt,n_steps);

    %% plot Delta
    f = figure();
    f.Position = [100 100 600 300];
    plot(t, Delta_steps,'LineWidth',2);hold on;
    set(gca, 'LineWidth',1,'Fontname', 'Times New Roman','FontSize',20)
    xlabel('t','Interpreter','latex','FontSize',20)
    ylabel('$\Delta$','Interpreter','latex','FontSize',20)
    saveas(gcf,[data_dirname,'/',DRL_dirname,'_picture/DeltaStep_test.png'])
    close(f)
    save([data_dirname,'/',DRL_dirname,'_picture_data/DeltaStep_test.mat'],...
        'Delta_steps')

     %% plot Delta, U
    f = figure();
    f.Position = [100 100 600 300];
    plot(t, U_steps,'LineWidth',2);hold on;
    set(gca, 'LineWidth',1,'Fontname', 'Times New Roman','FontSize',20)
    xlabel('t','Interpreter','latex','FontSize',20)
    ylabel('$U$','Interpreter','latex','FontSize',20)
    saveas(gcf,[data_dirname,'/',DRL_dirname,'_picture/UStep_test.png'])
    close(f)
    save([data_dirname,'/',DRL_dirname,'_picture_data/UStep_test.mat'],...
       'U_steps')

    %% plot reward
    f = figure();
    f.Position = [100 100 600 300];
    plot(t, reward_steps,'LineWidth',2);hold on;
    set(gca, 'LineWidth',1,'Fontname', 'Times New Roman','FontSize',20)
    xlabel('t','Interpreter','latex','FontSize',20)
    ylabel('Reward','Interpreter','latex','FontSize',20)
    saveas(gcf,[data_dirname,'/',DRL_dirname,'_picture/rewardStep_test.png'])
    close(f)
    save([data_dirname,'/',DRL_dirname,'_picture_data/rewardStep_test.mat'],'reward_steps')

    %% plot entropy
    f = figure();
    f.Position = [100 100 600 300];
    plot(t, subEntropy_steps,'LineWidth',2);hold on;
    plot(t, halfEntropy_steps,'LineWidth',2);hold on;
    legend({'sub chain','half chain'},'Fontname', 'Times New Roman','FontSize',20,'location','best')
    set(gca, 'LineWidth',1,'Fontname', 'Times New Roman','FontSize',20)
    xlabel('t','Interpreter','latex','FontSize',20)
    ylabel('$S$','Interpreter','latex','FontSize',20)
    saveas(gcf,[data_dirname,'/',DRL_dirname,'_picture/sub_half_EntropyStep_test.png'])
    close(f)
    save([data_dirname,'/',DRL_dirname,'_picture_data/sub_half_EntropyStep_test.mat'],...
        'subEntropy_steps',  'halfEntropy_steps')
    
    %% plot fidelity
    f = figure();
    f.Position = [100 100 600 300];
    plot(t, fullFidelity_steps,'LineWidth',2);hold on;
    plot(t, subFidelity_steps,'LineWidth',2);hold on;
    legend({'full chain','sub chain'},'Fontname', 'Times New Roman','FontSize',20,...
        'Location','best')
    set(gca, 'LineWidth',1,'Fontname', 'Times New Roman','FontSize',20)
    xlabel('t','Interpreter','latex','FontSize',20)
    ylabel('$|\langle\psi(0)|\psi(t)\rangle|^2$','Interpreter','latex','FontSize',20)
    saveas(gcf,[data_dirname,'/',DRL_dirname,'_picture/sub_full_fidelityStep_test.png'])
    close(f)     
    save([data_dirname,'/',DRL_dirname,'_picture_data/sub_full_fidelityStep_test.mat'], ...
        'subFidelity_steps','fullFidelity_steps')

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
    saveas(gcf,[data_dirname,'/',DRL_dirname,'_picture/imbalanceStep_test.png'])
    close(f)
    save([data_dirname,'/',DRL_dirname,'_picture_data/imbalanceStep_test.mat'],...
        'I_t_steps','I_t_up_steps','I_t_down_steps')

else
    IsDone = false;
end

end