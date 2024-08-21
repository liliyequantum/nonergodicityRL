clear;clc;close all

load('../numLattice_14_numAChain_4_numUp_7_numDown_7/train_picture_data/full_sub_fidelityEp_Ep1000.mat')
%% plot fidelity
num_episode = 1000;
f = figure();
f.Position = [100 100 600 300];
plot(1:1:num_episode, fullFidelity_episode,'LineWidth',2);hold on;
plot(1:1:num_episode, subFidelity_episode,'LineWidth',2);hold on;
 legend({'full-chain','sub-chain'},'Fontname', 'Times New Roman','FontSize',20)
set(gca, 'LineWidth',1,'Fontname', 'Times New Roman','FontSize',20)
xlabel('Episode','Interpreter','latex','FontSize',20)
ylabel('$|\langle\psi(0)|\psi(t)\rangle|^2$','Interpreter','latex','FontSize',20)

load('../numLattice_14_numAChain_4_numUp_7_numDown_7/train_picture_data/sub_full_fidelityStep_Ep1000.mat')
%% plot fidelity
t = linspace(0,1000*0.005,1000);
f = figure();
f.Position = [100 100 600 300];
plot(t, fullFidelity_steps,'LineWidth',2);hold on;
plot(t, subFidelity_steps,'LineWidth',2);hold on;
% legend({'full','sub'},'Fontname', 'Times New Roman','FontSize',20,...
%     'Location','eastoutside')
set(gca, 'LineWidth',1,'Fontname', 'Times New Roman','FontSize',20)
xlabel('$t(\tau)$','Interpreter','latex','FontSize',20)
ylabel('$|\langle\psi(0)|\psi(t)\rangle|^2$','Interpreter','latex','FontSize',20)
axis tight

load('../numLattice_14_numAChain_4_numUp_7_numDown_7/train_picture_data/imbalanceStep_Ep1000.mat')
%% plot imbalance
t = linspace(0,1000*0.005,1000);
f = figure();
f.Position = [100 100 600 300];
plot(t, I_t_steps,'LineWidth',2);hold on;
plot(t, I_t_up_steps,'LineWidth',2);hold on;
plot(t, I_t_down_steps,'LineWidth',2);hold on;
legend({'I(t)','I^{\uparrow}(t)', 'I^{\downarrow}(t)'},'Fontname', ...
    'Times New Roman','FontSize',20,'Location','eastoutside')
set(gca, 'LineWidth',1,'Fontname', 'Times New Roman','FontSize',20)
xlabel('$t(\tau)$','Interpreter','latex','FontSize',20)
ylabel('$\mathcal{I}(t)$','Interpreter','latex','FontSize',20)
axis tight
