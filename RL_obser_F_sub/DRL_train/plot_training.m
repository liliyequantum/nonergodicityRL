clear;clc;close all;

load('../numLattice_10_numAChain_4_numUp_5_numDown_5/train_picture_data/full_sub_fidelityEp_Ep3000.mat')

%% plot fidelity
num_episode = 3000;
f = figure();
f.Position = [100 100 600 300];
plot(1:1:num_episode, fullFidelity_episode,'LineWidth',2);hold on;
plot(1:1:num_episode, subFidelity_episode,'LineWidth',2);hold on;
 legend({'full-chain','sub-chain'},'Fontname', 'Times New Roman','FontSize',20)
set(gca, 'LineWidth',1,'Fontname', 'Times New Roman','FontSize',20)
xlabel('Episode','Interpreter','latex','FontSize',20)
ylabel('$|\langle\psi(0)|\psi(t)\rangle|^2$','Interpreter','latex','FontSize',20)
ylim([0,1])

load('../numLattice_10_numAChain_4_numUp_5_numDown_5/train_picture_data/sub_full_fidelityStep_Ep1000.mat')
%% plot fidelity
t = linspace(0,2000*0.005,2000);
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