clear;clc;close all;
load('sub_half_EntropyStep_Ep1.mat')

t = 0:0.005:(5-0.005);
f = figure();
f.Position = [100 100 600 300];
plot(t, subEntropy_steps,'LineWidth',2);hold on;
plot(t, halfEntropy_steps,'LineWidth',2);hold on;
legend({'sub chain','half chain'},'Fontname', 'Times New Roman','FontSize',20,'location','best')
set(gca, 'LineWidth',1,'Fontname', 'Times New Roman','FontSize',20)
xlabel('t','Interpreter','latex','FontSize',20)
ylabel('$S$','Interpreter','latex','FontSize',20)

load('sub_full_fidelityStep_Ep1.mat')
f = figure();
f.Position = [100 100 600 300];
plot(t, fullFidelity_steps,'LineWidth',2);hold on;
plot(t, subFidelity_steps,'LineWidth',2);hold on;
legend({'full chain','sub chain'},'Fontname', 'Times New Roman','FontSize',20,...
    'Location','best')
set(gca, 'LineWidth',1,'Fontname', 'Times New Roman','FontSize',20)
xlabel('t','Interpreter','latex','FontSize',20)
ylabel('$|\langle\psi(0)|\psi(t)\rangle|^2$','Interpreter','latex','FontSize',20)

load('imbalanceStep_Ep1.mat')
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