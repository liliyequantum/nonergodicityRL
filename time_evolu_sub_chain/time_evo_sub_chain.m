clear;clc;close all;
maxNumCompThreads(1);
    
numLattice = 10;
dirname = ['./numLattice_',num2str(numLattice),'_numAChain_4_numUp_',num2str(numLattice/2),...
    '_numDown_',num2str(numLattice/2)];
load([dirname,'/pre_data.mat'])
% save(['./numLattice_',num2str(numLattice),'_numAChain_',num2str(numAChain),...
%     '_numUp_',num2str(numUp),'_numDown_',num2str(numDown),'/pre_data.mat']...
%     ,'numLattice','dt','imbalance','imbalanceUp','imbalanceDown',...
%     'VpreUpOrDown','VpreUpAndDown','expHUphop','expHDownhop', ...
%     'numAChainBasis','numBChainBasis', 'numSubChainBasis','numHalfChainBasis',...
%     'upChain_row', 'downChain_col', ...
%     'subAChain_row', 'subBChain_col','halfChain_row','halfChain_col',...
%     'Mpsi_init','Mpsi_subChain','Mpsi_halfChain','rho_initial','-v7.3')

%% time evolution %%
disp('time evolution')
t = 0:dt:0.5;
Delta = -10; 
U_array = rand(1,length(t))*20-10;

max_k_sub = 10;
max_k_half = 10;

I_t = zeros(length(t),1);
I_t_up = zeros(length(t),1);
I_t_down = zeros(length(t),1);

F_full = zeros(length(t),1);
F_sub = zeros(length(t),1);

entropy_subChain = zeros(length(t),1);
entropy_halfChain = zeros(length(t),1);

svd_sub_accuracy = zeros(length(t),1);
svd_half_accuracy = zeros(length(t),1);

start_total = tic;

Mpsi = Mpsi_init;
for tdx =1:1:length(t)

    U = U_array(tdx);
    V = VpreUpOrDown * Delta +  VpreUpAndDown *  U;
    expEWiseV = exp(-1j*dt*V);

    Mpsi = expEWiseV .*(expHUphop * Mpsi * expHDownhop);
    F_full(tdx) = abs(sum(sum(conj(Mpsi).*Mpsi_init)))^2;
    I_t(tdx) = sum(sum(abs(Mpsi).^2.*imbalance));
    I_t_up(tdx) = sum(sum(abs(Mpsi).^2.*imbalanceUp));
    I_t_down(tdx) = sum(sum(abs(Mpsi).^2.*imbalanceDown));


    Mpsi_subChain(sub2ind(size(Mpsi_subChain),subAChain_row+1, subBChain_col+1)) = ...
        Mpsi(sub2ind(size(Mpsi),upChain_row+1,downChain_col+1));
    Mpsi_halfChain(sub2ind(size(Mpsi_halfChain),halfChain_row+1, halfChain_col+1)) = ...
        Mpsi(sub2ind(size(Mpsi),upChain_row+1,downChain_col+1));
    
    % sub entropy
    [U_svd,S,V_svd] = svds(Mpsi_subChain, max_k_sub,'L','Tolerance',1e-3);
    if abs(S(end,end))>0.005
        max_k_sub = min(2*max_k_sub,numSubChainBasis);
    end
    svd_sub_accuracy(tdx) = max(max(abs(Mpsi_subChain - U_svd*S*V_svd')));
    entropy_subChain(tdx) = -sum(diag(S).^2.*log(diag(S).^2));

    % half entropy
    [U_svd,S,V_svd] = svds(Mpsi_halfChain, max_k_half,'L','Tolerance',1e-3);
    if abs(S(end,end))>0.005
        max_k_half = min(2*max_k_half,numHalfChainBasis);
    end
    svd_half_accuracy(tdx) = max(max(abs(Mpsi_halfChain - U_svd*S*V_svd')));
    entropy_halfChain(tdx) = -sum(diag(S).^2.*log(diag(S).^2));

    % sub Fidelity
    rho =  Mpsi_subChain*Mpsi_subChain';
    sqrtm_rho = sqrtm(gather(rho));
    eig_vals = eig(sqrtm_rho*rho_initial*sqrtm_rho);
    F_sub(tdx) = real(sum(sqrt(eig_vals(eig_vals>0))));
 
    disp(['max_k sub:  ',num2str(max_k_sub),' max_k half: ',num2str(max_k_half)])
    disp(strcat(' numLattice: ',num2str(numLattice),' progress:  ',num2str(tdx/length(t)))) 

end

disp(strcat(' numLattice: ',num2str(numLattice),'  total time: ',num2str(toc(start_total))))
disp(strcat(' numLattice: ',num2str(numLattice),'  svd sub accuracy:  ',num2str(max(svd_sub_accuracy))))
disp(strcat(' numLattice: ',num2str(numLattice),'  svd half accuracy:  ',num2str(max(svd_half_accuracy))))

if exist([dirname,'/timeEvolution'], 'dir')
     rmdir([dirname,'/timeEvolution'],'s')
end
mkdir([dirname,'/timeEvolution'])

save([dirname,'/timeEvolution/svd_sub_accuracy_maxt',num2str(max(t)),'.mat'],'svd_sub_accuracy')
save([dirname,'/timeEvolution/svd_half_accuracy_maxt',num2str(max(t)),'.mat'],'svd_half_accuracy')

save([dirname,'/timeEvolution/I_t_up_down_maxt',num2str(max(t)),'.mat'],'t','I_t','I_t_up','I_t_down')
save([dirname,'/timeEvolution/fidelity_F_sub_maxt',num2str(max(t)),'.mat'],'t','F_full','F_sub')
save([dirname,'/timeEvolution/entropy_subChain_maxt',num2str(max(t)),'.mat'],'t','entropy_subChain','entropy_halfChain')
%% plot imbalance
f = figure();
f.Position = [100 100 600 300];
plot(t,I_t,'LineWidth',2);hold on;
plot(t,I_t_up,'LineWidth',2);hold on;
plot(t,I_t_down,'LineWidth',2);hold on;
legend({'I(t)','I^{\uparrow}(t)', 'I^{\downarrow}(t)'},'Fontname', 'Times New Roman','FontSize',20,'location','best')
axis tight
% ylim([-0.05,0.7])
set(gca, 'LineWidth',1,'Fontname', 'Times New Roman','FontSize',20)
xlabel('t','Interpreter','latex','FontSize',20)
ylabel('$I(t)$','Interpreter','latex','FontSize',20)

%% plot fidelity
f = figure();
f.Position = [100 100 600 300];
plot(t, F_full,'LineWidth',2);hold on;
plot(t, F_sub,'LineWidth',2);hold on;
legend({'full chain','sub chain'},'Fontname', 'Times New Roman','FontSize',20,'location','best')
set(gca, 'LineWidth',1,'Fontname', 'Times New Roman','FontSize',20)
xlabel('t','Interpreter','latex','FontSize',20)
ylabel('$|\langle\psi(0)|\psi(t)\rangle|^2$','Interpreter','latex','FontSize',20)

%% plot entropy
f = figure();
f.Position = [100 100 600 300];
plot(t, entropy_subChain,'LineWidth',2);hold on;
plot(t, entropy_halfChain,'LineWidth',2);hold on;
legend({'sub chain','half chain'},'Fontname', 'Times New Roman','FontSize',20,'location','best')
set(gca, 'LineWidth',1,'Fontname', 'Times New Roman','FontSize',20)
xlabel('t','Interpreter','latex','FontSize',20)
ylabel('$S$','Interpreter','latex','FontSize',20)
