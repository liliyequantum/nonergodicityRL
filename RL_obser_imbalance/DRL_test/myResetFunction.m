function [InitialObservation, LoggedSignals] = myResetFunction(Mpsi_init, I_t_down_0,I_t_up_0)
% Reset function to place custom cart-pole environment into a random
% initial state.
% % Return initial environment state variables as logged signals.

LoggedSignals.max_k_half = 10;

LoggedSignals.State = Mpsi_init;
InitialObservation = [I_t_down_0;I_t_up_0]; % imbalance down

LoggedSignals.num_steps = 0;

LoggedSignals.Delta_steps = [];
LoggedSignals.U_steps = [];
LoggedSignals.reward_steps = [];

LoggedSignals.fullFidelity_steps = [];
LoggedSignals.I_t_steps = [];
LoggedSignals.I_t_up_steps = [];
LoggedSignals.I_t_down_steps = [];
LoggedSignals.halfEntropy_steps = [];
% disp('reset')
end