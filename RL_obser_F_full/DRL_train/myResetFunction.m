function [InitialObservation, LoggedSignals] = myResetFunction(Mpsi_init)
% Reset function to place custom cart-pole environment into a random
% initial state.
% % Return initial environment state variables as logged signals.

LoggedSignals.State = Mpsi_init;
InitialObservation = [1]; % full fidelity

LoggedSignals.num_steps = 0;

LoggedSignals.Delta_steps = [];
LoggedSignals.U_steps = [];
LoggedSignals.reward_steps = [];

LoggedSignals.fullFidelity_steps = [];

LoggedSignals.I_t_steps = [];
LoggedSignals.I_t_up_steps = [];
LoggedSignals.I_t_down_steps = [];
% disp('reset')
end