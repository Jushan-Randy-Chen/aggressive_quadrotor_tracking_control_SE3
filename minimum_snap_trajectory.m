function minimum_snap_demo_scaled()
% Minimum-snap (order-7) trajectory with explicit nondimensional (tau),
% temporal scaling (alpha = T_total), and spatial scaling (beta1,beta2).
% Requires Optimization Toolbox (quadprog) or falls back to KKT solve.

%% ----------------------- Figure-8 waypoints (parametric) -----------------
t = linspace(0, 2*pi, 200);         % path parameter (NOT time)
x_T = sin(t);
y_T = sin(2*t);
z_T = 1.0 * ones(size(t));

% Desired yaw tangent to path (unwrap for continuity)
x_T_dot = cos(t);
y_T_dot = 2*cos(2*t);
yaw_T   = unwrap(atan2(y_T_dot, x_T_dot));

% Pick K keyframes along the path parameter (uniform in parameter here)
K   = 35;
idx = round(linspace(1, numel(t), K));
tau_k = t(idx);                          % "knot parameter"
tau_k = tau_k - tau_k(1);                % start at 0
tau_k = tau_k / tau_k(end);              % normalize to [0,1]  -> nondimensional time τ ; this step is important!

% waypoints at keyframes
wx   = x_T(idx);
wy   = y_T(idx);
wz   = z_T(idx);
wyaw = yaw_T(idx);

%% ----------------------- Temporal scaling (choose α) ---------------------
% Choose real execution time (seconds). α = T_total maps t = α * τ.
T_total = 10.0;      % <-- change this to fly slower/faster without re-solving in τ
tk_sec  = T_total * tau_k;      % physical knot times (seconds)
Tseg    = diff(tk_sec);  assert(all(Tseg>0), 'Times must be increasing.');

%% ----------------------- Spatial scaling (β1, β2) ------------------------
% Map w = β1 + β2 * w_tilde  <=>  w_tilde = (w - β1)/β2.
% This improves conditioning and decouples shape from size/offset.
[wx_tilde, beta1x, beta2x] = affine_normalize(wx);
[wy_tilde, beta1y, beta2y] = affine_normalize(wy);
[wz_tilde, beta1z, beta2z] = affine_normalize(wz);

% For yaw, a simple choice is identity scaling (β1=0, β2=1). Keep unwrap.
wyaw_tilde = wyaw; beta1yaw = 0; beta2yaw = 1;

%% ----------------------- Solve min-snap in τ-domain ----------------------
order    = 7;         % degree n
dmin_pos = 4;         % x,y,z: minimize snap
dmin_yaw = 2;         % yaw:   minimize angular acceleration (kψ=2)

bc = struct('use_start',[1 1 1 1], 'use_end',[1 1 1 1]);  % fix p,v,a,j at both ends

% Solve ON τ \in [0,1] with τ-knots `tau_k`
coefX_tau   = solveMinsnap(wx_tilde,   tau_k, order, dmin_pos, bc);  % in τ
coefY_tau   = solveMinsnap(wy_tilde,   tau_k, order, dmin_pos, bc);
coefZ_tau   = solveMinsnap(wz_tilde,   tau_k, order, dmin_pos, bc);
coefYaw_tau = solveMinsnap(wyaw_tilde, tau_k, order, dmin_yaw, bc);

%% ----------------------- Map coefficients to real time & space ----------
% For each segment polynomial:  q(τ) = Σ b_i τ^i   (coef_tau = high→low b_i)
% Real time t = α τ,  so τ = t/α  →  q(t) = Σ b_i (t/α)^i = Σ (b_i α^{-i}) t^i.
% Then spatial: w(t) = β1 + β2 q(t).
coefX_t   = tau_to_time_and_space(coefX_tau,   T_total, beta1x,   beta2x); %recover spatial dimensions
coefY_t   = tau_to_time_and_space(coefY_tau,   T_total, beta1y,   beta2y);
coefZ_t   = tau_to_time_and_space(coefZ_tau,   T_total, beta1z,   beta2z);
coefYaw_t = tau_to_time_and_space(coefYaw_tau, T_total, beta1yaw, beta2yaw);

%% ----------------------- Evaluate (physical time) ------------------------
ts = linspace(tk_sec(1), tk_sec(end), 1000);    % seconds

[X, dX, ddX]       = evalPiecewise(ts, tk_sec, coefX_t);
[Y, dY, ddY]       = evalPiecewise(ts, tk_sec, coefY_t);
[Z, dZ, ddZ]       = evalPiecewise(ts, tk_sec, coefZ_t);
[Yaw, dYaw, ddYaw] = evalPiecewise(ts, tk_sec, coefYaw_t);

figure('Name','Minimum-Snap with Time & Spatial Scaling');
tiledlayout(2,3,'Padding','compact','TileSpacing','compact');
nexttile; plot3(X,Y,Z,'-'); hold on; plot3(wx,wy,wz,'ro'); grid on; axis equal
title('3D path'); xlabel('x'); ylabel('y'); zlabel('z');
nexttile; plot(ts,X,'-'); hold on; stem(tk_sec,wx,'r.'); title('x(t)'); grid on
nexttile; plot(ts,Y,'-'); hold on; stem(tk_sec,wy,'r.'); title('y(t)'); grid on
nexttile; plot(ts,Z,'-'); hold on; title('z(t)'); grid on
nexttile; plot(ts,wrapToPi(Yaw),'-'); hold on; stem(tk_sec,wrapToPi(wyaw),'r.'); title('yaw(t)'); grid on
nexttile; plot(ts, sqrt(dX.^2 + dY.^2 + dZ.^2)); title('speed ||v||'); grid on

traj = table(ts.', X.', Y.', Z.', dX.', dY.', dZ.', ddX.', ddY.', ddZ.', Yaw.', dYaw.', ddYaw.', ...
    'VariableNames', {'t','x','y','z','vx','vy','vz','ax','ay','az','yaw','yawrate','yawacc'});
disp(traj(1:8,:))

end % main


% ====================== Scaling helpers ===================================
function [w_tilde, beta1, beta2] = affine_normalize(w)
% Center and scale for numeric conditioning in quadprog 
beta1 = mean(w);           % shift to zero meanm
rng_  = max(w)-min(w);
beta2 = max(rng_, 1);      % avoid tiny scale; set to 1 if nearly constant
w_tilde = (w - beta1) / beta2;
end

function coef_t = tau_to_time_and_space(coef_tau, alpha, beta1, beta2)
% Map coefficients from τ to physical time t and then apply spatial affine map.
% coef_* are (S x (n+1)) with columns high→low powers.
S = size(coef_tau,1);
n = size(coef_tau,2) - 1;
powers_high_to_low = n:-1:0;            % column j has power = powers_high_to_low(j)
time_scale = alpha.^(-powers_high_to_low); % α^{-i} for each power i

coef_t = coef_tau .* time_scale;        % broadcast across rows (τ → t)
coef_t = coef_t * beta2;                % scale
coef_t(:, end) = coef_t(:, end) + beta1; % add offset to constant term
end


% ======================= Core Solver (τ-domain) ===========================
function coef = solveMinsnap(waypoints, t_k, order, dmin, bc)
% Solve min-snap in nondimensional time (t_k ∈ [0,1]).
K = numel(waypoints);  S = K - 1;  n = order;
assert(n >= 2*dmin-1, 'Order must be >= 2*dmin-1.');

% Hessian over τ using segment local time τ_local ∈ [0, dur], where dur = t_{k+1}-t_k
H = zeros(S*(n+1));
for s = 1:S
    dur = t_k(s+1) - t_k(s);
    Hs  = intQ(n, dmin, 0, dur);     % integrate over local τ
    ixs = (s-1)*(n+1)+(1:(n+1));
    H(ixs,ixs) = H(ixs,ixs) + Hs;
end

rows = {}; vals = [];

% Position constraints at each segment start/end
for k = 1:K-1
    dur = t_k(k+1)-t_k(k);
    r = polyDerRow(n, 0, 0);  row = zeros(1,S*(n+1)); row((k-1)*(n+1)+(1:(n+1))) = r;
    rows{end+1} = row; vals(end+1,1) = waypoints(k);
    r = polyDerRow(n, 0, dur); row = zeros(1,S*(n+1)); row((k-1)*(n+1)+(1:(n+1))) = r;
    rows{end+1} = row; vals(end+1,1) = waypoints(k+1);
end

% C^3 continuity (v,a,j) at internal knots
for k = 2:K-1
    dur_prev = t_k(k)-t_k(k-1);
    for rder = 1:3
        row = zeros(1,S*(n+1));
        row((k-2)*(n+1)+(1:(n+1))) = polyDerRow(n, rder, dur_prev);
        row((k-1)*(n+1)+(1:(n+1))) = -polyDerRow(n, rder, 0);
        rows{end+1} = row; vals(end+1,1) = 0;
    end
end

% Boundary derivative constraints at ends (per bc flags)
derMask = [0 1 2 3];
if bc.use_start(2), r=polyDerRow(n,1,0);  row=zeros(1,S*(n+1)); row(1:(n+1))=r; rows{end+1}=row; vals(end+1,1)=0; end
if bc.use_start(3), r=polyDerRow(n,2,0);  row=zeros(1,S*(n+1)); row(1:(n+1))=r; rows{end+1}=row; vals(end+1,1)=0; end
if bc.use_start(4), r=polyDerRow(n,3,0);  row=zeros(1,S*(n+1)); row(1:(n+1))=r; rows{end+1}=row; vals(end+1,1)=0; end
durS = t_k(end)-t_k(end-1);
if bc.use_end(2), r=polyDerRow(n,1,durS); row=zeros(1,S*(n+1)); row((S-1)*(n+1)+(1:(n+1)))=r; rows{end+1}=row; vals(end+1,1)=0; end
if bc.use_end(3), r=polyDerRow(n,2,durS); row=zeros(1,S*(n+1)); row((S-1)*(n+1)+(1:(n+1)))=r; rows{end+1}=row; vals(end+1,1)=0; end
if bc.use_end(4), r=polyDerRow(n,3,durS); row=zeros(1,S*(n+1)); row((S-1)*(n+1)+(1:(n+1)))=r; rows{end+1}=row; vals(end+1,1)=0; end

Aeq = vertcat(rows{:}); beq = vals;

H = (H+H')/2;  H = H + 1e-10*eye(size(H));  % small ridge
c  = solveQP(H, Aeq, beq);

coef = reshape(c, [n+1, S]).';
end


% ======================= Utilities (unchanged) ============================
function H = intQ(n, r, t0, tf)
H = zeros(n+1);
for i = 0:n
    for j = 0:n
        if i>=r && j>=r
            ai = fallingFactorial(i, r);
            aj = fallingFactorial(j, r);
            pow = (i-r)+(j-r);
            Hij = ai*aj * (tf^(pow+1) - t0^(pow+1)) / (pow+1);
        else
            Hij = 0;
        end
        Hi = n - i + 1; Hj = n - j + 1;
        H(Hi, Hj) = Hij;
    end
end
end

function r = fallingFactorial(k, m)
if m==0, r=1; return; end
if k<m,  r=0; return; end
r = prod(k:-1:(k-m+1));
end

function row = polyDerRow(n, r, t)
row = zeros(1, n+1);
for i = 0:n
    if i>=r
        coef = factorial(i)/factorial(i-r) * t^(i-r);
    else
        coef = 0;
    end
    idx = n - i + 1;   % high→low
    row(idx) = coef;
end
end

function c = solveQP(H, Aeq, beq)
opts = [];
try
    if exist('optimoptions','file')==2
        opts = optimoptions('quadprog','Display','none','Algorithm','interior-point-convex');
    end
    c = quadprog(H, [], [], [], Aeq, beq, [], [], [], opts);
catch
    KKT = [H, Aeq'; Aeq, zeros(size(Aeq,1))];
    rhs = [zeros(size(H,1),1); beq];
    sol = KKT \ rhs;
    c = sol(1:size(H,1));
end
end

function [p, dp, ddp] = evalPiecewise(ts, tk, coef)
S = size(coef,1); n = size(coef,2)-1;
p = zeros(size(ts)); dp = p; ddp = p;
for k = 1:numel(ts)
    t = ts(k);
    if t >= tk(end), s = S; tau = tk(end)-tk(end-1);
    else
        s = find(t >= tk(1:end-1) & t < tk(2:end), 1, 'last');
        if isempty(s), s = 1; end
        tau = t - tk(s);
    end
    cseg = coef(s,:);
    p(k)   = polyval(cseg, tau);
    dcseg  = polyder(cseg);
    ddcseg = polyder(dcseg);
    dp(k)  = polyval(dcseg, tau);
    ddp(k) = polyval(ddcseg, tau);
end
end
