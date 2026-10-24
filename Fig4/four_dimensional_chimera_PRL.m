%% four_dimensional_chimera_PRL_rho_fixedsphere.m
% Four-dimensional extension of the fifth-order Kuramoto model.
%
% This version uses the instantaneous mean field rho as the polar axis.
%
% Main changes relative to the original P-axis version:
%   1. After each RK4 step, the whole configuration is rotated so that
%      the instantaneous rho points along e4 = (0,0,0,1).
%   2. R1(t), R2(t) are computed in rho^\perp.
%   3. In panels (b) and (c), oscillators are mapped onto the fixed spheres
%      corresponding to polar angles 2*pi/5 and 4*pi/5 with respect to rho.
%      That is, we preserve only the azimuthal direction in rho^\perp and
%      place each point on the sphere of radius sin(theta_alpha).
%
% No toolbox is required.

clear; clc;

%% -------------------- parameters --------------------

N = 100000;
K = 50;
dt = 0.001;
T = 10;

gammaC1 = 0.10;
gammaC2 = 0.10;

epsilon = 1e-3;
seed = 20260917;

sampleDt = 0.02;          % only controls stored/plot data, not integration
progressEvery = 0.05;     % print every 5%

theta1 = 2*pi/5;
theta2 = 4*pi/5;

rng(seed,'twister');

n1 = round(gammaC1*N);
n2 = round(gammaC2*N);
nP = N - n1 - n2;

if nP < 1
    error('gammaC1 + gammaC2 must be < 1.');
end

idx1 = 1:n1;
idx2 = n1 + (1:n2);
idxP = n1+n2 + (1:nP);

fprintf('4D fifth-order simulation (rho-axis version)\n');
fprintf('N=%d, K=%g, dt=%g, T=%g\n',N,K,dt,T);
fprintf('C1=%d, C2=%d, P=%d\n',n1,n2,nP);

%% -------------------- initial condition on S^3 --------------------
% We first construct the same type of initial condition as before:
% P near e4, C1 at polar angle theta1, C2 at polar angle theta2,
% then add a small Cartesian perturbation and normalize.
%
% After that, we rotate the whole state so that the instantaneous rho
% points along e4.

sigma = zeros(N,4);

u1 = randn(n1,3);
u1 = normalizeRows(u1);
sigma(idx1,1:3) = sin(theta1)*u1;
sigma(idx1,4)   = cos(theta1);

u2 = randn(n2,3);
u2 = normalizeRows(u2);
sigma(idx2,1:3) = sin(theta2)*u2;
sigma(idx2,4)   = cos(theta2);

sigma(idxP,4) = 1;

% Small independent Cartesian perturbation, followed by normalization.
sigma = sigma + epsilon*(2*rand(N,4)-1);
sigma = normalizeRows(sigma);

% Rotate the whole configuration so that rho is mapped to e4.
sigma = rotateMeanFieldToE4(sigma);

%% -------------------- storage --------------------

nSteps = round(T/dt);
sampleEvery = max(1,round(sampleDt/dt));
nSave = floor(nSteps/sampleEvery)+1;

tSave  = nan(nSave,1);
R1Save = nan(nSave,1);
R2Save = nan(nSave,1);
polar1 = nan(nSave,1);
polar2 = nan(nSave,1);
pCoh   = nan(nSave,1);

isave = 1;
[tSave(isave),R1Save(isave),R2Save(isave), ...
    polar1(isave),polar2(isave),pCoh(isave)] = ...
    observe4D_rho(sigma,idx1,idx2,idxP,0,theta1,theta2);

%% -------------------- RK4 integration --------------------

fprintf('Starting integration: %d RK4 steps...\n',nSteps);
tic;
nextProgress = progressEvery;

for step = 1:nSteps
    k1 = rhs5D(sigma,K);

    s2 = normalizeRows(sigma + 0.5*dt*k1);
    k2 = rhs5D(s2,K);

    s3 = normalizeRows(sigma + 0.5*dt*k2);
    k3 = rhs5D(s3,K);

    s4 = normalizeRows(sigma + dt*k3);
    k4 = rhs5D(s4,K);

    sigma = sigma + (dt/6)*(k1 + 2*k2 + 2*k3 + k4);
    sigma = normalizeRows(sigma);

    % Because the dynamics is globally rotationally equivariant,
    % this common orthogonal rotation removes only the global motion.
    % Here the reference axis is the instantaneous mean field rho.
    sigma = rotateMeanFieldToE4(sigma);

    if mod(step,sampleEvery)==0
        isave = isave + 1;
        t = step*dt;
        [tSave(isave),R1Save(isave),R2Save(isave), ...
            polar1(isave),polar2(isave),pCoh(isave)] = ...
            observe4D_rho(sigma,idx1,idx2,idxP,t,theta1,theta2);
    end

    frac = step/nSteps;
    if frac + 1e-12 >= nextProgress
        fprintf('%5.1f %%   t=%7.3f   elapsed=%7.1f s\n', ...
            100*frac,step*dt,toc);
        nextProgress = nextProgress + progressEvery;
    end
end

elapsed = toc;
fprintf('Finished in %.1f s.\n',elapsed);

% trim storage in case round-off changed the final sample count
tSave  = tSave(1:isave);
R1Save = R1Save(1:isave);
R2Save = R2Save(1:isave);
polar1 = polar1(1:isave);
polar2 = polar2(1:isave);
pCoh   = pCoh(1:isave);

%% -------------------- final mapped points on fixed spheres --------------------
% In the co-rotating frame rho = e4. Therefore rho^\perp is exactly the
% first-three-coordinate subspace.
%
% For visualization, we do NOT keep the actual projected radii.
% Instead, for each oscillator we keep its azimuthal direction in rho^\perp
% and map it to the fixed sphere corresponding to the target polar angle.
%
% So the displayed radii are exactly:
%   r1 = sin(theta1), r2 = sin(theta2).

V1 = sigma(idx1,1:3);
V2 = sigma(idx2,1:3);

r1Theory = sin(theta1);
r2Theory = sin(theta2);

Y1 = mapToFixedSphere(V1,r1Theory);
Y2 = mapToFixedSphere(V2,r2Theory);

fprintf('\nFinal observables:\n');
fprintf('R1(T) = %.6f\n',R1Save(end));
fprintf('R2(T) = %.6f\n',R2Save(end));
fprintf('C1 polar RMS deviation = %.3e rad\n',polar1(end));
fprintf('C2 polar RMS deviation = %.3e rad\n',polar2(end));
fprintf('P-component coherence  = %.6f\n',pCoh(end));

%% -------------------- paper-style three-panel figure --------------------
% Compact journal-style layout.
% Font: Times New Roman
% No titles; curves are labeled directly rather than using a legend.

fontName = 'Times New Roman';
fontSize = 40;

fig = figure('Color','w','Position',[100 60 1100 900]);

% Manual axes positions for a compact layout.
% Position = [left bottom width height], normalized figure units.
ax1 = axes(fig,'Position',[0.20 0.61 0.60 0.31]);
ax2 = axes(fig,'Position',[0.055 0.055 0.445 0.50]);
ax3 = axes(fig,'Position',[0.500 0.055 0.445 0.50]);

%% ---------------- (a): time evolution ----------------

h1 = plot(ax1,tSave,R1Save,'LineWidth',3.0); hold(ax1,'on');
h2 = plot(ax1,tSave,R2Save,'LineWidth',3.0);

xlabel(ax1,'t', ...
    'FontName',fontName,'FontSize',fontSize,'FontAngle','italic');

ylabel(ax1,'$|\mathbf{Z}_{\alpha}|$', ...
    'Interpreter','latex', ...
    'FontName',fontName, ...
    'FontSize',fontSize);

xlim(ax1,[0 T]);
ylim(ax1,[0 1.02]);

set(ax1, ...
    'FontName',fontName, ...
    'FontSize',fontSize, ...
    'LineWidth',1.6, ...
    'TickLabelInterpreter','tex');

box(ax1,'on');

% Professional panel label: just outside the upper-left corner.
text(ax1,-0.13,1.08,'(a)', ...
    'Units','normalized', ...
    'FontName',fontName, ...
    'FontSize',fontSize, ...
    'FontWeight','normal', ...
    'HorizontalAlignment','left', ...
    'VerticalAlignment','top', ...
    'Clipping','off');

% Direct curve labels near the right-hand side.
labelFraction = 0.82;
xLabel = labelFraction*T;
R1Label = interp1(tSave,R1Save,xLabel,'linear','extrap');
R2Label = interp1(tSave,R2Save,xLabel,'linear','extrap');

% Small vertical offsets prevent the text from sitting exactly on the lines.
dy = 0.025;

text(ax1,xLabel,R1Label+dy,'$|\mathbf{Z}_1|$', ...
    'Interpreter','latex', ...
    'FontName',fontName, ...
    'FontSize',fontSize, ...
    'FontAngle','italic', ...
    'Color',h1.Color, ...
    'HorizontalAlignment','left', ...
    'VerticalAlignment','bottom');

text(ax1,xLabel,R2Label-dy,'$|\mathbf{Z}_2|$', ...
    'Interpreter','latex', ...
    'FontName',fontName, ...
    'FontSize',fontSize, ...
    'FontAngle','italic', ...
    'Color',h2.Color, ...
    'HorizontalAlignment','left', ...
    'VerticalAlignment','top');

%% ---------------- (b): C1 fixed sphere ----------------

drawProjectedSphere(ax2,r1Theory);
hold(ax2,'on');
scatter3(ax2,Y1(:,1),Y1(:,2),Y1(:,3),8,h1.Color,'filled');

formatProjectionAxisCompact(ax2);

text(ax2,-0.02,1.02,'(b)', ...
    'Units','normalized', ...
    'FontName',fontName, ...
    'FontSize',fontSize, ...
    'FontWeight','normal', ...
    'HorizontalAlignment','left', ...
    'VerticalAlignment','top', ...
    'Clipping','off');

% Add the component label using the same color as the R_1 curve.
text(ax2,0.50,0.08,'C_1', ...
    'Units','normalized', ...
    'Interpreter','tex', ...
    'FontName',fontName, ...
    'FontSize',fontSize, ...
    'FontAngle','italic', ...
    'FontWeight','normal', ...
    'Color',h1.Color, ...
    'HorizontalAlignment','center', ...
    'VerticalAlignment','middle');

%% ---------------- (c): C2 fixed sphere ----------------

drawProjectedSphere(ax3,r2Theory);
hold(ax3,'on');
scatter3(ax3,Y2(:,1),Y2(:,2),Y2(:,3),12,h2.Color,'filled');

formatProjectionAxisCompact(ax3);

text(ax3,-0.02,1.02,'(c)', ...
    'Units','normalized', ...
    'FontName',fontName, ...
    'FontSize',fontSize, ...
    'FontWeight','normal', ...
    'HorizontalAlignment','left', ...
    'VerticalAlignment','top', ...
    'Clipping','off');

% Add the component label using the same color as the R_2 curve.
text(ax3,0.50,0.08,'C_2', ...
    'Units','normalized', ...
    'Interpreter','tex', ...
    'FontName',fontName, ...
    'FontSize',fontSize, ...
    'FontAngle','italic', ...
    'FontWeight','normal', ...
    'Color',h2.Color, ...
    'HorizontalAlignment','center', ...
    'VerticalAlignment','middle');

%% -------------------- optional export --------------------
% Uncomment if needed.
% savefig(fig,'Chimera4D_rho_fixedsphere.fig');
% exportgraphics(fig,'Chimera4D_rho_fixedsphere.pdf','ContentType','vector');
% exportgraphics(fig,'Chimera4D_rho_fixedsphere.png','Resolution',400);
% save('Chimera4D_rho_fixedsphere_data.mat', ...
%     'tSave','R1Save','R2Save','polar1','polar2','pCoh', ...
%     'sigma','Y1','Y2','idx1','idx2','idxP', ...
%     'N','K','dt','T','gammaC1','gammaC2','epsilon','seed');

fprintf(['\nSuggested output names:\n' ...
    '  Chimera4D_rho_fixedsphere.fig\n' ...
    '  Chimera4D_rho_fixedsphere.pdf\n' ...
    '  Chimera4D_rho_fixedsphere.png\n' ...
    '  Chimera4D_rho_fixedsphere_data.mat\n']);

%% ========================================================================
% Local functions
% ========================================================================

function ds = rhs5D(sigma,K)
% Fifth-harmonic mean-field vector field in arbitrary dimension:
%
% ds_i/dt = K [16 a_i^4 - 12 R^2 a_i^2 + R^4] [rho - a_i sigma_i],
%
% rho = <sigma>, R=|rho|, a_i=sigma_i.rho.
%
% On any invariant great circle this reduces to the fifth-harmonic
% Kuramoto interaction K R^5 sin[5(Psi-theta_i)].
    rho = mean(sigma,1);
    R2 = sum(rho.^2);
    a = sigma*rho.';
    coeff = K*(16*a.^4 - 12*R2*a.^2 + R2^2);
    tangent = rho - a.*sigma;
    ds = coeff.*tangent;
end

function X = normalizeRows(X)
    nr = sqrt(sum(X.^2,2));
    if any(~isfinite(nr)) || any(nr < 1e-14)
        error('Invalid vector encountered during normalization.');
    end
    X = X./nr;
end

function sigmaRot = rotateReferenceToE4(sigma,p)
% Minimal proper rotation in R^4 that maps p to e4.
% The same rotation is applied to every oscillator.
%
% For unit vectors u,v with u ~= -v, the matrix
% Q = I + A + A^2/(1+u.v), A = v*u' - u*v'
% maps u to v and acts as identity outside span(u,v).
    u = p(:);
    u = u/norm(u);
    v = [0;0;0;1];
    c = dot(u,v);
    D = 4;

    if c > 1-1e-14
        Q = eye(D);
    elseif c < -1+1e-12
        % Rare antipodal case: construct a pi rotation in a plane
        % containing u and one orthogonal unit vector.
        [~,k] = min(abs(u));
        w = zeros(D,1);
        w(k) = 1;
        w = w - u*dot(u,w);
        w = w/norm(w);
        Q = eye(D) - 2*(u*u' + w*w');
    else
        A = v*u' - u*v';
        Q = eye(D) + A + (A*A)/(1+c);
    end

    sigmaRot = sigma*Q.';
    sigmaRot = normalizeRows(sigmaRot);
end

function sigmaRot = rotateMeanFieldToE4(sigma)
% Rotate the whole configuration so that the instantaneous mean field rho
% points to e4.
    rho = mean(sigma,1);
    nr = norm(rho);
    if ~isfinite(nr) || nr < 1e-14
        error('Mean field rho is too small to define the reference axis.');
    end
    sigmaRot = rotateReferenceToE4(sigma,rho);
end

function [t,R1,R2,pol1,pol2,pcoh] = ...
    observe4D_rho(sigma,idx1,idx2,idxP,t,theta1,theta2)
% Orthogonal projection onto rho^\perp; rho=e4 in this frame.
    Y1 = sigma(idx1,1:3);
    Y2 = sigma(idx2,1:3);

    R1 = directionalOrder(Y1);
    R2 = directionalOrder(Y2);

    % Diagnostics: verify that populations remain near their expected
    % fixed polar components relative to the P population.
    pMean = mean(sigma(idxP,:),1);
    pcoh = norm(pMean);

    p = pMean / pcoh;

    c1 = max(-1,min(1,sigma(idx1,:)*p.'));
    c2 = max(-1,min(1,sigma(idx2,:)*p.'));

    th1 = acos(c1);
    th2 = acos(c2);

    pol1 = sqrt(mean((th1-theta1).^2));
    pol2 = sqrt(mean((th2-theta2).^2));
end

function R = directionalOrder(Y)
% R = | < y_j/|y_j| > |.
% This measures angular localization in rho^\perp and is
% insensitive to small radial/polar deviations.
    rr = sqrt(sum(Y.^2,2));
    good = rr > 1e-12;

    if ~any(good)
        R = NaN;
        return;
    end

    U = Y(good,:)./rr(good);
    R = norm(mean(U,1));
end

function Ymap = mapToFixedSphere(Y,rTarget)
% Preserve the direction in rho^\perp and map each point to the sphere
% of radius rTarget.
    rr = sqrt(sum(Y.^2,2));

    if any(rr < 1e-12)
        error(['Some projected vectors are too small to define an ', ...
            'azimuthal direction for fixed-sphere mapping.']);
    end

    Ymap = rTarget * (Y ./ rr);
end

function drawProjectedSphere(ax,r)
    [xs,ys,zs] = sphere(50);
    surf(ax,r*xs,r*ys,r*zs, ...
        'FaceAlpha',0.055, ...
        'EdgeAlpha',0.08, ...
        'HandleVisibility','off');
    hold(ax,'on');
end

function formatProjectionAxisCompact(ax)
% Keep the displayed range fixed and hide the coordinate system.
    axis(ax,'equal');
    xlim(ax,[-1 1]);
    ylim(ax,[-1 1]);
    zlim(ax,[-1 1]);
    axis(ax,'vis3d');
    view(ax,35,24);
    axis(ax,'off');
end