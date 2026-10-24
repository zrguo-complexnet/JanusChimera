%% three_dimensional_chimera_rho_1x3_with_Pleft.m
% 3D fifth-order Kuramoto model on S^2.
%
% One horizontal figure:
% (a) Oscillator positions on the sphere
% (b) Azimuthal histograms
% (c) Azimuthal order-parameter evolution
%
% Frame choice:
%   1. First align the instantaneous global mean field rho with +z.
%   2. Then rotate about z so that the P-component deflection direction
%      points to the left (target azimuth phiTarget = pi by default).
%
% Histogram convention:
%   psi = 0 is taken along the P deflection direction.
%
% Times New Roman, 40 pt.
% Sphere wireframe ON; Cartesian axes OFF.
% No figure export.

clear; clc; close all;

%% ============================================================
% Parameters
% =============================================================

N = 10000;
K = 50;
n = 5;

dt = 0.001;
T = 20;

gamma_C1 = 0.2;
gamma_C2 = 0.05;

perturbation_strength = 1e-3;
seed = 20261002;

animationEvery = 10;
saveData = true;
saveFolder = 'results_figures_rho';

fontName = 'Times New Roman';
fontSize = 40;

colorP   = [0.25 0.25 0.25];
colorC1  = [0.85 0.20 0.20];
colorC2  = [0.20 0.30 0.90];
colorRho = [0.55 0.20 0.55];

theta1 = 2*pi/n;
theta2 = 4*pi/n;

numSteps = round(T/dt);
time_vec = (0:numSteps)'*dt;

assert(n==5,'The vector field below implements n=5.');
assert(dt>0 && T>0 && abs(numSteps*dt-T)<1e-10, ...
    'T must be a positive multiple of dt.');
assert(gamma_C1>0 && gamma_C2>0 && gamma_C1+gamma_C2<1, ...
    'Invalid population fractions.');

%% ============================================================
% Initial conditions
% =============================================================

rng(seed,'twister');

[sigma,idxC1,idxC2,idxP] = ...
    generate_sigma(N,n,gamma_C1,gamma_C2);

sigma = normalizeRows( ...
    sigma + perturbation_strength*randn(N,3));

% First align rho to +z, then rotate around z so that P points left.
sigma = alignFrame(sigma,idxP);

%% ============================================================
% Storage
% =============================================================

rho_magnitudes = nan(numSteps+1,1);
Z1_abs = nan(numSteps+1,1);
Z2_abs = nan(numSteps+1,1);

[rho_magnitudes(1),psi_C1,psi_C2,Z1_abs(1),Z2_abs(1),phiP0] = ...
    observe(sigma,idxC1,idxC2,idxP);

%% ============================================================
% One horizontal figure
% =============================================================

fig = figure( ...
    'Color','w', ...
    'Units','pixels', ...
    'Position',[30 60 1600 650], ...
    'Name','Rho-axis chimera');

% Normalized figure coordinates [left bottom width height]
posA = [0.025 0.150 0.310 0.715];
posB = [0.420 0.225 0.225 0.535];
posC = [0.755 0.225 0.215 0.535];

%% ============================================================
% (a) Actual oscillator positions on S^2
% =============================================================

axPos = axes('Parent',fig,'Position',posA);
hold(axPos,'on');

% Sphere wireframe
[sx,sy,sz] = sphere(26);
mesh(axPos,sx,sy,sz, ...
    'FaceColor','none', ...
    'EdgeColor',[0.72 0.72 0.72], ...
    'EdgeAlpha',0.55, ...
    'LineWidth',0.55, ...
    'HandleVisibility','off');

% Fixed-polar-angle reference circles about rho
[rx1,ry1,rz1] = referenceRing(theta1,360);
[rx2,ry2,rz2] = referenceRing(theta2,360);

plot3(axPos,rx1,ry1,rz1,'-', ...
    'Color',[1.0 0.65 0.65], ...
    'LineWidth',1.5, ...
    'HandleVisibility','off');

plot3(axPos,rx2,ry2,rz2,'-', ...
    'Color',[0.62 0.72 1.0], ...
    'LineWidth',1.5, ...
    'HandleVisibility','off');

% Oscillators
hP = scatter3(axPos, ...
    sigma(idxP,1),sigma(idxP,2),sigma(idxP,3), ...
    24, ...
    'MarkerEdgeColor',colorP, ...
    'MarkerFaceColor','none', ...
    'LineWidth',0.8);

hC1 = scatter3(axPos, ...
    sigma(idxC1,1),sigma(idxC1,2),sigma(idxC1,3), ...
    35, ...
    'MarkerEdgeColor',colorC1, ...
    'MarkerFaceColor','none', ...
    'LineWidth',1.1);

hC2 = scatter3(axPos, ...
    sigma(idxC2,1),sigma(idxC2,2),sigma(idxC2,3), ...
    35, ...
    'MarkerEdgeColor',colorC2, ...
    'MarkerFaceColor','none', ...
    'LineWidth',1.1);

% Total mean-field vector rho
rhoVec = mean(sigma,1);
%% Manually draw rho arrow

rhoVec = mean(sigma,1);
L = norm(rhoVec);

% Arrowhead orientation about the z axis
arrowAngle = pi/4;    % 45 degrees; adjustable

% Arrowhead dimensions
headLength = 0.20;
headWidth  = 0.06;

dx = headWidth*cos(arrowAngle);
dy = headWidth*sin(arrowAngle);

% Arrow shaft
hRho = plot3(axPos, ...
    [0 0],[0 0],[0 L], ...
    '-', ...
    'Color',colorRho, ...
    'LineWidth',4, ...
    'HandleVisibility','off');

% V-shaped arrowhead
hRhoHead = plot3(axPos, ...
    [dx 0 -dx], ...
    [dy 0 -dy], ...
    [L-headLength L L-headLength], ...
    '-', ...
    'Color',colorRho, ...
    'LineWidth',4, ...
    'HandleVisibility','off');

hRhoLabel = text(axPos, ...
    0.10,0.04,0.60*max(rhoVec(3),0.15),'\rho', ...
    'Interpreter','tex', ...
    'FontName',fontName, ...
    'FontSize',fontSize, ...
    'Color',colorRho, ...
    'Clipping','off', ...
    'HorizontalAlignment','left');

axis(axPos,'equal');
axis(axPos,'vis3d');
xlim(axPos,[-1 1]);
ylim(axPos,[-1 1]);
zlim(axPos,[-1 1]);

view(axPos,35,25);

% Hide Cartesian axes but keep sphere wireframe.
axis(axPos,'off');

%% ============================================================
% (b) Azimuthal histograms
% =============================================================

axHist = axes('Parent',fig,'Position',posB);
hold(axHist,'on');

edges = linspace(-pi,pi,37);

hHist1 = histogram(axHist,psi_C1,edges, ...
    'Normalization','probability', ...
    'FaceColor',colorC1, ...
    'FaceAlpha',0.45, ...
    'EdgeColor','none');

hHist2 = histogram(axHist,psi_C2,edges, ...
    'Normalization','probability', ...
    'FaceColor',colorC2, ...
    'FaceAlpha',0.45, ...
    'EdgeColor','none');

xlim(axHist,[-pi pi]);
xticks(axHist,[-pi 0 pi]);
xticklabels(axHist,{'-\pi','0','\pi'});

ylim(axHist,[0 1]);
yticks(axHist,[0 0.5 1]);

xlabel(axHist,'\psi', ...
    'Interpreter','tex', ...
    'FontName',fontName, ...
    'FontSize',fontSize);

ylabel(axHist,'Probability', ...
    'FontName',fontName, ...
    'FontSize',fontSize);

formatAxis(axHist,fontName,fontSize);

legend(axHist,[hHist1,hHist2],{'C_1','C_2'}, ...
    'Location','northeast', ...
    'Orientation','horizontal', ...
    'Interpreter','tex', ...
    'FontName',fontName, ...
    'FontSize',fontSize, ...
    'Box','off', ...
    'AutoUpdate','off');

%% ============================================================
% (c) Azimuthal order-parameter evolution
% =============================================================

axZ = axes('Parent',fig,'Position',posC);
hold(axZ,'on');

hZ1 = plot(axZ,time_vec(1),Z1_abs(1),'-', ...
    'Color',colorC1, ...
    'LineWidth',3);

hZ2 = plot(axZ,time_vec(1),Z2_abs(1),'-', ...
    'Color',colorC2, ...
    'LineWidth',3);

xlim(axZ,[0 T]);
ylim(axZ,[0 1.02]);
xticks(axZ,linspace(0,T,3));

xlabel(axZ,'t', ...
    'FontName',fontName, ...
    'FontSize',fontSize, ...
    'FontAngle','italic');

ylabel(axZ,'|Z_\alpha|', ...
    'Interpreter','tex', ...
    'FontName',fontName, ...
    'FontSize',fontSize);

formatAxis(axZ,fontName,fontSize);

legend(axZ,[hZ1,hZ2],{'|Z_1|','|Z_2|'}, ...
    'Location','northeast', ...
    'Interpreter','tex', ...
    'FontName',fontName, ...
    'FontSize',fontSize, ...
    'Box','off', ...
    'AutoUpdate','off');

%% ============================================================
% Figure-level labels
% =============================================================

addFigureLabel(fig,[0.030 0.905 0.065 0.070], ...
    '(a)',[0 0 0],fontName,fontSize,'left');

addFigureLabel(fig,[0.407 0.905 0.065 0.070], ...
    '(b)',[0 0 0],fontName,fontSize,'left');

addFigureLabel(fig,[0.742 0.905 0.065 0.070], ...
    '(c)',[0 0 0],fontName,fontSize,'left');

% Sphere-panel component labels
addFigureLabel(fig,[0.060 0.040 0.065 0.070], ...
    'P',colorP,fontName,fontSize,'center');

addFigureLabel(fig,[0.147 0.040 0.075 0.070], ...
    'C_1',colorC1,fontName,fontSize,'center');

addFigureLabel(fig,[0.237 0.040 0.075 0.070], ...
    'C_2',colorC2,fontName,fontSize,'center');


drawnow;

%% ============================================================
% Time integration
% =============================================================

fprintf('Starting simulation...\n');

startClock = tic;
nextProgress = 0.05;

for step = 1:numSteps

    % Euler integration
    sigma = normalizeRows( ...
        sigma + dt*rhs5D(sigma,K));

    % First rho -> +z, then rotate around z so that P points left.
    sigma = alignFrame(sigma,idxP);

    % Observables
    [rho_magnitudes(step+1),psi_C1,psi_C2, ...
        Z1_abs(step+1),Z2_abs(step+1),phiP] = ...
        observe(sigma,idxC1,idxC2,idxP);

    %% Update panels
    if mod(step,animationEvery)==0 || step==numSteps

        % (a) Update oscillator positions
        set(hP, ...
            'XData',sigma(idxP,1), ...
            'YData',sigma(idxP,2), ...
            'ZData',sigma(idxP,3));

        set(hC1, ...
            'XData',sigma(idxC1,1), ...
            'YData',sigma(idxC1,2), ...
            'ZData',sigma(idxC1,3));

        set(hC2, ...
            'XData',sigma(idxC2,1), ...
            'YData',sigma(idxC2,2), ...
            'ZData',sigma(idxC2,3));

        % Update rho vector
        rhoVec = mean(sigma,1);
        L = norm(rhoVec);
        
        % Update shaft length
        set(hRho,'ZData',[0 L]);
        
        % Update arrowhead position
        set(hRhoHead, ...
            'ZData',[L-headLength L L-headLength]);

        set(hRhoLabel, ...
            'Position',[0.10,0.04,0.60*max(rhoVec(3),0.15)]);

        hHist1.Data = psi_C1;
        hHist2.Data = psi_C2;

        % (c) Update order parameters
        set(hZ1, ...
            'XData',time_vec(1:step+1), ...
            'YData',Z1_abs(1:step+1));

        set(hZ2, ...
            'XData',time_vec(1:step+1), ...
            'YData',Z2_abs(1:step+1));

        drawnow limitrate;
    end

    %% Progress
    if step/numSteps + 1e-12 >= nextProgress
        fprintf('%5.1f %%   t=%7.3f   elapsed=%6.1f s\n', ...
            100*step/numSteps,step*dt,toc(startClock));
        nextProgress = nextProgress + 0.05;
    end
end

drawnow;

fprintf('Finished in %.1f s.\n',toc(startClock));
fprintf('|Z1| = %.6f\n',Z1_abs(end));
fprintf('|Z2| = %.6f\n',Z2_abs(end));
fprintf('|rho| = %.6f\n',rho_magnitudes(end));

%% ============================================================
% Save numerical data only
% =============================================================

if saveData
    if ~exist(saveFolder,'dir')
        mkdir(saveFolder);
    end

    save(fullfile(saveFolder,'simulation_data_rho_1x3.mat'), ...
        'sigma','time_vec','rho_magnitudes', ...
        'Z1_abs','Z2_abs', ...
        'psi_C1','psi_C2', ...
        'idxP','idxC1','idxC2', ...
        'N','K','n','dt','T', ...
        'gamma_C1','gamma_C2', ...
        'seed','perturbation_strength');
end

%% ============================================================
% Local functions
% =============================================================

function ds = rhs5D(sigma,K)
    rho = mean(sigma,1);
    R2 = sum(rho.^2);
    a = sigma*rho.';
    coeff = K*(16*a.^4 - 12*R2*a.^2 + R2^2);
    ds = coeff.*(rho - a.*sigma);
end

function X = normalizeRows(X)
    d = sqrt(sum(X.^2,2));
    if any(~isfinite(d)) || any(d<=1e-14)
        error('Invalid vector encountered in normalization.');
    end
    X = X./d;
end

function X = alignFrame(X,idxP)
    % Step 1: align rho with +z
    X = rotateToRhoAxis(X);

    % Step 2: rotate around z so that P deflection points left
    X = rotatePLeft(X,idxP);
end

function X = rotateToRhoAxis(X)
    r = mean(X,1).';
    nr = norm(r);

    if ~isfinite(nr) || nr<=1e-12
        error('rho is too small to define an axis.');
    end

    u = r/nr;
    v = [0;0;1];

    c = max(-1,min(1,dot(u,v)));

    if c > 1-1e-14
        Q = eye(3);

    elseif c < -1+1e-12
        [~,k] = min(abs(u));
        w = zeros(3,1);
        w(k) = 1;
        w = w-u*dot(u,w);
        w = w/norm(w);
        Q = eye(3)-2*(u*u'+w*w');

    else
        A = v*u' - u*v';
        Q = eye(3) + A + (A*A)/(1+c);
    end

    X = X*Q.';
end

function X = rotatePLeft(X,idxP)
    % After rho has been aligned with +z, rotate around z so that
    % the projected P direction points toward the "left".
    %
    % You can tweak phiTarget if the visual "left" on your screen
    % is not ideal under the current camera view.
    p = mean(X(idxP,:),1);

    pxy = hypot(p(1),p(2));

    if pxy <= 1e-12
        return;
    end

    phiP = atan2(p(2),p(1));

    % Default target: negative x direction
    phiTarget = pi;

    delta = phiTarget - phiP;

    c = cos(delta);
    s = sin(delta);

    Rz = [ c -s  0;
           s  c  0;
           0  0  1 ];

    X = X*Rz.';
end

function [rhoAbs,psi1,psi2,Z1,Z2,phiP] = ...
    observe(sigma,idxC1,idxC2,idxP)

    rhoAbs = norm(mean(sigma,1));

    p = mean(sigma(idxP,:),1);
    pxy = hypot(p(1),p(2));

    if pxy <= 1e-12
        phiP = 0;
    else
        phiP = atan2(p(2),p(1));
    end

    % psi = 0 along the P deflection direction
    psi1 = relativeAzimuths(sigma(idxC1,:),phiP);
    psi2 = relativeAzimuths(sigma(idxC2,:),phiP);

    Z1 = abs(mean(exp(1i*psi1)));
    Z2 = abs(mean(exp(1i*psi2)));
end

function psi = relativeAzimuths(X,phi0)
    rr = hypot(X(:,1),X(:,2));
    if any(rr<=1e-12)
        error('Azimuth is undefined near the rho axis.');
    end

    raw = atan2(X(:,2),X(:,1)) - phi0;

    % Wrap to [-pi,pi]
    psi = atan2(sin(raw),cos(raw));
end

function [x,y,z] = referenceRing(theta,npts)
    phi = linspace(0,2*pi,npts);
    x = sin(theta)*cos(phi);
    y = sin(theta)*sin(phi);
    z = cos(theta)*ones(size(phi));
end

function [sigma,idxC1,idxC2,idxP] = ...
    generate_sigma(N,n,g1,g2)

    n1 = round(N*g1);
    n2 = round(N*g2);
    nP = N-n1-n2;

    assert(n1>0 && n2>0 && nP>0);

    idxC1 = 1:n1;
    idxC2 = n1+(1:n2);
    idxP  = n1+n2+(1:nP);

    sigma = zeros(N,3);

    phi1 = 2*pi*rand(n1,1);
    phi2 = 2*pi*rand(n2,1);

    th1 = 2*pi/n;
    th2 = 4*pi/n;

    sigma(idxC1,:) = [ ...
        sin(th1)*cos(phi1), ...
        sin(th1)*sin(phi1), ...
        cos(th1)*ones(n1,1)];

    sigma(idxC2,:) = [ ...
        sin(th2)*cos(phi2), ...
        sin(th2)*sin(phi2), ...
        cos(th2)*ones(n2,1)];

    sigma(idxP,:) = repmat([0 0 1],nP,1);
end

function formatAxis(ax,fontName,fontSize)
    set(ax, ...
        'FontName',fontName, ...
        'FontSize',fontSize, ...
        'LineWidth',1.8, ...
        'TickLabelInterpreter','tex', ...
        'Box','on', ...
        'XGrid','off', ...
        'YGrid','off', ...
        'ZGrid','off');
    grid(ax,'off');
end

function h = addFigureLabel(fig,pos,str,color,fontName,fontSize,alignment)
    h = annotation(fig,'textbox',pos, ...
        'String',str, ...
        'Interpreter','tex', ...
        'FontName',fontName, ...
        'FontSize',fontSize, ...
        'Color',color, ...
        'HorizontalAlignment',alignment, ...
        'VerticalAlignment','middle', ...
        'EdgeColor','none', ...
        'FitBoxToText','off');
end