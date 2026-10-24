%% three_dimensional_chimera_rho_initial_final_reference_style.m
% Three-panel S^2 fifth-order Kuramoto simulation:
% (a) initial oscillator positions at t=0
% (b) final oscillator positions at t=T
% (c) evolution of |Z_1| and |Z_2|
%
% Initial populations are uniformly sampled on polar bands:
%   P : theta in [0, pi/5]
%   C1: theta in [pi/5, 3*pi/5]
%   C2: theta in [3*pi/5, pi]
%
% Frame at BOTH snapshots:
%   rho -> +z; transverse P mean -> -x (when defined).
% Neither snapshot is rescaled or projected onto a reference ring.
% The mean-field rho arrow is drawn with its actual length.
% Sphere wireframe ON; Cartesian grid/axes OFF.
% No figure export; numerical data can be saved to MAT.

clear; clc; close all;

%% Parameters
N = 10000;
K = 50;
n = 5;
dt = 0.001;
T = 20;
gamma_C1 = 0.2;
gamma_C2 = 0.05;
perturbation_strength = 1e-3;
seed = 20261001;
samplingMode = 'surfaceArea';  % 'surfaceArea' or 'uniformTheta'
phiTarget = pi;               % P transverse mean points toward -x

showReferenceRings = true;   % Ideal rings at 2*pi/5 and 4*pi/5
saveData = true;
saveFolder = 'results_figures_rho';

fontName = 'Times New Roman';
fontSize = 40;
colorP   = [0.25 0.25 0.25];
colorC1  = [0.85 0.20 0.20];
colorC2  = [0.20 0.30 0.90];
colorRho = [0.55 0.20 0.55];

% Manually drawn rho arrow: its V-shaped head can be rotated about z.
arrowAngle = pi/4;
headLength = 0.20;
headWidth = 0.06;

numSteps = round(T/dt);
time_vec = (0:numSteps)'*dt;
assert(n==5,'The vector field below implements n=5.');
assert(dt>0 && T>0 && abs(numSteps*dt-T)<1e-10, ...
    'T must be a positive multiple of dt.');
assert(gamma_C1>0 && gamma_C2>0 && gamma_C1+gamma_C2<1, ...
    'All three populations must be nonempty.');

%% Initial state: broad polar bands
rng(seed,'twister');
[sigma,idxC1,idxC2,idxP,thetaInitial,phiInitial] = ...
    generate_sigma_bands(N,gamma_C1,gamma_C2,samplingMode);

% Set to zero if you want the sampled polar-band bounds to be exact
% before frame alignment.
sigma = normalizeRows(sigma+perturbation_strength*randn(N,3));
sigma = alignFrame(sigma,idxP,phiTarget);

% CRITICAL: copy the initial positions before integrating.
sigmaInitial = sigma;

%% Storage: both t=0 and t=T are included
rho_magnitudes = nan(numSteps+1,1);
Z1_abs = nan(numSteps+1,1);
Z2_abs = nan(numSteps+1,1);
[rho_magnitudes(1),Z1_abs(1),Z2_abs(1)] = ...
    observe(sigma,idxC1,idxC2,idxP);

%% Integrate (same Euler dynamics and rho/P gauge as before)
fprintf('Polar-band initialization: %s\n',samplingMode);
fprintf('N=%d, K=%g, dt=%g, T=%g\n',N,K,dt,T);
fprintf('P=%d, C1=%d, C2=%d\n',numel(idxP),numel(idxC1),numel(idxC2));
fprintf('Starting simulation...\n');
simClock = tic;
nextProgress = 0.05;

for step = 1:numSteps
    sigma = normalizeRows(sigma+dt*rhs5D(sigma,K));
    sigma = alignFrame(sigma,idxP,phiTarget);
    [rho_magnitudes(step+1),Z1_abs(step+1),Z2_abs(step+1)] = ...
        observe(sigma,idxC1,idxC2,idxP);

    if step/numSteps+1e-12>=nextProgress
        fprintf('%5.1f %%   t=%7.3f   elapsed=%6.1f s\n', ...
            100*step/numSteps,step*dt,toc(simClock));
        nextProgress = nextProgress+0.05;
    end
end

sigmaFinal = sigma;
fprintf('Finished in %.1f s.\n',toc(simClock));
fprintf('Initial: |rho|=%.6f, |Z1|=%.6f, |Z2|=%.6f\n', ...
    rho_magnitudes(1),Z1_abs(1),Z2_abs(1));
fprintf('Final:   |rho|=%.6f, |Z1|=%.6f, |Z2|=%.6f\n', ...
    rho_magnitudes(end),Z1_abs(end),Z2_abs(end));

%% Create 1x3 figure AFTER integration, to display true t=0 and t=T
fig = figure('Color','w','Units','pixels', ...
    'Position',[30 60 1600 650], ...
    'Name','Initial vs final states (rho-axis frame)');

% [left bottom width height], normalized to figure.
% The two sphere panels reuse the source sphere dimensions (0.310 x 0.715);
% the time-series panel uses the exact source posC.
% Only plotting parameters differ from initial_final_Z.m.
posInitial = [0.07 0.185 0.23 0.6];
posFinal   = [0.335 0.185 0.23 0.6];
posOrder   = [0.64 0.212692307692308 0.23 0.535];

% (a) and (b): identical 3D display settings
axInitial = drawSpherePanel(fig,posInitial,sigmaInitial, ...
    idxP,idxC1,idxC2,showReferenceRings, ...
    colorP,colorC1,colorC2,colorRho, ...
    fontName,fontSize,arrowAngle,headLength,headWidth);

axFinal = drawSpherePanel(fig,posFinal,sigmaFinal, ...
    idxP,idxC1,idxC2,showReferenceRings, ...
    colorP,colorC1,colorC2,colorRho, ...
    fontName,fontSize,arrowAngle,headLength,headWidth);

% (c) Evolution of azimuthal order parameter magnitudes
axZ = axes('Parent',fig,'Position',posOrder);
hold(axZ,'on');
hZ1 = plot(axZ,time_vec,Z1_abs,'-', ...
    'Color',colorC1,'LineWidth',3);
hZ2 = plot(axZ,time_vec,Z2_abs,'-', ...
    'Color',colorC2,'LineWidth',3);
xlim(axZ,[0 T]);
ylim(axZ,[0 1.02]);
xticks(axZ,linspace(0,T,3));
yticks(axZ,[0 0.5 1]);
xlabel(axZ,'t','FontName',fontName,'FontSize',fontSize, ...
    'FontAngle','italic');
set(axZ,'FontName',fontName,'FontSize',fontSize, ...
    'LineWidth',1.8,'Box','on','TickLabelInterpreter','tex');
grid(axZ,'off');
legend(axZ,[hZ1,hZ2],{'|{\bf\itZ}_1|','|{\bf\itZ}_2|'}, ...
    'Location','northeast','Interpreter','tex', ...
    'FontName',fontName,'FontSize',fontSize, ...
    'Box','off','AutoUpdate','off');

% Figure-level labels: stable locations independent of 3D camera angle.
addFigureLabel(fig,[0.030 0.905 0.065 0.070], ...
    '(a)',[0 0 0],fontName,fontSize,'left');
addFigureLabel(fig,[0.407 0.905 0.065 0.070], ...
    '(b)',[0 0 0],fontName,fontSize,'left');
addFigureLabel(fig,[0.742 0.905 0.065 0.070], ...
    '(c)',[0 0 0],fontName,fontSize,'left');

% Time labels below each sphere, separated from panel letters.
addFigureLabel(fig,[0.13 0.105 0.10 0.060], ...
    't = 0',[0 0 0],fontName,fontSize,'center');
addFigureLabel(fig,[0.525 0.105 0.10 0.060], ...
    't = T',[0 0 0],fontName,fontSize,'center');

% Component labels at the bottom of each sphere column.
for left = [0.025,0.420]
    addFigureLabel(fig,[left+0.035 0.030 0.065 0.065], ...
        'P',colorP,fontName,fontSize,'center');
    addFigureLabel(fig,[left+0.125 0.030 0.075 0.065], ...
        'C_1',colorC1,fontName,fontSize,'center');
    addFigureLabel(fig,[left+0.220 0.030 0.075 0.065], ...
        'C_2',colorC2,fontName,fontSize,'center');
end

drawnow;

%% Save numeric data only (NO PNG/PDF/FIG export)
if saveData
    if ~exist(saveFolder,'dir'), mkdir(saveFolder); end
    save(fullfile(saveFolder,'simulation_data_rho_initial_final_Z.mat'), ...
        'sigmaInitial','sigmaFinal', ...
        'time_vec','rho_magnitudes','Z1_abs','Z2_abs', ...
        'idxP','idxC1','idxC2','thetaInitial','phiInitial', ...
        'N','K','n','dt','T','gamma_C1','gamma_C2', ...
        'seed','perturbation_strength','samplingMode','phiTarget');
end

%% ============================================================
% Local functions
% =============================================================
function ax = drawSpherePanel(fig,pos,sigma,idxP,idxC1,idxC2, ...
    showReferenceRings,colorP,colorC1,colorC2,colorRho, ...
    fontName,fontSize,arrowAngle,headLength,headWidth)

    ax = axes('Parent',fig,'Position',pos);
    hold(ax,'on');

    % Physical unit-sphere wireframe, not Cartesian axes grids.
    [sx,sy,sz] = sphere(26);
    mesh(ax,sx,sy,sz,'FaceColor','none', ...
        'EdgeColor',[0.72 0.72 0.72], ...
        'EdgeAlpha',0.55,'LineWidth',0.55, ...
        'HandleVisibility','off');

    if showReferenceRings
        % Ideal fixed-angle rings; these are NOT the initial band edges.
        [x1,y1,z1] = referenceRing(2*pi/5,360);
        [x2,y2,z2] = referenceRing(4*pi/5,360);
        plot3(ax,x1,y1,z1,'-','Color',[1.0 0.65 0.65], ...
            'LineWidth',1.5,'HandleVisibility','off');
        plot3(ax,x2,y2,z2,'-','Color',[0.62 0.72 1.0], ...
            'LineWidth',1.5,'HandleVisibility','off');
    end

    scatter3(ax,sigma(idxP,1),sigma(idxP,2),sigma(idxP,3),24, ...
        'MarkerEdgeColor',colorP,'MarkerFaceColor','none', ...
        'LineWidth',0.8,'HandleVisibility','off');
    scatter3(ax,sigma(idxC1,1),sigma(idxC1,2),sigma(idxC1,3),35, ...
        'MarkerEdgeColor',colorC1,'MarkerFaceColor','none', ...
        'LineWidth',1.1,'HandleVisibility','off');
    scatter3(ax,sigma(idxC2,1),sigma(idxC2,2),sigma(idxC2,3),35, ...
        'MarkerEdgeColor',colorC2,'MarkerFaceColor','none', ...
        'LineWidth',1.1,'HandleVisibility','off');

    % The actual mean-field magnitude sets the z-directed arrow length.
    L = norm(mean(sigma,1));
    hx = headWidth*cos(arrowAngle);
    hy = headWidth*sin(arrowAngle);
    plot3(ax,[0 0],[0 0],[0 L],'-', ...
        'Color',colorRho,'LineWidth',4,'HandleVisibility','off');
    plot3(ax,[hx 0 -hx],[hy 0 -hy], ...
        [L-headLength L L-headLength],'-', ...
        'Color',colorRho,'LineWidth',4,'HandleVisibility','off');
    text(ax,0.10,0.04,0.60*max(L,0.15),'\rho', ...
        'Interpreter','tex','FontName',fontName,'FontSize',fontSize, ...
        'Color',colorRho,'Clipping','off');

    axis(ax,'equal');
    axis(ax,'vis3d');
    xlim(ax,[-1 1]); ylim(ax,[-1 1]); zlim(ax,[-1 1]);
    view(ax,35,25);
    axis(ax,'off');
end

function ds = rhs5D(sigma,K)
    rho = mean(sigma,1);
    R2 = sum(rho.^2);
    a = sigma*rho.';
    coeff = K*(16*a.^4-12*R2*a.^2+R2^2);
    ds = coeff.*(rho-a.*sigma);
end

function X = normalizeRows(X)
    d = sqrt(sum(X.^2,2));
    if any(~isfinite(d)) || any(d<=1e-14)
        error('Invalid vector encountered during normalization.');
    end
    X = X./d;
end

function X = alignFrame(X,idxP,phiTarget)
    X = rotateToRhoAxis(X);
    p = mean(X(idxP,:),1);
    if hypot(p(1),p(2))<=1e-12
        return;
    end
    delta = phiTarget-atan2(p(2),p(1));
    c = cos(delta); s = sin(delta);
    Q = [c -s 0; s c 0; 0 0 1];
    X = X*Q.';
end

function X = rotateToRhoAxis(X)
    r = mean(X,1).';
    nr = norm(r);
    if ~isfinite(nr) || nr<=1e-12
        error('Mean-field axis rho is undefined.');
    end
    u = r/nr;
    v = [0;0;1];
    c = max(-1,min(1,dot(u,v)));
    if c>1-1e-14
        Q = eye(3);
    elseif c < -1+1e-12
        [~,k] = min(abs(u));
        w = zeros(3,1); w(k)=1;
        w = w-u*dot(u,w); w = w/norm(w);
        Q = eye(3)-2*(u*u'+w*w');
    else
        A = v*u'-u*v';
        Q = eye(3)+A+(A*A)/(1+c);
    end
    X = X*Q.';
end

function [rhoAbs,Z1,Z2] = observe(sigma,idxC1,idxC2,idxP)
    rhoAbs = norm(mean(sigma,1));
    p = mean(sigma(idxP,:),1);
    if hypot(p(1),p(2))<=1e-12
        phiP = 0;
    else
        phiP = atan2(p(2),p(1));
    end
    psi1 = relativeAzimuths(sigma(idxC1,:),phiP);
    psi2 = relativeAzimuths(sigma(idxC2,:),phiP);
    Z1 = abs(mean(exp(1i*psi1)));
    Z2 = abs(mean(exp(1i*psi2)));
end

function psi = relativeAzimuths(X,phiP)
    rr = hypot(X(:,1),X(:,2));
    if any(~isfinite(rr) | rr<=1e-12)
        error('Some azimuthal angles are undefined near the rho axis.');
    end
    raw = atan2(X(:,2),X(:,1))-phiP;
    psi = atan2(sin(raw),cos(raw));
end

function [x,y,z] = referenceRing(theta,npts)
    phi = linspace(0,2*pi,npts);
    x = sin(theta)*cos(phi);
    y = sin(theta)*sin(phi);
    z = cos(theta)*ones(size(phi));
end

function [sigma,idxC1,idxC2,idxP,theta,phi] = ...
    generate_sigma_bands(N,g1,g2,samplingMode)
    n1 = round(N*g1);
    n2 = round(N*g2);
    nP = N-n1-n2;
    assert(n1>0 && n2>0 && nP>0);
    idxC1 = 1:n1;
    idxC2 = n1+(1:n2);
    idxP = n1+n2+(1:nP);

    theta = zeros(N,1);
    phi = 2*pi*rand(N,1);
    if strcmp(samplingMode,'surfaceArea')
        theta(idxP) = sampleThetaArea(0,pi/5,nP);
        theta(idxC1) = sampleThetaArea(pi/5,3*pi/5,n1);
        theta(idxC2) = sampleThetaArea(3*pi/5,pi,n2);
    elseif strcmp(samplingMode,'uniformTheta')
        theta(idxP) = (pi/5)*rand(nP,1);
        theta(idxC1) = pi/5+(2*pi/5)*rand(n1,1);
        theta(idxC2) = 3*pi/5+(2*pi/5)*rand(n2,1);
    else
        error('Unknown samplingMode: %s',samplingMode);
    end
    sigma = [sin(theta).*cos(phi), ...
             sin(theta).*sin(phi), ...
             cos(theta)];
end

function theta = sampleThetaArea(thetaMin,thetaMax,numPoints)
    u = rand(numPoints,1);
    mu = cos(thetaMin)+(cos(thetaMax)-cos(thetaMin))*u;
    theta = acos(max(-1,min(1,mu)));
end

function h = addFigureLabel(fig,pos,str,color,fontName,fontSize,alignment)
    h = annotation(fig,'textbox',pos,'String',str, ...
        'Interpreter','tex','FontName',fontName, ...
        'FontSize',fontSize,'Color',color, ...
        'HorizontalAlignment',alignment,'VerticalAlignment','middle', ...
        'EdgeColor','none','FitBoxToText','off');
end
