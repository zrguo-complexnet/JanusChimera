
%% Higher-dimensional spherical Kuramoto model: collective coherence
%
% FIGURE (all panel positions editable at top of file):
% (a) R1(t), d = 2, 3, 4, 5
% (b) d = 4: transverse 3D coordinates, colored by polar height z
% (c-e) final polar-height probability distributions for d = 3, 4, 5
%       with circled pole peaks and probability-preserving zoom insets.
%
% Equations for oscillators sigma_i in S^(d-1):
%   dot(sigma_i) = (K/N) sum_j sin(n*theta_ij) e_ij,
%   theta_ij = acos(sigma_i dot sigma_j),
%   e_ij = (sigma_j - cos(theta_ij)*sigma_i)/sin(theta_ij).
%
% For n = 3:
%   sin(3theta)/sin(theta) = 4 cos(theta)^2 - 1.
%
% A third-order-moment algorithm evaluates the exact vector field
% with O(d^3*N) operations.
%
% Polar height:
%   z_i = sigma_i dot rho_hat,
%   rho_hat = rho / norm(rho),
%   rho = mean(sigma,2).
%
% MATLAB R2020a or later recommended.

clear; clc; 

%% ============================================================
% 1. Model and numerical parameters
% =============================================================

N            = 1000;
n            = 3;
K            = -10;
DT           = 0.01;
Tmax         = 20;

dimensions   = [2,3,4,5];
seedBase     = 18;
recordEvery  = 1;

% Appearance
fontName     = 'Times New Roman';
fontSize     = 40;
legendSize   = 17;
lineWidth    = 2.2;
markerSize3D = 20;
histBins     = 50;

% ------------------------------------------------------------
% Publication figure layout (normalized coordinates, 0 to 1).
% Each row is [left, bottom, width, height].
% These five axes can be moved and resized independently.
% ------------------------------------------------------------
subplotPos.a = [0.08, 0.59, 0.50, 0.34];
subplotPos.b = [0.65, 0.59, 0.23, 0.34];
subplotPos.c = [0.10, 0.12, 0.24, 0.32];
subplotPos.d = [0.42, 0.12, 0.24, 0.32];
subplotPos.e = [0.74, 0.12, 0.24, 0.32];

% Position of the (b) colorbar in whole-figure coordinates.
colorbarPos = [0.91, 0.61, 0.012, 0.28];

% X label in panels (c-e), in axes-normalized coordinates.
% More positive (e.g. -0.18) -> z moves CLOSER to the x axis.
% More negative (e.g. -0.30) -> z moves FARTHER away.
zLabelY = -0.18;

% Optional tick locations in the time evolution panel (a).
timeXTicks = 0:5:Tmax;

% Pole magnification in panels (c-e).
poleZoom.enabled  = true;
poleZoom.nBins    = 3;                    % First bins next to z = -1.
poleZoom.position = [0.25,0.30,0.25,0.25]; % [left bottom width height]
                                         % relative to each main axes.
poleZoom.yMax     = [];                   % Automatic; optional common limit.
poleZoom.fontSize = 18;

% Output
saveOutputs = true;
outputDir   = 'HighDim_Results';

assert(N >= 1 && n >= 1 && n == floor(n), ...
    'n must be a positive integer.');

assert(all(dimensions >= 2) && ...
       all(dimensions == floor(dimensions)), ...
       'All embedding dimensions must be integers >= 2.');

assert(DT > 0 && Tmax > 0, ...
    'DT and Tmax must be positive.');

assert(poleZoom.nBins >= 1 && poleZoom.nBins <= histBins && ...
       poleZoom.nBins == floor(poleZoom.nBins), ...
       'poleZoom.nBins must be an integer between 1 and histBins.');

assert(isempty(poleZoom.yMax) || ...
       (isscalar(poleZoom.yMax) && isfinite(poleZoom.yMax) && ...
        poleZoom.yMax > 0), ...
       'poleZoom.yMax must be empty or a positive probability.');

nSteps = round(Tmax / DT);

recordSteps = unique([0, ...
    recordEvery:recordEvery:nSteps, nSteps]);

time = recordSteps*DT;

D = struct('d',{},'time',{},'R1',{}, ...
           'sigmaFinal',{},'rhoFinal',{}, ...
           'zFinal',{},'speedRMS',{},'speedMax',{});

%% ============================================================
% 2. Run the microscopic N-oscillator dynamics
% =============================================================

fprintf('Simulating %d oscillators; n = %d, K = %g, dt = %g, T = %g\n', ...
    N,n,K,DT,Tmax);

for k = 1:numel(dimensions)

    d = dimensions(k);
    rng(seedBase + d,'twister');

    % Uniform random initial conditions on S^(d-1).
    sigma = randn(d,N);
    sigma = sigma ./ vecnorm(sigma,2,1);

    R1 = zeros(1,numel(recordSteps));
    R1(1) = norm(mean(sigma,2));

    rec = 2;

    for step = 1:nSteps

        vel = spherical_harmonic_velocity(sigma,K,n);

        % Euler integration with normalization onto the sphere.
        sigma = sigma + DT*vel;
        sigma = sigma ./ vecnorm(sigma,2,1);

        if rec <= numel(recordSteps) && ...
                step == recordSteps(rec)

            R1(rec) = norm(mean(sigma,2));
            rec = rec + 1;

        end
    end

    % Final polarization vector.
    rho = mean(sigma,2);
    rhoMag = norm(rho);

    % Polar heights relative to the final polarization direction.
    if rhoMag > 1e-12

        axisNorth = rho/rhoMag;
        zFinal = axisNorth.'*sigma;

    else

        axisNorth = nan(d,1);
        zFinal = nan(1,N);

    end

    zFinal = min(1,max(-1,zFinal));

    % Final microscopic velocities.
    velFinal = spherical_harmonic_velocity(sigma,K,n);
    speeds = vecnorm(velFinal,2,1);

    % Store results.
    D(k).d          = d;
    D(k).time       = time;
    D(k).R1         = R1;
    D(k).sigmaFinal = sigma;
    D(k).rhoFinal   = rho;
    D(k).zFinal     = zFinal;

    D(k).speedRMS = sqrt(mean(speeds.^2));
    D(k).speedMax = max(speeds);

    fprintf(['d = %d: R1(0)=%.5f, R1(T)=%.5f, ' ...
             'final RMS speed=%.3e, max speed=%.3e\n'], ...
        d,R1(1),R1(end),D(k).speedRMS,D(k).speedMax);

end

%% ============================================================
% 3. Build multi-panel figure
% =============================================================

fig = figure('Color','w', ...
    'Units','inches', ...
    'Position',[0.5,0.5,17.5,10]);

% Use freely positioned axes rather than tiledlayout.
% This avoids tiledlayout automatically changing subplot positions.

% Colors for different dimensions
colors = [
    0.1333, 0.1333, 0.1333;  % d=2: #222222
    0.1333, 0.1333, 0.1333;  % d=3: #222222
    0.2667, 0.4667, 0.6667;  % d=4: #4477AA
    0.2667, 0.4667, 0.6667;  % d=5: #4477AA
];

%% ------------------------------------------------------------
% (a) First-order polarization R1(t)
% -------------------------------------------------------------

axA = axes('Parent',fig, ...
    'Units','normalized', ...
    'Position',subplotPos.a);
hold(axA,'on');

h = gobjects(1,numel(D));
legendText = cell(1,numel(D));

for k = 1:numel(D)

    h(k) = plot(axA,D(k).time,D(k).R1, ...
        'Color',colors(k,:), ...
        'LineWidth',lineWidth);

    legendText{k} = sprintf('$d=%d$',D(k).d);

end

xlabel(axA,'$t$','Interpreter','latex');
ylabel(axA,'$\mathcal{R}_1$','Interpreter','latex');

xlim(axA,[0,Tmax]);
ylim(axA,[0,1]);
xticks(axA,timeXTicks);

legend(axA,h,legendText, ...
    'Interpreter','latex', ...
    'FontSize',legendSize, ...
    'Location','southeast', ...
    'Box','off');

apply_axes_style(axA,fontName,fontSize);
box(axA,'off');

text(axA,0.025,0.94,'(a)', ...
    'Units','normalized', ...
    'FontName',fontName, ...
    'FontSize',fontSize);

%% ------------------------------------------------------------
% (b) d = 4: transverse 3D projection
%
% Four-dimensional positions are represented by:
%   (u1,u2,u3) : three transverse coordinates
%   z          : polar height encoded by color
% -------------------------------------------------------------

axB = axes('Parent',fig, ...
    'Units','normalized', ...
    'Position',subplotPos.b);
hold(axB,'on');

idx4 = find([D.d]==4,1);

if ~isempty(idx4) && norm(D(idx4).rhoFinal)>1e-12

    s4 = D(idx4).sigmaFinal;

    north4 = D(idx4).rhoFinal / ...
             norm(D(idx4).rhoFinal);

    % Orthonormal basis of the transverse 3D subspace.
    B = null(north4.');

    % Three transverse coordinates.
    U = B.'*s4;

    % Polar heights.
    z4 = D(idx4).zFinal;

    hScatter = scatter3(axB, ...
        U(1,:),U(2,:),U(3,:), ...
        markerSize3D,z4, ...
        'filled', ...
        'MarkerEdgeColor','none');

    hScatter.MarkerFaceAlpha = 0.65;

    colormap(axB,parula(256));
    caxis(axB,[-1,1]);

    cb = colorbar(axB,'eastoutside');
    cb.Units = 'normalized';
    cb.Position = colorbarPos;

    % Adding a colorbar may shrink its main axes; restore its position.
    axB.Position = subplotPos.b;

    cb.Label.String = '$z$';
    cb.Label.Interpreter = 'latex';
    cb.FontName = fontName;
    cb.FontSize = legendSize;

    axis(axB,'equal');
    axis(axB,[-1,1,-1,1,-1,1]);

    view(axB,35,22);
    grid(axB,'off');

    % No u1/u2/u3 coordinate labels: the 3D point cloud remains visible.


else

    text(axB,0.5,0.5,'$d=4$ data unavailable', ...
        'Units','normalized', ...
        'HorizontalAlignment','center', ...
        'Interpreter','latex');

end

title(axB,'$d=4$: transverse projection', ...
    'Interpreter','latex');

apply_axes_style(axB,fontName,fontSize-2);
axis(axB,'off');  % Hide the 3D axes, ticks and grid; keep scatter and colorbar.

text(axB,0.02,0.93,'(b)', ...
    'Units','normalized', ...
    'FontName',fontName, ...
    'FontSize',fontSize);

%% ------------------------------------------------------------
% (c,d,e) Final polar-height probability distributions
%
% d = 3: S^2
% d = 4: S^3
% d = 5: S^4
%
% Histogram normalization:
%   Probability = number of oscillators in bin / N.
%
% The sum of all histogram probabilities equals 1.
% -------------------------------------------------------------

zoomData = struct('ax',{},'edges',{},'counts',{}, ...
                  'total',{},'color',{});

for d = 3:5

    panelNames = {'c','d','e'};
    ax = axes('Parent',fig, ...
        'Units','normalized', ...
        'Position',subplotPos.(panelNames{d-2}));
    hold(ax,'on');

    k = find([D.d]==d,1);

    if isempty(k) || any(~isfinite(D(k).zFinal))

        text(ax,0.5,0.5,'No polar data', ...
            'Units','normalized', ...
            'HorizontalAlignment','center');

        continue;

    end

    z = D(k).zFinal;

    % Probability histogram.
    binEdges = linspace(-1,1,histBins+1);

    hh = histogram(ax,z,binEdges, ...
        'Normalization','probability', ...
        'FaceColor',[68,119,170]/255, ...
        'FaceAlpha',0.6, ...
        'EdgeColor','none');

    if poleZoom.enabled
        entry = struct('ax',ax, ...
                       'edges',binEdges, ...
                       'counts',histcounts(z,binEdges), ...
                       'total',numel(z), ...
                       'color',[68,119,170]/255);
        zoomData(end+1) = entry; %#ok<SAGROW>
    end

    % Linear probability axis with automatic range.
    set(ax,'YScale','linear');

    maxProb = max(hh.Values);

    ylim(ax,[0,1.1*maxProb]);
    xlim(ax,[-1,1]);

    xticks(ax,[-1,0,1]);
    xticklabels(ax,{'-1','0','1'});

    % Polar height: keep a handle so that we can adjust label spacing.
    hZLabel = xlabel(ax,'$z$','Interpreter','latex');

    if d==3
        ylabel(ax,'Probability');
    end

    title(ax,sprintf('$d=%d$',d), ...
        'Interpreter','latex');

    apply_axes_style(ax,fontName,fontSize-2);

    % Move the z label up or down using zLabelY at the top of the file.
    hZLabel.Units = 'normalized';
    hZLabel.Position = [0.5,zLabelY,0];
    hZLabel.HorizontalAlignment = 'center';

    text(ax,0.03,0.85, ...
        sprintf('(%c)',char('c'+d-3)), ...
        'Units','normalized', ...
        'FontName',fontName, ...
        'FontSize',fontSize);

end

% Place the inset axes after the main axes positions are established.
if poleZoom.enabled && ~isempty(zoomData)

    drawnow;
    zoomItems = struct('parent',{},'inset',{},'circle',{}, ...
                       'insetCircle',{},'connector',{},'sourceBox',{}, ...
                       'insetPosition',{});

    for k = 1:numel(zoomData)
        zoomItems(end+1) = create_pole_zoom( ...
            fig,zoomData(k),poleZoom,fontName); %#ok<SAGROW>
    end

    setappdata(fig,'PoleZoomItems',zoomItems);
    setappdata(fig,'PoleZoomUpdating',false);

    if isprop(fig,'AutoResizeChildren')
        fig.AutoResizeChildren = 'off';
    end
    fig.SizeChangedFcn = @update_pole_zooms;
    update_pole_zooms(fig,[]);

end

%% ============================================================
% 4. Save results
% =============================================================

if saveOutputs

    if ~exist(outputDir,'dir')
        mkdir(outputDir);
    end

    % Save simulation data.
    save(fullfile(outputDir,'highdim_data.mat'), ...
        'D','N','n','K','DT','Tmax', ...
        'dimensions','seedBase','histBins','poleZoom');

    % Export high-resolution PNG.
    exportgraphics(fig, ...
        fullfile(outputDir,'highdim_comparison.png'), ...
        'Resolution',300, ...
        'BackgroundColor','white');

    % Export vector PDF.
    exportgraphics(fig, ...
        fullfile(outputDir,'highdim_comparison.pdf'), ...
        'ContentType','vector', ...
        'BackgroundColor','white');

    % Export SVG for Affinity editing
    print(fig, ...
        fullfile(outputDir,'highdim_comparison.svg'), ...
        '-dsvg', '-vector');

    fprintf('\nSaved: %s\n', ...
        fullfile(outputDir,'highdim_comparison.pdf'));

end

%% ============================================================
% Local functions
% =============================================================

function item = create_pole_zoom(fig,data,cfg,fontName)

    % Bin probabilities retain the full-population denominator.
    firstBins = 1:cfg.nBins;
    probability = data.counts(firstBins)/data.total;
    edges = data.edges(1:cfg.nBins+1);
    poleCount = sum(data.counts(firstBins));
    poleFraction = poleCount/data.total;

    % yMax is a common scale when supplied, with room for every bar.
    yMax = max([1/data.total,1.25*max(probability),cfg.yMax]);

    axInset = axes('Parent',fig, ...
        'Units','normalized', ...
        'Position',[0.2,0.2,0.15,0.12], ...
        'Color','w', ...
        'Tag','PoleZoomInset');
    hold(axInset,'on');

    % Exact bin edges also support a zoom containing just one bin.
    left = edges(1:end-1);
    right = edges(2:end);
    bottom = zeros(size(probability));
    patch('Parent',axInset, ...
        'XData',[left;right;right;left], ...
        'YData',[bottom;bottom;probability;probability], ...
        'FaceColor',data.color, ...
        'FaceAlpha',0.70, ...
        'EdgeColor','none');

    xlim(axInset,[edges(1),edges(end)]);
    ylim(axInset,[0,yMax]);
    set(axInset,'YScale','linear','TickLabelInterpreter','none');
    % X-axis: only two endpoints, one decimal place
    xticks(axInset,[edges(1),edges(end)]);
    xticklabels(axInset,{ ...
        sprintf('%.1f',edges(1)), ...
        sprintf('%.1f',edges(end))});
    
    % Y-axis: only two endpoints, one decimal place
    yTicks = [0,yMax];
    yticks(axInset,yTicks);
    yticklabels(axInset,arrayfun(@(v) ...
    sprintf('%.1f%%',100*v), ...
    yTicks,'UniformOutput',false));
    

    % title(axInset,{'Pole region', ...
    %     sprintf('%.3g%% (%d/%d)',100*poleFraction,poleCount,data.total)}, ...
    %     'Interpreter','none','FontWeight','normal');
    apply_axes_style(axInset,fontName,cfg.fontSize);
    % box(axInset,'off');

    % Circle the occupied pole bins in the full histogram.
    occupied = find(probability>0);
    if isempty(occupied)
        occupied = 1;
    end
    sourceBox = [edges(occupied(1)),0, ...
                 edges(occupied(end)+1),max(probability)];
    circle = annotation(fig,'ellipse',[0.1,0.1,0.01,0.01], ...
        'Color',data.color,'LineWidth',1.5, ...
        'FaceColor','none','Tag','PoleZoomSource');
    insetCircle = annotation(fig,'ellipse',[0.1,0.1,0.01,0.01], ...
        'Color',data.color,'LineWidth',3, ...
        'FaceColor','none','Tag','PoleZoomInsetCircle');
    connector = annotation(fig,'arrow',[0.1,0.2],[0.1,0.2], ...
        'Color',data.color,'LineWidth',1.1,'LineStyle','--', ...
        'HeadLength',6,'HeadWidth',6,'Tag','PoleZoomConnector');

    item = struct('parent',data.ax,'inset',axInset, ...
        'circle',circle,'insetCircle',insetCircle, ...
        'connector',connector,'sourceBox',sourceBox, ...
        'insetPosition',cfg.position);

end

function update_pole_zooms(fig,~)

    % Anchor circles and insets to their panels after a window resize.
    if ~isgraphics(fig) || ~isappdata(fig,'PoleZoomItems')
        return;
    end
    if getappdata(fig,'PoleZoomUpdating')
        return;
    end
    setappdata(fig,'PoleZoomUpdating',true);
    cleanup = onCleanup(@() finish_pole_zoom_update(fig)); %#ok<NASGU>
    drawnow nocallbacks;

    figPixels = getpixelposition(fig);
    figureWidth = figPixels(3);
    figureHeight = figPixels(4);
    if figureWidth<=0 || figureHeight<=0
        return;
    end
    items = getappdata(fig,'PoleZoomItems');

    for k = 1:numel(items)
        item = items(k);
        if ~isgraphics(item.parent) || ~isgraphics(item.inset) || ...
                ~isgraphics(item.circle) || ~isgraphics(item.insetCircle) || ...
                ~isgraphics(item.connector)
            continue;
        end

        % Pixel positions relative to the figure.
        p = getpixelposition(item.parent,true);
        r = item.insetPosition;
        insetPixels = [p(1)+r(1)*p(3),p(2)+r(2)*p(4), ...
                       r(3)*p(3),r(4)*p(4)];
        item.inset.Position = [ ...
            (insetPixels(1)-1)/figureWidth, ...
            (insetPixels(2)-1)/figureHeight, ...
            insetPixels(3)/figureWidth,insetPixels(4)/figureHeight];

        % Colored circle around the inset itself.
        
        % Circle size
        insetPad = 60;
        
        % Circle position relative to the inset (pixels)
        circleOffsetX = -20;
        circleOffsetY = -5;
        
        % Circle diameter
        insetDiameter = max(insetPixels(3),insetPixels(4)) + 2*insetPad;
        
        % Circle center
        insetCenterX = insetPixels(1) + insetPixels(3)/2 + circleOffsetX;
        insetCenterY = insetPixels(2) + insetPixels(4)/2 + circleOffsetY;
        
        item.insetCircle.Position = [ ...
            (insetCenterX-insetDiameter/2-1)/figureWidth, ...
            (insetCenterY-insetDiameter/2-1)/figureHeight, ...
            insetDiameter/figureWidth, ...
            insetDiameter/figureHeight];

        xLimits = item.parent.XLim;
        yLimits = item.parent.YLim;
        boxData = item.sourceBox;
        centerX = (boxData(1)+boxData(3))/2;
        centerY = (boxData(2)+boxData(4))/2;
        centerPixels = [ ...
            p(1)+(centerX-xLimits(1))/diff(xLimits)*p(3), ...
            p(2)+(centerY-yLimits(1))/diff(yLimits)*p(4)];
        widthPixels = (boxData(3)-boxData(1))/diff(xLimits)*p(3);
        heightPixels = (boxData(4)-boxData(2))/diff(yLimits)*p(4);
        diameter = max([widthPixels,heightPixels,14])+12;

        item.circle.Position = [ ...
            (centerPixels(1)-diameter/2-1)/figureWidth, ...
            (centerPixels(2)-diameter/2-1)/figureHeight, ...
            diameter/figureWidth,diameter/figureHeight];

        % Arrow points to the edge of the inset circle

        % Direction from the source circle to the inset circle
        direction = [insetCenterX,insetCenterY] - centerPixels;
        direction = direction / max(norm(direction),eps);
        
        % Arrow starts at the edge of the source circle
        start = centerPixels + (diameter/2)*direction;
        
        % Arrow ends at the edge of the inset circle
        target = [insetCenterX,insetCenterY] ...
                 - (insetDiameter/2)*direction;
        
        % Update arrow
        item.connector.X = ([start(1),target(1)]-1)/figureWidth;
        item.connector.Y = ([start(2),target(2)]-1)/figureHeight;
    end

end

function finish_pole_zoom_update(fig)

    if isgraphics(fig)
        setappdata(fig,'PoleZoomUpdating',false);
    end

end

function velocity = spherical_harmonic_velocity(sigma,K,n)

    [d,N] = size(sigma);

    if n == 3

        % Exact third-harmonic microscopic force:
        %
        % F_i = (1/N) sum_j
        %       [4(sigma_i.sigma_j)^2-1] sigma_j
        %
        %     = 4*M3(sigma_i,sigma_i) - rho.

        rho = mean(sigma,2);

        G = zeros(d,N);

        for b = 1:d

            for c = b:d

                w = sigma(b,:).*sigma(c,:);

                % Third-order moment contraction.
                m = (sigma*w.')/N;

                mult = 1 + (b~=c);

                G = G + mult*(m*w);

            end

        end

        F = 4*G-rho;

    else

        % General harmonic order:
        %
        % sin(n*theta)/sin(theta)
        % = U_(n-1)(cos(theta)).

        C = sigma.'*sigma;
        C = max(-1,min(1,C));

        W = chebyshev_U(C,n-1);

        F = (sigma*W.')/N;

    end

    % Tangential projection onto S^(d-1).
    velocity = K*(F-sigma.*sum(sigma.*F,1));

end

function U = chebyshev_U(C,m)

    if m==0
        U = ones(size(C));
        return;
    end

    U0 = ones(size(C));
    U1 = 2*C;

    for k = 2:m

        U2 = 2*C.*U1-U0;

        U0 = U1;
        U1 = U2;

    end

    U = U1;

end

function apply_axes_style(ax,fontName,fontSize)

    set(ax, ...
        'FontName',fontName, ...
        'FontSize',fontSize, ...
        'LineWidth',1.25, ...
        'TickDir','out', ...
        'Box','on');

end
