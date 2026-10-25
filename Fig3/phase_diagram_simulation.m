%% phase_diagram_simulation.m
% Numerical phase maps for the fifth-harmonic Kuramoto model on S^2.
%
% WHAT IS MEASURED
%   Z_C1, Z_C2: mean azimuthal coherence of the two initially populated
%   rings, measured around the INSTANTANEOUS mean-field axis rho/|rho|.
%   Colors are time averages of abs(Z_C1), abs(Z_C2) over the last 2W.
%
% WHAT COUNTS AS CONVERGENCE
%   After burnIn, compare the means of abs(Z) in two adjacent windows of
%   width W (sampled every sampleDt). Each occupied ring must satisfy
%        abs(mean_later - mean_earlier) <= meanTol.
%   Two consecutive successful comparisons are required by default.
%   Each parameter point runs until it satisfies the convergence criterion
%   or reaches tMax. The status flag records its convergence outcome.
%
% RUNNING
%   cfg.mode='simulate' calculates the grid; 'plot' redraws existing data.
%   cfg.quickTest=true runs six representative test cases
%   and saves the results in the quick_test/ folder.
%   The separate theory script reads the resulting MAT file.
%
% REQUIREMENTS
%   MATLAB supporting local functions in scripts. The optional parallel
%   parameter scan uses the Parallel Computing Toolbox.
clear; clc;

%% 1. Model and initial-state parameters
cfg.mode = 'simulate';      % 'simulate' | 'plot'
cfg.quickTest = false;      % Set true for an isolated six-point smoke test
cfg.N = 1000;
cfg.K = 50;
cfg.n = 5;                % This implementation is specific to n=5
cfg.Delta = 0;
cfg.gamma1 = 0:0.01:0.50;
cfg.gamma2 = 0:0.01:0.35;
cfg.perturbation = 1e-3;
cfg.seedBase = 20260919;
cfg.seedOffset = 0;

%% 2. Adaptive Dormand-Prince RK45 integration
cfg.relTol = 1e-6;
cfg.absTol = 1e-8;
cfg.dtMax = 2;
cfg.phaseStep = 0.30;
cfg.phaseGuard = 0.30;
cfg.feedbackStep = 0.50;
cfg.stepFactor = 1;         % Use 0.5 for a step-size convergence comparison

%% 3. Convergence from consecutive-window time averages
cfg.sampleDt = 0.25;
cfg.burnIn = 6000;
cfg.window = 2000;         % W: each of the two compared windows
cfg.tMax = 30000;
cfg.meanTol = 0.005;
cfg.requiredPasses = 2;    % Consecutive passing comparisons
cfg.allowEarlyStop = true;

%% 4. Execution and plotting
cfg.useParallel = false;    % true: parallel parameter scan with parfor
cfg.resume = true;          % Reuse compatible completed rows after interruption
cfg.saveFigures = true;
cfg.fontName = 'Arial';     % Font for phase-map axes
cfg.tickFont = 22;
cfg.labelFont = 27;

if cfg.quickTest
    cfg.mode = 'simulate';
    cfg.N = 200;
    cfg.gamma1 = [0.05, 0.10, 0.45];
    cfg.gamma2 = [0.10, 0.20];
    cfg.burnIn = 40;
    cfg.window = 40;
    cfg.tMax = 240;
    cfg.useParallel = false;
    cfg.resume = false;
end

rootDir = fileparts(mfilename('fullpath'));
if isempty(rootDir), rootDir = pwd; end
if cfg.quickTest, rootDir = fullfile(rootDir,'quick_test'); end
if ~exist(rootDir,'dir'), mkdir(rootDir); end
outDir = fullfile(rootDir,'matlab_output_gpu_batch_rho_axis');
if ~exist(outDir,'dir'), mkdir(outDir); end
% Numerical results file consumed by the theory-boundary script.
matFile = fullfile(rootDir,'chimera_phase_gpu_batch_rho_axis_results.mat');
checkpoint = fullfile(outDir,'mean_window_checkpoint.mat');

switch lower(cfg.mode)
    case 'simulate'
        validateRun(cfg);
        [results,simulationCfg] = scanGrid(cfg,checkpoint);
        save(matFile,'results','simulationCfg','-v7');
    case 'plot'
        assert(exist(matFile,'file')==2,'No MAT file: %s',matFile);
        d = load(matFile,'results','simulationCfg');
        checkSavedData(d,cfg);
        results = d.results;
    otherwise
        error('cfg.mode must be ''simulate'' or ''plot''.');
end

%% 5. Numerical phase maps and convergence diagnostic
for group = 1:2
    plotMap(results,group,outDir,cfg);
end
plotMeanDrift(results,outDir,cfg);
fprintf('Converged %d; unresolved %d; undefined ring %d.\n', ...
    nnz(results.status==1),nnz(results.status==0),nnz(results.status==-1));
fprintf('Numerical data: %s\n',matFile);

%% Parameter-grid integration and checkpoint management
function [results,simulationCfg] = scanGrid(c,checkpoint)
    ny = numel(c.gamma2); nx = numel(c.gamma1);
    points = cell(ny,nx);
    % Restore completed parameter points when configuration signatures match.
    signature = [c.N,c.K,c.n,c.Delta,c.gamma1,c.gamma2,c.perturbation, ...
        c.seedBase,c.seedOffset,c.relTol,c.absTol,c.dtMax,c.phaseStep, ...
        c.phaseGuard,c.feedbackStep,c.stepFactor,c.sampleDt,c.burnIn, ...
        c.window,c.tMax,c.meanTol,c.requiredPasses,c.allowEarlyStop];
    if c.resume && exist(checkpoint,'file')==2
        old = load(checkpoint,'points','signature');
        if isfield(old,'points') && isfield(old,'signature') && ...
                isequaln(old.signature,signature) && isequal(size(old.points),size(points))
            points = old.points;
            fprintf('Compatible row checkpoint loaded.\n');
        else
            warning('Old checkpoint is incompatible; recomputing.');
        end
    end
    parallel = c.useParallel && license('test','Distrib_Computing_Toolbox') && ...
        exist('parpool','file')~=0;
    for j = 1:ny
        pending = find(cellfun(@isempty,points(j,:)));
        if isempty(pending), continue; end
        newRow = cell(1,numel(pending));
        if parallel
            parfor k = 1:numel(pending)
                i = pending(k);
                newRow{k} = simulatePoint(c.gamma1(i),c.gamma2(j),c);
            end
        else
            for k = 1:numel(pending)
                i = pending(k);
                newRow{k} = simulatePoint(c.gamma1(i),c.gamma2(j),c);
            end
        end
        points(j,pending) = newRow;
        save(checkpoint,'points','signature','-v7');
        fprintf('gamma_C2 = %.3f: %d/%d rows completed\n',c.gamma2(j),j,ny);
    end
    results.gamma1 = c.gamma1;
    results.gamma2 = c.gamma2;
    keys = fieldnames(points{1});
    for k = 1:numel(keys)
        vals = nan(ny,nx);
        for id = 1:numel(points), vals(id) = points{id}.(keys{k}); end
        results.(keys{k}) = vals;
    end
    results.meanTolerance = c.meanTol;
    results.averageDefinition = 'Time average of abs(Z) relative to instantaneous rho';
    results.convergenceDefinition = 'Successive adjacent-window means only';
    simulationCfg = c;
    % Metadata keys used by the separate rho-axis theory overlay script.
    simulationCfg.checkVersion = 3;
    simulationCfg.engineVersion = 5;
    simulationCfg.integrator = 'rk45';
    simulationCfg.convergenceCriterion = 'two_window_means_only';
end

%% One oscillator population: RK45 + mean-order-parameter checks
function out = simulatePoint(g1,g2,c)
    n1 = round(c.N*g1); n2 = round(c.N*g2);
    seed = mod(c.seedBase + 1000003*round(1000*g1) + ...
        9176*round(1000*g2) + c.seedOffset,2147483646)+1;
    out = struct('Z1mean',nan,'Z2mean',nan,'windowMeanDrift1',nan, ...
        'windowMeanDrift2',nan,'meanCheckPassed',0,'status',-1, ...
        'tEnd',0,'nSteps',0,'seed',seed);
    occupied = [n1>0,n2>0];
    if ~any(occupied), return; end
    sigma = initialState(c.N,n1,n2,c.perturbation,seed);
    nWindow = round(c.window/c.sampleDt);
    burnIdx = round(c.burnIn/c.sampleDt);
    checkIdx = burnIdx+2*nWindow;
    % Circular buffer holds two windows plus their shared endpoint.
    buf = nan(2*nWindow+1,2);
    pos = 0; numStored = 0;
    if burnIdx==0
        buf(1,:) = observe(sigma,n1,n2);
        pos = 1; numStored = 1;
        nextSample = c.sampleDt;
    else
        nextSample = c.burnIn;
    end
    t = 0; steps = 0; passes = 0;
    proposal = c.dtMax*c.stepFactor;
    k1 = []; R = nan;
    delta = [nan,nan]; passed = false;
    while t<c.tMax
        if isempty(k1), [k1,R] = rhs(sigma,c.K); end
        % Step limits prevent excessive rotation/feedback even when
        % the embedded RK45 local truncation error happens to be small.
        free = min([proposal,c.dtMax*c.stepFactor, ...
            0.9*c.phaseStep*c.stepFactor/(c.K*max(R,1e-12)^5), ...
            0.9*c.feedbackStep*c.stepFactor/(c.K*max(R,1e-12)^4*(1+R))]);
        h = min(free,nextSample-t);
        clipped = h<free;
        [sigma,k1,R,h,proposal,rejected] = stepRK45(sigma,k1,R,h,c,t);
        if clipped && ~rejected, proposal = max(proposal,free); end
        t = t+h; steps = steps+1;
        if t>=nextSample-32*eps(max(1,nextSample))
            t = nextSample;
            sampleIdx = round(t/c.sampleDt);
            if sampleIdx>=burnIdx
                pos = mod(pos,size(buf,1))+1;
                buf(pos,:) = observe(sigma,n1,n2);
                numStored = numStored+1;
            end
            nextSample = (sampleIdx+1)*c.sampleDt;
            if sampleIdx==checkIdx
                assert(numStored>=size(buf,1),'Need two complete mean windows.');
                tail = buf(mod(pos+(0:size(buf,1)-1),size(buf,1))+1,:);
                avgEarly = trapz(tail(1:nWindow+1,:),1)/nWindow;
                avgLate = trapz(tail(nWindow+1:end,:),1)/nWindow;
                delta = abs(avgLate-avgEarly);
                passed = all(isfinite(delta(occupied))) && ...
                    all(delta(occupied)<=c.meanTol);
                if passed, passes = passes+1; else, passes = 0; end
                checkIdx = checkIdx+nWindow;
                if c.allowEarlyStop && passes>=c.requiredPasses, break; end
            end
        end
    end
    tail = buf(mod(pos+(0:size(buf,1)-1),size(buf,1))+1,:);
    means = trapz(tail,1)/(2*nWindow);
    out.Z1mean = means(1); out.Z2mean = means(2);
    out.windowMeanDrift1 = delta(1); out.windowMeanDrift2 = delta(2);
    out.meanCheckPassed = double(passed);
    out.status = double(passes>=c.requiredPasses);
    out.tEnd = t; out.nSteps = steps;
end

%% Seeded initialization: two latitude rings and point P
function sigma = initialState(N,n1,n2,epsilon,seed)
    sigma = zeros(N,3); state = seed;
    for i = 1:N
        if i<=n1, theta = 2*pi/5;
        elseif i<=n1+n2, theta = 4*pi/5;
        else, theta = 0;
        end
        state = mod(16807*state,2147483647);
        az = 2*pi*state/2147483647;
        v = [sin(theta)*cos(az),sin(theta)*sin(az),cos(theta)];
        for d = 1:3
            state = mod(16807*state,2147483647);
            v(d) = v(d)+epsilon*(2*state/2147483647-1);
        end
        sigma(i,:) = v/norm(v);
    end
end

%% Fifth-harmonic tangent vector field on S^2
function [v,R] = rhs(s,K)
    s = s./sqrt(sum(s.^2,2));
    rho = mean(s,1); R2 = sum(rho.^2);
    a = s*rho.'; a2 = a.^2;
    v = K*((16*a2-12*R2).*a2+R2^2).*(rho-a.*s);
    R = sqrt(R2);
end

%% Dormand-Prince (5,4) step, with unit-sphere renormalization
function [sn,kn,Rn,h,hNext,rejected] = stepRK45(s,k1,R,h,c,t)
    rejected = false;
    for attempt = 1:60
        assert(h>0 && t+h>t,'RK45 step underflow.');
        [k2,r2] = rhs(s+h*k1/5,c.K);
        [k3,r3] = rhs(s+h*(3*k1/40+9*k2/40),c.K);
        [k4,r4] = rhs(s+h*(44*k1/45-56*k2/15+32*k3/9),c.K);
        [k5,r5] = rhs(s+h*(19372*k1/6561-25360*k2/2187+ ...
            64448*k3/6561-212*k4/729),c.K);
        [k6,r6] = rhs(s+h*(9017*k1/3168-355*k2/33+ ...
            46732*k3/5247+49*k4/176-5103*k5/18656),c.K);
        raw5 = s+h*(35*k1/384+500*k3/1113+125*k4/192- ...
            2187*k5/6784+11*k6/84);
        sn = raw5./sqrt(sum(raw5.^2,2));
        [kn,r7] = rhs(sn,c.K);
        raw4 = s+h*(5179*k1/57600+7571*k3/16695+ ...
            393*k4/640-92097*k5/339200+187*k6/2100+kn/40);
        scale = c.absTol+c.relTol*max(abs(s),abs(sn));
        err = max(abs(raw5-raw4)./scale,[],'all');
        err = max(err,max(abs(sn-raw5)./scale,[],'all'));
        peakR = max([R,r2,r3,r4,r5,r6,r7]);
        phase = c.K*peakR^5*h;
        feedback = c.K*peakR^4*(1+peakR)*h;
        finite = all(isfinite(sn),'all') && all(isfinite(raw4),'all');
        if finite && isfinite(err) && err<=1 && ...
                phase<=c.phaseGuard*c.stepFactor*(1+1e-12) && ...
                feedback<=c.feedbackStep*c.stepFactor*(1+1e-12)
            Rn = r7;
            if err==0, factor=4;
            else, factor=min(4,max(0.2,0.9*err^(-1/5))); end
            hNext = h*factor;
            return;
        end
        rejected = true;
        if ~finite || ~isfinite(err)
            factor = 0.2;
        else
            factor = min(0.9,max(0.1,0.9*max(err,1e-16)^(-1/5)));
            if phase>c.phaseGuard*c.stepFactor
                factor = min(factor,0.9*c.phaseGuard*c.stepFactor/phase);
            end
            if feedback>c.feedbackStep*c.stepFactor
                factor = min(factor,0.9*c.feedbackStep*c.stepFactor/feedback);
            end
        end
        h = h*factor;
    end
    error('RK45 failed after 60 step attempts.');
end

%% Azimuthal order parameters measured around the instantaneous rho axis
function Z = observe(s,n1,n2)
    Z = [nan,nan];
    % Check the P-component mean direction before defining ring observables.
    pMean = mean(s(n1+n2+1:end,:),1);
    pNorm = norm(pMean);
    if ~isfinite(pNorm) || pNorm<=1e-12, return; end
    rho = mean(s,1); R = norm(rho);
    if ~isfinite(R) || R<=1e-12, return; end
    north = rho/R;
    if abs(north(3))<0.9, reference=[0,0,1];
    else, reference=[1,0,0]; end
    e1 = cross(reference,north); e1 = e1/norm(e1);
    e2 = cross(north,e1);
    for group = 1:2
        if group==1, ids=1:n1;
        else, ids=n1+(1:n2); end
        if isempty(ids), continue; end
        x = s(ids,:)*e1.'; y = s(ids,:)*e2.';
        r = hypot(x,y);
        if all(r>1e-10), Z(group)=abs(mean((x+1i*y)./r)); end
    end
end

%% C1 and C2 numerical phase maps
function plotMap(results,group,outDir,c)
    field = sprintf('Z%dmean',group);
    values = results.(field);
    f = figure('Color','w','Position',[80+40*group,60,900,790]);
    ax = axes(f,'Position',[.135,.24,.67,.66]);
    h = imagesc(ax,results.gamma1,results.gamma2,values,[0,1]);
    set(h,'AlphaData',isfinite(values));
    set(ax,'Color',[.88,.88,.88], 'YDir','normal', ...
        'XLim',[0,.5],'YLim',[0,.35], ...
        'XTick',0:.1:.5,'YTick',0:.05:.35, ...
        'FontName',c.fontName,'FontSize',c.tickFont, ...
        'LineWidth',1.3,'TickDir','out','Layer','top');
    xlabel(ax,'$\gamma_{C_1}$','Interpreter','latex','FontSize',c.labelFont);
    ylabel(ax,'$\gamma_{C_2}$','Interpreter','latex','FontSize',c.labelFont);
    title(ax,sprintf('Ring C%d',group),'FontSize',c.labelFont,'FontWeight','normal');
    colormap(ax,sequentialMap());
    cb = colorbar(ax); cb.Ticks=0:.2:1; cb.FontSize=c.tickFont;
    cb.Label.Interpreter='latex';
    cb.Label.String = '$Z_{\alpha}$';
    cb.Label.FontSize=c.labelFont;
    ax.Position=[.135,.24,.64,.66]; cb.Position=[.82,.24,.025,.66];
    if c.saveFigures, savePlot(f,fullfile(outDir,sprintf('C%d_simulation',group))); end
end

%% Diagnostic: magnitude of the last adjacent-window mean difference
function plotMeanDrift(r,outDir,c)
    % Draw the per-ring adjacent-window mean differences from stored data.
    if ~all(isfield(r,{'windowMeanDrift1','windowMeanDrift2'}))
        warning('This MAT file lacks per-ring window drift; diagnostic skipped.');
        return;
    end
    f = figure('Color','w','Position',[120,90,1450,620]);
    for group = 1:2
        ax = subplot(1,2,group);
        field = sprintf('windowMeanDrift%d',group);
        vals = r.(field);
        h = imagesc(ax,r.gamma1,r.gamma2,vals);
        set(h,'AlphaData',isfinite(vals));
        set(ax,'YDir','normal','XLim',[0,.5],'YLim',[0,.35], ...
            'FontName',c.fontName,'FontSize',16,'Color',[.88,.88,.88]);
        xlabel(ax,'\gamma_{C_1}'); ylabel(ax,'\gamma_{C_2}');
        tolerance = c.meanTol;
        if isfield(r,'meanTolerance'), tolerance = r.meanTolerance; end
        title(ax,sprintf('C%d mean drift (threshold %.3g)',group,tolerance));
        colormap(ax,sequentialMap());
        cb = colorbar(ax); cb.Label.String='|mean_{later} - mean_{earlier}|';
    end
    if c.saveFigures, savePlot(f,fullfile(outDir,'mean_window_diagnostics')); end
end

function cmap = sequentialMap()
    anchors = [.267,.005,.329;.283,.141,.458;.254,.265,.530; ...
        .207,.372,.553;.164,.471,.558;.128,.567,.551; ...
        .135,.659,.518;.267,.749,.441;.478,.821,.318; ...
        .741,.873,.150;.993,.906,.144];
    cmap = interp1(linspace(0,1,11),anchors,linspace(0,1,256),'linear');
end

function savePlot(f,base)
    % Save editable FIG, 300 dpi PNG, and vector PDF files.
    savefig(f,[base,'.fig']);
    if exist('exportgraphics','file')
        exportgraphics(f,[base,'.png'],'Resolution',300);
        exportgraphics(f,[base,'.pdf'],'ContentType','vector');
    else
        set(f,'PaperPositionMode','auto');
        print(f,[base,'.png'],'-dpng','-r300');
        print(f,[base,'.pdf'],'-dpdf','-painters');
    end
end

function validateRun(c)
    assert(c.n==5 && c.Delta==0 && c.K>0,'Only n=5, Delta=0, K>0.');
    assert(c.N==round(c.N) && c.N>=3,'N must be an integer >=3.');
    assert(all(c.gamma1>=0) && all(c.gamma2>=0) && ...
        round(c.N*max(c.gamma1))+round(c.N*max(c.gamma2))<c.N, ...
        'P must contain at least one oscillator.');
    assert(c.sampleDt>0 && c.window>=c.sampleDt && c.burnIn>=0);
    assert(c.tMax>=c.burnIn+2*c.window, ...
        'tMax must be >= burnIn + 2*window.');
    times = [c.burnIn,c.window,c.tMax]/c.sampleDt;
    assert(all(abs(times-round(times))<1e-8), ...
        'burnIn, window, tMax must be multiples of sampleDt.');
    assert(abs((c.tMax-c.burnIn)/c.window- ...
        round((c.tMax-c.burnIn)/c.window))<1e-8, ...
        'tMax-burnIn must be a multiple of window.');
    assert(c.requiredPasses>=1 && c.requiredPasses==round(c.requiredPasses));
    assert(c.meanTol>0 && c.relTol>0 && c.absTol>0 && c.dtMax>0);
    assert(c.phaseStep>0 && c.phaseGuard>=c.phaseStep && c.feedbackStep>0);
end

function checkSavedData(d,c)
    assert(isfield(d,'results') && isfield(d,'simulationCfg'), ...
        'Expected results and simulationCfg in the MAT file.');
    s = d.simulationCfg;
    assert(isfield(s,'checkVersion') && isfield(s,'engineVersion') && ...
        s.checkVersion==3 && s.engineVersion==5, ...
        'Expected rho-axis data with checkVersion=3, engineVersion=5.');
    assert(s.n==c.n && s.K==c.K && s.Delta==c.Delta, ...
        'Numerical data and current code have different model parameters.');
    assert(all(isfield(d.results,{'gamma1','gamma2','Z1mean','Z2mean'})), ...
        'Incomplete numerical phase-map data.');
end
