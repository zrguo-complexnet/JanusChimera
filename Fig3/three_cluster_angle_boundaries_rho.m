
%% three_cluster_angle_boundaries_rho.m
% n = 5, Delta = 0
%
% Three-cluster theoretical stability boundaries.
% Overlay with phase diagrams measured relative to rho.
%
% Red solid: C1 internal stability boundary.
% Black solid: P internal stability boundary (three clusters).
% Black dashed: approximate P stability boundary (chimera side).
% Blue dotted: azimuthal feedback boundary.
%
% The blue dotted feedback line spans the full displayed x range.
% Its intersection with the approximate P-instability line is also
% computed and exported as a diagnostic point (NOT a plotting endpoint).
% The black P-unstable mask combines an exact three-cluster boundary on
% the left and an approximation on the chimera side; the latter is not
% an exact chimera-state stability threshold.
%
% No N-particle simulation is performed in this script.
% No fsolve, GPU, Parallel Toolbox or MEX is required.

clear;
clc;

%% ============================================================
%  1. Configuration
% =============================================================

cfg.K = 50;

cfg.gamma1Limits = [0, 0.50];
cfg.gamma2Limits = [0, 0.35];

cfg.minimumGamma = 0.001;

% Figure settings
cfg.fontName = 'Times New Roman';
cfg.fontSize = 26;
cfg.lineWidth = 2.5;
cfg.legendFontSize = 17;

cfg.saveFigures = true;
cfg.exportPDF = true;

cfg.progressSeconds = 2;

% Phase diagram generated with instantaneous rho axis.
% Leave empty for automatic discovery.
cfg.phaseMatFile = '';
cfg.autoFindPhaseFile = true;
cfg.requireRhoPhase = true;

% Theory and display switches
cfg.showFormalExtensions = false;
cfg.showOldPApproximation = true;
cfg.showOldFeedback = true;
cfg.shadePUnstable = true;

cfg.theorySamples = 1201;

cfg.feedbackColor = [0.03, 0.42, 0.75];
cfg.oldPColor = [0.15, 0.15, 0.15];

% Row 1: P boundary
% Row 2: C1 boundary
cfg.colors = [
    0.10, 0.10, 0.10;
    0.84, 0.16, 0.13
];

%% Initial guesses for three-cluster solutions

cfg.anchorGamma = [0.05, 0.20];

cfg.anchorTheta = [
    0.19972462;
    1.20199845;
    2.33465573
];

cfg.jointGammaGuess = [
    0.07431062;
    0.25708516
];

cfg.jointThetaGuess = [
    0.31636233;
    1.15556750;
    2.28773575
];

%% Numerical continuation settings

cfg.parameterScale = 5;

cfg.initialStep = 0.01;
cfg.maximumStep = 0.02;
cfg.minimumStep = 2e-7;

cfg.maximumPoints = 1500;
cfg.maximumRetries = 18;

cfg.maxShapeJump = 0.08;

cfg.fdStep = 1e-7;

cfg.newtonTolerance = 2e-13;
cfg.rootTolerance = 5e-11;
cfg.boundaryMarginTolerance = 2e-8;

cfg.maxNewtonIterations = 50;
cfg.maxNewtonStep = 0.25;

cfg.minimumR = 1e-5;
cfg.poleTolerance = 1e-7;

cfg.geometryTolerance = 2e-9;
cfg.coplanarTolerance = 2e-9;
cfg.rotationTolerance = 2e-7;
cfg.distinctTolerance = 1e-7;

cfg.stabilityTolerance = 1e-7;
cfg.validationTolerance = 5e-5;
cfg.validationPoints = 10;

%% Output paths

scriptDir = fileparts(mfilename('fullpath'));

if isempty(scriptDir)
    scriptDir = pwd;
end

cfg.outputDir = fullfile( ...
    scriptDir, 'angle_boundary_output_rho_v4');

if ~exist(cfg.outputDir, 'dir')
    mkdir(cfg.outputDir);
end

% Automatically locate the rho-axis phase-map data.
if isempty(cfg.phaseMatFile) && cfg.autoFindPhaseFile

    candidate = fullfile( ...
        scriptDir, ...
        'chimera_phase_gpu_batch_rho_axis_results.mat');

    if exist(candidate, 'file') == 2
        cfg.phaseMatFile = candidate;
    end

end

if ~isempty(cfg.phaseMatFile)

    assert(exist(cfg.phaseMatFile, 'file') == 2, ...
        'Phase data file not found: %s', cfg.phaseMatFile);

    verify_rho_phase_file(cfg.phaseMatFile, cfg);

    fprintf('Loaded rho-axis phase map:\n%s\n', ...
        cfg.phaseMatFile);

else

    warning(['Rho-axis phase data not found. ', ...
        'Only theoretical boundaries will be plotted.']);

end

assert(cfg.K > 0);
assert(cfg.minimumGamma > 0);

assert( ...
    sum([cfg.gamma1Limits(2), cfg.gamma2Limits(2)]) < 1, ...
    'The P population must be nonempty.');

fprintf('\nThree-cluster boundary calculation\n');
fprintf('n = 5, Delta = 0, K = %.3g\n', cfg.K);
fprintf('Position stability is used as a filter.\n');

runTimer = tic;

%% ============================================================
%  2. Stable anchor and exact joint critical point
% =============================================================

anchor = solve_root( ...
    cfg.anchorTheta, ...
    weights(cfg.anchorGamma), ...
    cfg);

assert(anchor.valid && all(anchor.mu < 0), ...
    'The stable anchor could not be recovered.');

selfTests = check_known_configurations(anchor, cfg);

assert(all(selfTests.passed), ...
    'Startup vector-matrix validation failed.');

% Unknowns:
% x = [theta_P; theta_C1; theta_C2;
%      parameterScale*gamma_C1;
%      parameterScale*gamma_C2]

x0 = [
    cfg.jointThetaGuess;
    cfg.parameterScale * cfg.jointGammaGuess
];

% Simultaneous critical conditions:
% F = 0, mu_P = 0, mu_C1 = 0

[jointX, ok] = newton_correct( ...
    @(x) critical_values(x, [2, 3], cfg), ...
    x0, cfg);

joint = accept_critical(jointX, [2, 3], cfg);

assert(ok && joint.valid, ...
    'The simultaneous P/C1 critical point was not resolved.');

assert(all(joint.mu([1, 4]) < 0), ...
    'The joint point has an unstable background.');

fprintf('\nJoint critical point:\n');
fprintf('gamma_C1 = %.9f\n', joint.g(2));
fprintf('gamma_C2 = %.9f\n', joint.g(3));
fprintf('residual = %.3e\n', joint.rootResidual);

%% ============================================================
%  3. Continue exact stability boundaries
% =============================================================

% mode = 1: position stability
% mode = 2: P internal stability
% mode = 3: C1 internal stability
% mode = 4: C2 internal stability
%
% Trace modes 3 and 2 only.

curves = struct( ...
    'mode', {}, ...
    'roots', {}, ...
    'xy', {}, ...
    'backgroundOK', {}, ...
    'eligible', {});

traceInfo = struct( ...
    'mode', {}, ...
    'direction', {}, ...
    'points', {}, ...
    'gammaC1', {}, ...
    'gammaC2', {}, ...
    'stopReason', {});

for mode = [3, 2]

    fprintf('\nTracing %s boundary...\n', mode_name(mode));

    [minus, infoMinus] = trace_boundary( ...
        jointX, mode, -1, cfg);

    [plus, infoPlus] = trace_boundary( ...
        jointX, mode, +1, cfg);

    curveRoots = [flipud(minus); plus(2:end)];

    curves(end+1) = pack_curve( ...
        curveRoots, mode, cfg); %#ok<SAGROW>

    traceInfo(end+1) = infoMinus; %#ok<SAGROW>
    traceInfo(end+1) = infoPlus; %#ok<SAGROW>

    save( ...
        fullfile(cfg.outputDir, 'boundary_checkpoint.mat'), ...
        'cfg', 'joint', 'curves', 'traceInfo', '-v7');

end

%% ============================================================
%  4. Validation and boundary data
% =============================================================

boundaryTable = pack_boundary_table(curves, cfg);

endpointTable = struct2table(traceInfo);

intersectionTable = array2table( ...
    [joint.g(2:3).', joint.theta.', joint.mu, ...
     joint.rootResidual, joint.rotationResidual], ...
    'VariableNames', ...
    {'gammaC1', 'gammaC2', ...
     'thetaP', 'thetaC1', 'thetaC2', ...
     'muPosition', 'muP', 'muC1', 'muC2', ...
     'rootResidual', 'rotationResidual'});

validation = validate_curves( ...
    curves, joint, selfTests, cfg);

% Approximate P boundary and feedback boundary.
[theory, theoryTable] = original_approximation( ...
    curves, joint, cfg);

% Intersection of two approximations (diagnostic; not plot endpoint).
feedbackEndpointTable = table( ...
    theory.feedbackEndpoint(1), ...
    theory.feedbackEndpoint(2), ...
    theory.feedbackEndpointMaskY, ...
    theory.feedbackEndpointGap, ...
    'VariableNames', ...
    {'gammaC1', 'gammaC2', ...
     'PmaskAtEndpoint', 'matchingError'});

writetable(boundaryTable, ...
    fullfile(cfg.outputDir, 'boundary_points.csv'));

writetable(endpointTable, ...
    fullfile(cfg.outputDir, 'trace_endpoints.csv'));

writetable(intersectionTable, ...
    fullfile(cfg.outputDir, 'boundary_intersection.csv'));

writetable(validation, ...
    fullfile(cfg.outputDir, 'validation.csv'));

writetable(theoryTable, ...
    fullfile(cfg.outputDir, 'original_approximation.csv'));

writetable(feedbackEndpointTable, ...
    fullfile(cfg.outputDir, 'feedback_boundary_endpoint.csv'));

save( ...
    fullfile(cfg.outputDir, 'angle_boundaries.mat'), ...
    'cfg', 'anchor', 'joint', 'curves', 'traceInfo', ...
    'boundaryTable', 'endpointTable', ...
    'intersectionTable', 'validation', ...
    'theory', 'theoryTable', ...
    'feedbackEndpointTable', '-v7');

assert(all(validation.passed), ...
    ['Vector-matrix validation failed. ', ...
     'Inspect validation.csv before plotting.']);

%% ============================================================
%  5. Draw theoretical boundaries
% =============================================================

f = figure( ...
    'Color', 'w', ...
    'Position', [100, 60, 1080, 880]);

ax = axes('Parent', f);

hold(ax, 'on');

format_axes(ax, cfg);
draw_phase_overlays(ax, curves, theory, cfg);

save_figure(f, 'phase_boundaries', cfg);

%% ============================================================
%  6. Overlay onto the rho-axis phase diagrams
% =============================================================

if ~isempty(cfg.phaseMatFile)

    overlay_phase_file(curves, theory, cfg);

else

    fprintf('\nNo phase map selected.\n');

end

fprintf('\nFinished in %.1f seconds.\n', toc(runTimer));

fprintf('Boundary points: %d\n', height(boundaryTable));

fprintf('Valid solid-line points: %d\n', ...
    nnz(boundaryTable.solidLine));

fprintf('Validation passed: %d/%d\n', ...
    nnz(validation.passed), height(validation));

fprintf('Feedback/P approximation intersection: (%.9f, %.9f)\n', ...
    theory.feedbackEndpoint(1), ...
    theory.feedbackEndpoint(2));

fprintf('Output: %s\n', cfg.outputDir);

disp(endpointTable);


%% ============================================================
%  LOCAL FUNCTIONS
% =============================================================

function g = weights(xy)

g = [
    1 - xy(1) - xy(2);
    xy(1);
    xy(2)
];

end


%% ============================================================
%  Angle equations and analytical Jacobian
% =============================================================

function [F, D, R, H, C] = angle_equations(th, g)

th = th(:);
g = g(:);

s = sin(th);
c = cos(th);

s5 = sin(5*th);
c5 = cos(5*th);

R = g.' * c;

C = eye(3);
dC = zeros(3, 3, 3);

for a = 1:2

    for b = a+1:3

        n = 6 - a - b;

        v = ( ...
            g(n)^2*s(n)^2 ...
            - g(a)^2*s(a)^2 ...
            - g(b)^2*s(b)^2 ) ...
            / (2*g(a)*g(b)*s(a)*s(b));

        C(a,b) = v;
        C(b,a) = v;

        dC(a,b,a) = ...
            -v*c(a)/s(a) ...
            -g(a)*c(a)/(g(b)*s(b));

        dC(a,b,b) = ...
            -v*c(b)/s(b) ...
            -g(b)*c(b)/(g(a)*s(a));

        dC(a,b,n) = ...
            g(n)^2*s(n)*c(n) ...
            /(g(a)*g(b)*s(a)*s(b));

        dC(b,a,:) = dC(a,b,:);

    end

end

v = g .* s5 .* c;

F = -R*s5 + C*v;

D = s5*(g.*s).' ...
    -5*R*diag(c5) ...
    +C*diag(g.*(5*c5.*c-s5.*s));

for ell = 1:3

    D(:,ell) = D(:,ell) + dC(:,:,ell)*v;

end

H = c*c.' + (s*s.').*C;

end


function [f, A] = solver_equations(th, g)

[F, D, R] = angle_equations(th, g);

f = F/R;

A = D/R + F*(g.*sin(th)).'/R^2;

end


function ok = regular_angles(th, g, cfg)

ok = all(isfinite(th)) && isreal(th) && ...
    all(th > cfg.poleTolerance & ...
        th < pi-cfg.poleTolerance) && ...
    g.'*cos(th) > cfg.minimumR;

end


%% ============================================================
%  Three-cluster stationary solution
% =============================================================

function q = solve_root(th, g, cfg)

th = th(:);

q = invalid_root(-1, g, th);

if ~regular_angles(th, g, cfg)

    q.code = -5;
    return;

end

for it = 1:cfg.maxNewtonIterations

    [f, A] = solver_equations(th, g);

    if any(~isfinite([f(:); A(:)]))
        return;
    end

    if norm(f, inf) < cfg.newtonTolerance
        break;
    end

    if rcond(A) > 1e-11

        step = -A\f;

    else

        step = -pinv(A, 1e-12)*f;

    end

    step = step * min( ...
        1, cfg.maxNewtonStep/max(norm(step,inf),eps));

    accepted = false;
    old = norm(f,2);

    for method = 1:2

        if method == 2

            step = -A.'*f;

            step = step * min( ...
                1, cfg.maxNewtonStep/max(norm(step,inf),eps));

        end

        alpha = 1;

        for ls = 1:19

            trial = th + alpha*step;

            if regular_angles(trial,g,cfg)

                ft = solver_equations(trial,g);

                if all(isfinite(ft)) && ...
                        norm(ft,2) < (1-1e-4*alpha)*old

                    accepted = true;
                    break;

                end

            end

            alpha = alpha/2;

        end

        if accepted
            break;
        end

    end

    if ~accepted
        break;
    end

    th = trial;

end

q = analyse_root(th, g, cfg);

end


function q = invalid_root(code, g, th)

q = struct( ...
    'valid', false, ...
    'code', code, ...
    'g', g, ...
    'theta', th);

end


%% ============================================================
%  Geometry and stability of three clusters
% =============================================================

function q = analyse_root(th, g, cfg)

q = invalid_root(-1, g, th);

if ~regular_angles(th,g,cfg)

    q.code = -5;
    return;

end

[F, D, R, H, C] = angle_equations(th,g);

res = norm(F,inf)/R;

if ~isfinite(res) || res > cfg.rootTolerance
    return;
end

[V,L] = eig((H+H.')/2);

he = real(diag(L));

if min(he) < -cfg.geometryTolerance || ...
        R > 1+cfg.geometryTolerance || ...
        max(abs(C(:))) > 1+20*cfg.geometryTolerance

    q.code = -2;
    return;

end

if min(1-H([2,3,6])) < cfg.distinctTolerance || ...
        nnz(he > cfg.coplanarTolerance) < 2

    q.code = -2;
    return;

end

coplanar = min(he) <= cfg.coplanarTolerance;

% Eigenvalue clipping is used only for geometry reconstruction.
he(abs(he) <= cfg.coplanarTolerance) = 0;

S = V*diag(sqrt(max(he,0)));

S = S./sqrt(sum(S.^2,2));

rho = g.'*S;

Rrec = norm(rho);
u = rho/Rrec;

cosine = S*u.';

A = 16*cosine.^4 - 12*cosine.^2 + 1;

force = A.*(u-cosine.*S);

rotationMatrix = 3*eye(3) - S.'*S;

if rcond(rotationMatrix) < 1e-11

    q.code = -2;
    return;

end

Om = rotationMatrix \ ...
    sum(cross(S,force,2),1).';

rotErr = norm( ...
    force-cross(repmat(Om.',3,1),S,2), 'fro');

recErr = max( ...
    norm(S*S.'-H,inf), abs(Rrec-R));

if rotErr > cfg.rotationTolerance || ...
        recErr > 30*cfg.geometryTolerance

    q.code = -3;
    return;

end

% Position stability
Jscaled = D/R;

ev = eig(Jscaled);

inner = zeros(2,2,3);

eigInner = complex(zeros(2,3));

% mu(1): position stability
% mu(2): P internal stability
% mu(3): C1 internal stability
% mu(4): C2 internal stability

mu = zeros(1,4);

mu(1) = max(real(ev));

for a = 1:3

    spin = Om.'*S(a,:).';

    inner(:,:,a) = [
        -5*cos(5*th(a)), spin;
        -spin, -sin(5*th(a))*cos(th(a))/sin(th(a))
    ];

    eigInner(:,a) = eig(inner(:,:,a));

    mu(a+1) = max(real(eigInner(:,a)));

end

q.valid = true;
q.code = 1;

q.R = R;
q.H = H;
q.S = S;
q.u = u;

q.D = D;

q.Jscaled = Jscaled;
q.eigAngleScaled = ev;

q.Mscaled = inner;
q.eigInternalScaled = eigInner;

q.mu = mu;

q.OmegaScaled = Om;
q.Omega = cfg.K*R^5*Om;

q.rootResidual = res;
q.rotationResidual = rotErr;
q.reconstructionError = recErr;

q.coplanar = coplanar;

q.gramEigenvalues = sort(real(diag(L)));

q.lambdaPerpScaled = sum( ...
    (sin(5*th)./sin(th)).*(g/R-cos(th)));

end


function th = geometry_seed(H,g)

R2 = g.'*H*g;

if R2 <= 0

    th = nan(3,1);
    return;

end

q = H*g/sqrt(R2);

if max(abs(q)) > 1+1e-8

    th = nan(3,1);
    return;

end

th = acos(max(-1,min(1,q)));

end


%% ============================================================
%  Critical equations for direct continuation
% =============================================================

function f = critical_values(x,modes,cfg)

f = [];

x = x(:);

th = x(1:3);

g = weights(x(4:5)/cfg.parameterScale);

if any(~isfinite(x)) || ...
        ~isreal(x) || ...
        any(g <= 0) || ...
        ~regular_angles(th,g,cfg)

    return;

end

[F,D,R,H] = angle_equations(th,g);

B = 3*eye(3)-H;

if any(~isfinite(B(:))) || rcond(B) < 1e-13
    return;
end

A = sin(5*th)./sin(th);

w = (B\cross(A,g))/R;

spinSquared = det(H)*w.^2;

a = -5*cos(5*th);

d = -sin(5*th).*cos(th)./sin(th);

internal = (a+d)/2 ...
    + sqrt(max(0,((a-d)/2).^2-spinSquared));

mu = [
    max(real(eig(D/R)));
    internal
];

f = [F/R; mu(modes)];

if any(~isfinite(f))
    f = [];
end

end


function q = accept_critical(x,modes,cfg)

g = weights(x(4:5)/cfg.parameterScale);

q = analyse_root(x(1:3),g,cfg);

if q.valid && ...
        max(abs(q.mu(modes))) > cfg.boundaryMarginTolerance

    q.valid = false;
    q.code = -6;

end

end


%% ============================================================
%  Numerical Jacobian and Newton corrector
% =============================================================

function A = numerical_jacobian(fun,x,f,cfg)

A = zeros(numel(f),numel(x));

for k = 1:numel(x)

    h = cfg.fdStep*max(1,abs(x(k)));

    if k <= 3

        h = min(h,0.01*min(x(k),pi-x(k)));

    else

        h = min(h,0.01*min( ...
            [x(4),x(5), ...
             cfg.parameterScale-x(4)-x(5)]));

    end

    h = max(h,1e-10);

    xp = x;
    xm = x;

    xp(k) = xp(k)+h;
    xm(k) = xm(k)-h;

    fp = fun(xp);
    fm = fun(xm);

    if ~isempty(fp) && ~isempty(fm)

        A(:,k) = (fp-fm)/(2*h);

    elseif ~isempty(fp)

        A(:,k) = (fp-f)/h;

    elseif ~isempty(fm)

        A(:,k) = (f-fm)/h;

    else

        A = [];
        return;

    end

end

end


function [x,ok,it] = newton_correct(fun,x,cfg)

x = x(:);
ok = false;

for it = 1:cfg.maxNewtonIterations

    f = fun(x);

    if isempty(f)
        return;
    end

    if norm(f,inf) <= cfg.newtonTolerance

        ok = true;
        return;

    end

    A = numerical_jacobian(fun,x,f,cfg);

    if isempty(A) || rcond(A) < 1e-13
        return;
    end

    step = -A\f;

    step = step * min( ...
        1,cfg.maxNewtonStep/max(norm(step,inf),eps));

    accepted = false;

    for ls = 1:20

        trial = x+step;

        ft = fun(trial);

        if ~isempty(ft) && ...
                norm(ft) < (1-1e-4)*norm(f)

            accepted = true;
            break;

        end

        step = step/2;

    end

    if ~accepted
        return;
    end

    x = trial;

end

f = fun(x);

ok = ~isempty(f) && ...
    norm(f,inf) <= cfg.newtonTolerance;

end


%% ============================================================
%  Pseudo-arclength continuation
% =============================================================

function t = boundary_tangent(x,mode,previous,cfg)

fun = @(z) critical_values(z,mode,cfg);

f = fun(x);
t = [];

if isempty(f)
    return;
end

A = numerical_jacobian(fun,x,f,cfg);

if isempty(A)
    return;
end

[~,S,V] = svd(A);

sv = diag(S);

if min(sv)/max(sv) < 1e-12
    return;
end

t = V(:,end);

if ~isempty(previous) && dot(t,previous) < 0
    t = -t;
end

end


function f = constrained_values(x,mode,row,target,cfg)

f = critical_values(x,mode,cfg);

if ~isempty(f)

    f = [f; dot(row,x)-target];

end

end


function [points,info] = trace_boundary( ...
    start,mode,direction,cfg)

x = start;

q = accept_critical(x,mode,cfg);

points = {q};

t = boundary_tangent(x,mode,[],cfg);

assert(~isempty(t), ...
    'The initial boundary tangent is singular.');

if mode == 3
    axisIndex = 5;
else
    axisIndex = 4;
end

if t(axisIndex) < 0
    t = -t;
end

t = direction*t;

ds = cfg.initialStep;

timer = tic;

reason = 'maximum_points';

lower = cfg.parameterScale * [
    max(cfg.minimumGamma,cfg.gamma1Limits(1));
    max(cfg.minimumGamma,cfg.gamma2Limits(1))
];

upper = cfg.parameterScale * [
    cfg.gamma1Limits(2);
    cfg.gamma2Limits(2)
];

for k = 1:cfg.maximumPoints

    % Reduce step size near a vanishing population.
    for a = 4:5

        if t(a) < 0

            ds = min(ds,0.2*x(a)/abs(t(a)));

        end

    end

    accepted = false;

    for retry = 1:cfg.maximumRetries

        prediction = x+ds*t;

        [candidate,ok,it] = newton_correct( ...
            @(z) constrained_values( ...
                z,mode,t,dot(t,prediction),cfg), ...
            prediction,cfg);

        if ok

            next = accept_critical(candidate,mode,cfg);

            if next.valid && ...
                    norm(candidate-prediction) <= 0.8*ds && ...
                    max(abs(next.H(:)-q.H(:))) <= cfg.maxShapeJump

                accepted = true;
                break;

            end

        end

        ds = ds/2;

        if ds < cfg.minimumStep
            break;
        end

    end

    if ~accepted

        reason = 'corrector_failed';
        break;

    end

    % Detect the first boundary of the parameter domain.
    fraction = inf;
    boundaryAxis = 0;
    bound = nan;

    for a = 1:2

        if candidate(a+3) < lower(a)

            v = lower(a);

        elseif candidate(a+3) > upper(a)

            v = upper(a);

        else

            continue;

        end

        part = (v-x(a+3))/ ...
            (candidate(a+3)-x(a+3));

        if part < fraction

            fraction = part;
            boundaryAxis = a+3;
            bound = v;

        end

    end

    if boundaryAxis > 0

        row = zeros(5,1);
        row(boundaryAxis) = 1;

        initial = x+fraction*(candidate-x);

        [ending,ok] = newton_correct( ...
            @(z) constrained_values( ...
                z,mode,row,bound,cfg), ...
            initial,cfg);

        last = accept_critical(ending,mode,cfg);

        if ok && last.valid

            points{end+1,1} = last; %#ok<AGROW>
            reason = 'domain_edge';

        else

            reason = 'edge_corrector_failed';

        end

        break;

    end

    points{end+1,1} = next; %#ok<AGROW>

    q = next;
    x = candidate;

    % Background stability:
    % mode 1 = cluster-position stability
    % mode 4 = C2 internal stability

    if any(q.mu([1,4]) > 10*cfg.stabilityTolerance)

        reason = 'background_unstable';
        break;

    end

    nextT = boundary_tangent(x,mode,t,cfg);

    if isempty(nextT)

        reason = 'singular_tangent';
        break;

    end

    turn = dot(t,nextT);
    t = nextT;

    if it <= 6 && turn > 0.98

        ds = min(cfg.maximumStep,1.25*ds);

    elseif it > 11 || turn < 0.9

        ds = max(cfg.minimumStep,0.7*ds);

    end

    if toc(timer) >= cfg.progressSeconds

        fprintf( ...
            '%s, direction %+d: %d points, gamma=(%.5f,%.5f)\n', ...
            mode_name(mode),direction,numel(points), ...
            q.g(2),q.g(3));

        drawnow;

        timer = tic;

    end

    if k > 30 && norm(x-start) < 0.5*ds

        reason = 'closed_curve';
        break;

    end

end

last = points{end};

info = struct( ...
    'mode',mode, ...
    'direction',direction, ...
    'points',numel(points), ...
    'gammaC1',last.g(2), ...
    'gammaC2',last.g(3), ...
    'stopReason',reason);

fprintf( ...
    '%s, direction %+d: %d points, %s\n', ...
    mode_name(mode),direction,numel(points),reason);

if ~any(strcmp(reason, ...
        {'domain_edge','background_unstable','closed_curve'}))

    warning( ...
        'Incomplete trace (%s). Inspect its endpoint.',reason);

end

end


function name = mode_name(mode)

switch mode

    case 2
        name = 'P internal';

    case 3
        name = 'C1 internal';

    otherwise
        name = 'background';

end

end


%% ============================================================
%  Organize traced curves
% =============================================================

function curve = pack_curve(roots,mode,cfg)

n = numel(roots);

xy = zeros(n,2);

bg = false(n,1);
eligible = false(n,1);

for k = 1:n

    q = roots{k};

    xy(k,:) = q.g(2:3).';

    bg(k) = all( ...
        q.mu([1,4]) <= cfg.stabilityTolerance);

    otherMode = setdiff([2,3],mode);

    eligible(k) = bg(k) && all( ...
        q.mu(otherMode) <= cfg.stabilityTolerance);

end

curve = struct( ...
    'mode',mode, ...
    'roots',{roots}, ...
    'xy',xy, ...
    'backgroundOK',bg, ...
    'eligible',eligible);

end


function T = pack_boundary_table(curves,cfg)

rows = zeros(0,17);

reasons = cell(0,1);
names = cell(0,1);

for c = 1:numel(curves)

    curve = curves(c);

    for k = 1:numel(curve.roots)

        q = curve.roots{k};

        rows(end+1,:) = [ ...
            c,k,curve.mode, ...
            q.g(2:3).', ...
            q.theta.', ...
            q.R,q.mu, ...
            q.rootResidual, ...
            q.rotationResidual, ...
            double(curve.backgroundOK(k)), ...
            double(curve.eligible(k))]; %#ok<AGROW>

        if q.mu(1) > cfg.stabilityTolerance

            reason = 'position_unstable';

        elseif q.mu(4) > cfg.stabilityTolerance

            reason = 'C2_internal_unstable';

        elseif ~curve.eligible(k)

            reason = 'other_target_cluster_unstable';

        else

            reason = 'solid_line';

        end

        reasons{end+1,1} = reason; %#ok<AGROW>

        names{end+1,1} = mode_name(curve.mode); %#ok<AGROW>

    end

end

T = array2table(rows, ...
    'VariableNames', ...
    {'curve','point','mode','gammaC1','gammaC2', ...
     'thetaP','thetaC1','thetaC2','R', ...
     'muPosition','muP','muC1','muC2', ...
     'rootResidual','rotationResidual', ...
     'backgroundOK','solidLine'});

T.boundaryName = names;
T.plotStatus = reasons;

end


%% ============================================================
%  Cross-validation against vector dynamics
% =============================================================

function T = validate_curves(curves,joint,initial,cfg)

roots = {joint};

labels = {'joint_point'};

for c = 1:numel(curves)

    allRoots = curves(c).roots;

    n = numel(allRoots);

    ids = unique(round( ...
        linspace(1,n,min(cfg.validationPoints,n))));

    for k = ids

        roots{end+1} = allRoots{k}; %#ok<AGROW>

        labels{end+1} = ...
            sprintf('curve_%d_point_%d',c,k); %#ok<AGROW>

    end

end

rows = zeros(numel(roots),9);

fprintf('\nChecking %d boundary points...\n',numel(roots));

for k = 1:numel(roots)

    rows(k,:) = cross_check(roots{k},cfg);

end

T = array2table(rows, ...
    'VariableNames',validation_names());

T.caseName = labels.';

T = movevars(T,'caseName','Before',1);

T = [initial;T];

end


function T = check_known_configurations(anchor,cfg)

cases = {anchor};
labels = {'anchor'};

% Coplanar configuration
g = [0.70;0.20;0.10];

phi = [0;2*pi/5;4*pi/5];

S = [
    sin(phi), ...
    zeros(3,1), ...
    cos(phi)
];

cases{end+1} = analyse_root( ...
    geometry_seed(S*S.',g),g,cfg);

labels{end+1} = 'coplanar_exact';

% Noncoplanar configuration
g = ones(3,1)/3;

az = [0;2*pi/3;4*pi/3];

th = 2*pi/5;

S = [
    sin(th)*cos(az), ...
    sin(th)*sin(az), ...
    cos(th)*ones(3,1)
];

cases{end+1} = analyse_root( ...
    geometry_seed(S*S.',g),g,cfg);

labels{end+1} = 'noncoplanar_exact';

rows = zeros(numel(cases),9);

for k = 1:numel(cases)

    assert(cases{k}.valid, ...
        'Exact-configuration geometry check failed.');

    rows(k,:) = cross_check(cases{k},cfg);

end

T = array2table(rows, ...
    'VariableNames',validation_names());

T.caseName = labels.';

T = movevars(T,'caseName','Before',1);

end


function names = validation_names()

names = { ...
    'gammaC1', ...
    'gammaC2', ...
    'coplanar', ...
    'jacobianRelativeError', ...
    'positionSpectrumRelativeError', ...
    'internalMatrixRelativeError', ...
    'rotationSubspaceError', ...
    'rootResidual', ...
    'passed'};

end


function row = cross_check(q,cfg)

% Complex-step check of the analytical Jacobian.
Dcs = zeros(3);

hcs = 1e-20;

for ell = 1:3

    th = q.theta;

    th(ell) = th(ell)+1i*hcs;

    f = angle_equations(th,q.g);

    Dcs(:,ell) = imag(f)/hcs;

end

errJ = norm(Dcs-q.D,inf) / ...
    max(1,norm(q.D,inf));

S = q.S;
u = q.u;

c = S*u.';

E = zeros(3,2,3);

for a = 1:3

    p = (u.'-c(a)*S(a,:).') / ...
        sqrt(max(1-c(a)^2,eps));

    E(:,:,a) = [
        p, cross(S(a,:).',p)
    ];

end

% Finite-difference Jacobian of the original vector model.
h = 2e-6;

L = zeros(6);
innerFD = zeros(2,2,3);

for b = 1:3

    for ell = 1:2

        sp = S;
        sm = S;

        sp(b,:) = sp(b,:)+h*E(:,ell,b).';
        sp(b,:) = sp(b,:)/norm(sp(b,:));

        sm(b,:) = sm(b,:)-h*E(:,ell,b).';
        sm(b,:) = sm(b,:)/norm(sm(b,:));

        df = ( ...
            vector_rhs(sp,q,false) - ...
            vector_rhs(sm,q,false))/(2*h);

        for a = 1:3

            L(2*a-1:2*a,2*b-2+ell) = ...
                E(:,:,a).'*df(a,:).';

        end

        df = ( ...
            vector_rhs(sp,q,true) - ...
            vector_rhs(sm,q,true))/(2*h);

        innerFD(:,ell,b) = ...
            E(:,:,b).'*df(b,:).';

    end

end

% Project out the three collective rotational modes.
Qrot = zeros(6,3);

axes3 = eye(3);

for a = 1:3

    for b = 1:3

        Qrot(2*a-1:2*a,b) = ...
            E(:,:,a).'*cross(axes3(:,b),S(a,:).');

    end

end

[Q,~] = qr(Qrot);

Qr = Q(:,1:3);
Qs = Q(:,4:6);

eshape = eig(Qs.'*L*Qs);

expected = q.eigAngleScaled;

if q.coplanar

    [~,k] = min( ...
        abs(expected-2*q.lambdaPerpScaled));

    expected(k) = expected(k)/2;

end

orders = perms(1:3);

errSpectrum = inf;

for k = 1:size(orders,1)

    e = eshape(orders(k,:));
    e = e(:);

    errSpectrum = min( ...
        errSpectrum,max(abs(expected-e)));

end

errSpectrum = errSpectrum / ...
    max(1,max(abs(expected)));

errInternal = ...
    max(abs(innerFD(:)-q.Mscaled(:))) / ...
    max(1,max(abs(q.Mscaled(:))));

errRotation = ...
    norm(Qs.'*L*Qr,inf) / ...
    max(1,norm(L,inf));

passed = errJ < 1e-8 && ...
    max([errSpectrum,errInternal,errRotation]) ...
    < cfg.validationTolerance;

row = [ ...
    q.g(2:3).', ...
    double(q.coplanar), ...
    errJ,errSpectrum,errInternal, ...
    errRotation,q.rootResidual,double(passed)];

end


function f = vector_rhs(S,q,fixedMean)

if fixedMean

    rho = q.g.'*q.S;

else

    rho = q.g.'*S;

end

R2 = sum(rho.^2);

d = S*rho.';

B = 16*d.^4-12*R2*d.^2+R2^2;

f = B.*(rho-d.*S)/q.R^5 ...
    -cross(repmat(q.OmegaScaled.',3,1),S,2);

end


%% ============================================================
%  Approximate boundaries and P-instability mask
% =============================================================

function [theory,T] = original_approximation( ...
    curves,joint,cfg)

% Original fixed-polar-angle approximations.
%
% Feedback:
%
% gamma_C2 = slope * gamma_C1
%
% P instability:
%
% gamma_C2 = intercept + pSlope * gamma_C1
%
% Record their analytical intersection; the feedback curve itself
% remains displayed across the full plotting interval.

c1 = cos(2*pi/5);
c2 = cos(4*pi/5);

r2 = sin(4*pi/5);

q = tan(pi/10);

slope = c1/abs(c2);

intercept = q/((1-c2)*q+r2);

pSlope = -intercept*(1-c1);

% Analytical intersection of the two approximations.
xCross = intercept/(slope-pSlope);
yCross = slope*xCross;

theory.jointGamma = joint.g(2:3).';

theory.approxIntersection = [xCross,yCross];

theory.PatJoint = ...
    intercept+pSlope*joint.g(2);

theory.jointMismatch = ...
    theory.PatJoint-joint.g(3);

% Include the analytical crossing explicitly in the mesh.
x = linspace( ...
    cfg.gamma1Limits(1), ...
    cfg.gamma1Limits(2), ...
    cfg.theorySamples).';

extra = [
    joint.g(2);
    xCross
];

extra = extra( ...
    extra >= cfg.gamma1Limits(1) & ...
    extra <= cfg.gamma1Limits(2));

theory.x = unique([x;extra]);

theory.P = ...
    intercept+pSlope*theory.x;

theory.feedback = ...
    slope*theory.x;

% Approximate P boundary valid on the chimera side.
theory.PLineMask = ...
    theory.x >= joint.g(2) & ...
    theory.P >= theory.feedback-1e-12;

%% Construct P-instability mask

theory.maskSections = {};
theory.sectionSource = [];

% First, use the exact three-cluster P boundary
% wherever it passes the stability filters.

for c = 1:numel(curves)

    if curves(c).mode ~= 2
        continue;
    end

    ids = find(curves(c).eligible);

    cuts = [0;find(diff(ids)>1);numel(ids)];

    for b = 1:numel(cuts)-1

        take = ids(cuts(b)+1:cuts(b+1));

        if numel(take) < 2
            continue;
        end

        xy = curves(c).xy(take,:);

        if all(diff(xy(:,1)) < 0)

            xy = flipud(xy);

        end

        if any(diff(xy(:,1)) <= 0)

            warning( ...
                'Nonmonotonic exact P branch was skipped.');

            continue;

        end

        % Analytical limit at gamma_C1 = 0.
        % Used for visualization only.

        if cfg.gamma1Limits(1) == 0 && ...
                abs(xy(1,1)-cfg.minimumGamma) < 1e-8

            yAxis = sin(pi/10) / ...
                (sin(pi/10)+sin(7*pi/10));

            xy = [0,yAxis;xy]; %#ok<AGROW>

        end

        theory.maskSections{end+1} = xy; %#ok<AGROW>

        theory.sectionSource(end+1) = 1; %#ok<AGROW>

    end

end

% Chimera side: the approximate P stability criterion.
%
% The use of max(P,feedback) follows the original
% conservative shading convention.

right = theory.x >= joint.g(2);

if nnz(right) >= 2

    xy = [
        theory.x(right), ...
        max(theory.P(right),theory.feedback(right))
    ];

    theory.maskSections{end+1} = xy;

    theory.sectionSource(end+1) = 3;

end

[threshold,source] = p_mask_cutoff( ...
    theory.x,theory,cfg);

theory.maskCutoff = threshold;
theory.maskSource = source;

%% Locate analytical feedback/P crossing (not a plot cutoff)

assert(xCross >= joint.g(2)-1e-10, ...
    'Feedback/P intersection is not on the chimera side.');

assert( ...
    xCross >= cfg.gamma1Limits(1) && ...
    xCross <= cfg.gamma1Limits(2) && ...
    yCross >= cfg.gamma2Limits(1) && ...
    yCross <= cfg.gamma2Limits(2), ...
    'Feedback/P intersection is outside the plotting range.');

% Check that the point lies on the actual P-mask boundary.
[maskAtEnd,~] = p_mask_cutoff( ...
    xCross,theory,cfg);

joinError = abs(maskAtEnd-yCross);

assert(isfinite(maskAtEnd) && joinError < 1e-8, ...
    'Blue feedback line does not meet the P mask.');

theory.feedbackEndpoint = [xCross,yCross];

theory.feedbackEndpointMaskY = maskAtEnd;

theory.feedbackEndpointGap = joinError;

% The feedback boundary is not terminated by P instability.
% Draw the complete theoretical line within the plotting range.

theory.feedbackLineMask = ...
    theory.feedback >= cfg.gamma2Limits(1) & ...
    theory.feedback <= cfg.gamma2Limits(2);

theory.feedbackPlotX = theory.x(theory.feedbackLineMask);
theory.feedbackPlotY = theory.feedback(theory.feedbackLineMask);

theory.note = [ ...
    'P mask: exact three-cluster boundary on the left; ', ...
    'original approximation on the chimera side. ', ...
    'The feedback/P intersection is a diagnostic; ', ...
    'the plotted blue line spans the full displayed x interval.'];

T = table( ...
    theory.x, ...
    theory.P, ...
    theory.feedback, ...
    threshold, ...
    source, ...
    theory.feedbackLineMask, ...
    'VariableNames', ...
    {'gammaC1','P_original','feedback_original', ...
     'P_mask_cutoff','maskSource', ...
     'feedbackBoundaryPlotted'});

fprintf('\nApproximate boundaries:\n');

fprintf('P: gamma_C2 = %.9f %+.9f*gamma_C1\n', ...
    intercept,pSlope);

fprintf('Feedback: gamma_C2 = %.9f*gamma_C1\n', ...
    slope);

fprintf('Intersection: (%.9f, %.9f)\n', ...
    xCross,yCross);

fprintf('P-mask endpoint mismatch: %.3e\n', ...
    joinError);

fprintf('Exact/approximate joint mismatch: %.6g\n', ...
    theory.jointMismatch);

end


%% ============================================================
%  Evaluate the P-mask cutoff
% =============================================================

function [threshold,source] = p_mask_cutoff(x,theory,cfg)

threshold = nan(size(x));

source = zeros(size(x));

% Approximate sections first, exact sections last.
% The exact P boundary takes precedence at a shared point.

order = [
    find(theory.sectionSource == 3), ...
    find(theory.sectionSource == 1)
];

for k = order

    xy = theory.maskSections{k};

    inside = ...
        x >= xy(1,1) & ...
        x <= xy(end,1);

    threshold(inside) = interp1( ...
        xy(:,1),xy(:,2),x(inside),'linear');

    source(inside) = theory.sectionSource(k);

    if theory.sectionSource(k) == 1

        source(inside & x < cfg.minimumGamma) = 2;

    else

        feedback = interp1( ...
            theory.x,theory.feedback,x, ...
            'linear',nan);

        p = interp1( ...
            theory.x,theory.P,x, ...
            'linear',nan);

        source(inside & feedback > p) = 4;

    end

end

end


%% ============================================================
%  Plot all theoretical overlays
% =============================================================

function draw_phase_overlays(ax,curves,theory,cfg)

if cfg.shadePUnstable

    shade_sections(ax,theory.maskSections,cfg);

end

% Exact three-cluster stability curves.
draw_curves(ax,curves,cfg,false);

% Approximate P stability boundary.
if cfg.showOldPApproximation

    y = theory.P;

    y(~theory.PLineMask) = nan;

    outlined_line( ...
        ax,theory.x,y,'--', ...
        cfg.oldPColor,cfg.lineWidth);

end

% Blue feedback boundary: drawn across the full displayed x interval.
if cfg.showOldFeedback

    outlined_line( ...
        ax, ...
        theory.feedbackPlotX, ...
        theory.feedbackPlotY, ...
        ':', ...
        cfg.feedbackColor, ...
        cfg.lineWidth);

end

%% Legend

hs = gobjects(0);
labels = {};

hs(end+1) = plot( ...
    ax,nan,nan,'-', ...
    'Color',cfg.colors(2,:), ...
    'LineWidth',cfg.lineWidth);

labels{end+1} = 'C_1 internal (three clusters)';

hs(end+1) = plot( ...
    ax,nan,nan,'-', ...
    'Color',cfg.colors(1,:), ...
    'LineWidth',cfg.lineWidth);

labels{end+1} = 'P internal (three clusters)';

if cfg.showOldPApproximation

    hs(end+1) = plot( ...
        ax,nan,nan,'--', ...
        'Color',cfg.oldPColor, ...
        'LineWidth',cfg.lineWidth);

    labels{end+1} = 'Chimera / P unstable (approx.)';

end

if cfg.showOldFeedback

    hs(end+1) = plot( ...
        ax,nan,nan,':', ...
        'Color',cfg.feedbackColor, ...
        'LineWidth',cfg.lineWidth);

    labels{end+1} = ...
        'Chimera / both rings incoherent (approx.)';

end

if cfg.shadePUnstable

    hs(end+1) = plot( ...
        ax,nan,nan,'s', ...
        'Color','k', ...
        'MarkerFaceColor','k', ...
        'MarkerSize',9);

    labels{end+1} = 'P unstable';

    x = cfg.gamma1Limits(1) ...
        +0.65*diff(cfg.gamma1Limits);

    y = cfg.gamma2Limits(1) ...
        +0.92*diff(cfg.gamma2Limits);

    cutoff = p_mask_cutoff(x,theory,cfg);

    if isfinite(cutoff) && y > cutoff

        text( ...
            ax,x,y,'P unstable', ...
            'Color','w', ...
            'FontSize',22, ...
            'FontName',cfg.fontName, ...
            'HorizontalAlignment','center');

    end

end

legend( ...
    ax,hs,labels, ...
    'Location','southoutside', ...
    'NumColumns',2, ...
    'FontSize',cfg.legendFontSize, ...
    'Box','off');

end


%% ============================================================
%  Draw P-unstable black regions
% =============================================================

function shade_sections(ax,sections,cfg)

if isempty(sections)
    return;
end

starts = cellfun(@(xy)xy(1,1),sections);

[~,order] = sort(starts);

combined = {};

current = sections{order(1)};

for k = order(2:end)

    next = sections{k};

    if abs(current(end,1)-next(1,1)) < 1e-12

        current = [current;next]; %#ok<AGROW>

    else

        combined{end+1} = current; %#ok<AGROW>

        current = next;

    end

end

combined{end+1} = current;

for k = 1:numel(combined)

    xy = combined{k};

    x = xy(:,1);

    low = min( ...
        cfg.gamma2Limits(2), ...
        max(cfg.gamma2Limits(1),xy(:,2)));

    top = cfg.gamma2Limits(2)*ones(size(low));

    patch( ...
        ax, ...
        [x;flipud(x)], ...
        [low;top], ...
        [0,0,0], ...
        'EdgeColor','none', ...
        'HandleVisibility','off');

end

end


%% ============================================================
%  Draw outlined theoretical curves
% =============================================================

function outlined_line(ax,x,y,style,color,width)

% White background line improves visibility on phase maps.

plot( ...
    ax,x,y,'-', ...
    'Color','w', ...
    'LineWidth',width+1.8, ...
    'HandleVisibility','off');

plot( ...
    ax,x,y,style, ...
    'Color',color, ...
    'LineWidth',width, ...
    'HandleVisibility','off');

end


function draw_curves(ax,curves,cfg,showLegend)

for c = 1:numel(curves)

    curve = curves(c);

    color = cfg.colors(curve.mode-1,:);

    if cfg.showFormalExtensions

        xy = curve.xy;

        xy(~curve.backgroundOK,:) = nan;

        plot( ...
            ax,xy(:,1),xy(:,2),'--', ...
            'Color',color, ...
            'LineWidth',1.3, ...
            'HandleVisibility','off');

    end

    xy = curve.xy;

    xy(~curve.eligible,:) = nan;

    outlined_line( ...
        ax,xy(:,1),xy(:,2),'-', ...
        color,cfg.lineWidth);

end

if showLegend

    h1 = plot( ...
        ax,nan,nan,'-', ...
        'Color',cfg.colors(2,:), ...
        'LineWidth',cfg.lineWidth);

    h2 = plot( ...
        ax,nan,nan,'-', ...
        'Color',cfg.colors(1,:), ...
        'LineWidth',cfg.lineWidth);

    legend( ...
        ax,[h1,h2], ...
        {'C_1 internal instability', ...
         'P internal instability'}, ...
        'Location','southoutside', ...
        'NumColumns',2, ...
        'FontSize',cfg.legendFontSize, ...
        'Box','off');

end

end


%% ============================================================
%  Figure axes
% =============================================================

function format_axes(ax,cfg)

set( ...
    ax, ...
    'YDir','normal', ...
    'XLim',cfg.gamma1Limits, ...
    'YLim',cfg.gamma2Limits, ...
    'XTick',0:0.1:0.5, ...
    'YTick',0:0.05:0.35, ...
    'DataAspectRatio',[1,1,1], ...
    'FontName',cfg.fontName, ...
    'FontSize',cfg.fontSize, ...
    'LineWidth',1.2, ...
    'TickDir','out', ...
    'Layer','top', ...
    'Color',[0.88,0.88,0.88]);

xlabel( ...
    ax,'\gamma_{C_1}', ...
    'Interpreter','tex');

ylabel( ...
    ax,'\gamma_{C_2}', ...
    'Interpreter','tex');

box(ax,'on');

end


%% ============================================================
%  Overlay onto the rho-axis phase diagrams
% =============================================================

function overlay_phase_file(curves,theory,cfg)

verify_rho_phase_file(cfg.phaseMatFile,cfg);

d = load( ...
    cfg.phaseMatFile,'results','simulationCfg');

r = d.results;

gamma1 = r.gamma1;
gamma2 = r.gamma2;

[X,Y] = meshgrid(gamma1,gamma2);

[Pcutoff,maskSource] = p_mask_cutoff( ...
    X,theory,cfg);

Pmask = isfinite(Pcutoff) & Y>Pcutoff;

save( ...
    fullfile(cfg.outputDir,'P_mask_for_phase.mat'), ...
    'gamma1','gamma2', ...
    'Pmask','Pcutoff','maskSource','-v7');

for group = 1:2

    field = sprintf('Z%dmean',group);

    values = r.(field);

    assert( ...
        isequal(size(values), ...
        [numel(gamma2),numel(gamma1)]), ...
        'Unexpected dimensions of %s',field);

    f = figure( ...
        'Color','w', ...
        'Position',[100+30*group,60,1080,880]);

    ax = axes('Parent',f);

    h = imagesc( ...
        ax,gamma1,gamma2,values,[0,1]);

    % Undefined order parameters remain transparent.
    set(h,'AlphaData',isfinite(values));

    colormap(ax,phase_colormap());

    format_axes(ax,cfg);

    hold(ax,'on');

    draw_phase_overlays(ax,curves,theory,cfg);

    cb = colorbar(ax);

    cb.FontSize = 20;

    cb.Label.Interpreter = 'latex';

    cb.Label.String = sprintf( ...
        '$\\langle |Z_{C_%d}|\\rangle_t$',group);

    cb.Label.FontSize = cfg.fontSize;

    labelColor = 'k';

    if cfg.shadePUnstable
        labelColor = 'w';
    end

    text( ...
        ax,0.97,0.97, ...
        sprintf('C_%d',group), ...
        'Units','normalized', ...
        'HorizontalAlignment','right', ...
        'VerticalAlignment','top', ...
        'FontSize',cfg.fontSize, ...
        'Color',labelColor);

    save_figure( ...
        f,sprintf('C%d_with_angle_boundaries',group),cfg);

end

fprintf('\nPhase overlays read from:\n%s\n', ...
    cfg.phaseMatFile);

end


%% ============================================================
%  Save figures
% =============================================================

function save_figure(f,name,cfg)

if ~cfg.saveFigures
    return;
end

set(f,'PaperPositionMode','auto');

savefig( ...
    f,fullfile(cfg.outputDir,[name,'.fig']));

print( ...
    f,fullfile(cfg.outputDir,[name,'.png']), ...
    '-dpng','-r250');

if cfg.exportPDF

    if exist('exportgraphics','file') ~= 0

        exportgraphics( ...
            f,fullfile(cfg.outputDir,[name,'.pdf']), ...
            'ContentType','vector');

    else

        print( ...
            f,fullfile(cfg.outputDir,[name,'.pdf']), ...
            '-dpdf','-painters');

    end

end

end


%% ============================================================
%  Verify the input rho-axis phase map
% =============================================================

function verify_rho_phase_file(file,cfg)

if ~cfg.requireRhoPhase
    return;
end

meta = load(file,'simulationCfg','results');

assert(isfield(meta,'simulationCfg'), ...
    'Missing simulation configuration.');

assert( ...
    isfield(meta.simulationCfg,'checkVersion') && ...
    isfield(meta.simulationCfg,'engineVersion'), ...
    'Missing reference-axis version metadata.');

assert( ...
    meta.simulationCfg.checkVersion == 3 && ...
    meta.simulationCfg.engineVersion == 5, ...
    ['Expected rho-axis data: ', ...
     'checkVersion=3 and engineVersion=5.']);

assert(isfield(meta,'results'), ...
    'Missing simulation results.');

assert( ...
    all(isfield(meta.results, ...
    {'gamma1','gamma2','Z1mean','Z2mean'})), ...
    'Missing phase grid or Z values.');

assert( ...
    meta.simulationCfg.K == cfg.K && ...
    meta.simulationCfg.n == 5 && ...
    meta.simulationCfg.Delta == 0, ...
    'Theory and phase-map model parameters disagree.');

end


%% ============================================================
%  Phase-diagram colormap
% =============================================================

function map = phase_colormap()

% Same sequential colormap as the rho-axis simulation.

anchors = [
    0.267,0.005,0.329;
    0.283,0.141,0.458;
    0.254,0.265,0.530;
    0.207,0.372,0.553;
    0.164,0.471,0.558;
    0.128,0.567,0.551;
    0.135,0.659,0.518;
    0.267,0.749,0.441;
    0.478,0.821,0.318;
    0.741,0.873,0.150;
    0.993,0.906,0.144
];

map = interp1( ...
    linspace(0,1,size(anchors,1)), ...
    anchors, ...
    linspace(0,1,256), ...
    'linear');

end
